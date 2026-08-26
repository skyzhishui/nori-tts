"""
TTS 推理引擎 — 直接对接 GPT_SoVITS

去除 Download/AudioQueue/字幕等非必要逻辑，
仅保留 WS 流式合成服务所需的推理管线。

依赖：
  - config.py（设备检测 + Config/GlobalConfig）
  - lang_segment.py（多语言识别与分段）
  - src/（模型定义 + G2P + Featurizer + SV）
"""

from __future__ import annotations

import gc
import os
import re
import asyncio
import threading
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import av
import torch
import torchaudio
import numpy as np
import logging
from torch.nn import functional as F
from typing import Literal
import pysbd
from safetensors.torch import load_model

# ---- 本地模块 ----
from config import Config, global_config
from lang_segment import LangSegment
from src.Featurizer import CNHubert, CNRoberta
from src.SV import ERes2Net
from src.G2P import text_to_phonemes, phonemes_to_ids, Pause
from src.SoVITS.models import SynthesizerTrn
from src.GPT.t2s_model import Text2SemanticDecoder
from src import utils as gsv_utils

logger = logging.getLogger("tts_engine")


# ===========================================================================
# 文本处理（精简：删除字幕相关函数）
# ===========================================================================

_pysbd_segmenter = pysbd.Segmenter()


def _get_semantic_length(text, en_weight=1.75):
    cjk_count = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fa5\uff66-\uff9f]', text))
    en_count = len(re.findall(r'[a-zA-Z0-9]+', text))
    return cjk_count + (en_count * en_weight)


def cut_text(text, cut_minlen=10):
    """按语义长度切分长文本"""
    sentences = _pysbd_segmenter.segment(text)

    for i in text:
        if i == '\n':
            sentences[0] = '\n' + sentences[0]
        else:
            break

    text_cuts = []
    punds_pattern = r'([，,；;：:、~・…]+|[\.。]{2,})'

    clauses = []
    for sentence in sentences:
        parts = re.split(punds_pattern, sentence)
        for i in range(0, len(parts) - 1, 2):
            clause = parts[i] + parts[i + 1]
            clauses.append(clause)
        if len(parts) % 2 != 0 and parts[-1]:
            clauses.append(parts[-1])

    current_segment = ""
    for s in clauses:
        current_segment += s
        if _get_semantic_length(current_segment) >= cut_minlen:
            text_cuts.append(current_segment)
            current_segment = ""

    if current_segment:
        if text_cuts:
            text_cuts[-1] += current_segment
        else:
            text_cuts.append(current_segment)

    for i in range(1, len(text_cuts)):
        while text_cuts[i][0] in ['!', '！', '?', '？', '.', '。']:
            text_cuts[i - 1] += text_cuts[i][0]
            text_cuts[i] = text_cuts[i][1:]

    return text_cuts


def get_phones_and_bert(texts, tts_config: Config):
    """
    将文本转为音素 ID + BERT embedding。
    单条文本时返回 (phones, word2ph, bert, norm_text)，
    批量时返回列表。
    """
    is_batch = True
    if isinstance(texts, str):
        texts = [texts]
        is_batch = False

    batch_phones = []
    batch_word2ph = []
    batch_bert = []
    batch_norm_text = []

    bert_tasks = {"pos": [], "word2ph": []}

    for text in texts:
        segments = LangSegment.getTexts(text)

        if not segments:
            raise ValueError(
                f"Text processing produced no valid segments for input: {repr(text)}. "
                "Please ensure the input text is not empty and contains valid characters."
            )

        phones_list = []
        norm_text_list = []
        word2ph = {"word": [], "ph": []}
        batch_bert.append([])

        for segment in segments:
            phones_raw, _word2ph, norm_text = text_to_phonemes(segment['text'], segment['lang'])
            phones = phonemes_to_ids(phones_raw)

            word2ph["word"] += _word2ph["word"]
            word2ph["ph"] += _word2ph["ph"]
            if tts_config.cnroberta and segment['lang'] == "zh":
                bert_tasks["pos"].append((len(batch_bert) - 1, len(batch_bert[-1])))
                bert_tasks["word2ph"].append(_word2ph)
                batch_bert[-1].append(None)
            else:
                batch_bert[-1].append(torch.zeros((len(phones), 1024), dtype=tts_config.dtype, device=tts_config.device))

            phones_list.append(phones)
            norm_text_list.append(norm_text)

        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)

        batch_phones.append(phones)
        batch_word2ph.append(word2ph)
        batch_norm_text.append(norm_text)

    if bert_tasks["word2ph"]:
        berts = tts_config.cnroberta(bert_tasks["word2ph"])
        for (i, j), bert in zip(bert_tasks["pos"], berts):
            batch_bert[i][j] = bert

    processed_batch_bert = []
    for bert_tensors in batch_bert:
        processed_batch_bert.append(torch.cat(bert_tensors))
    batch_bert = processed_batch_bert

    if is_batch:
        return batch_phones, batch_word2ph, batch_bert, batch_norm_text
    else:
        return batch_phones[0], batch_word2ph[0], batch_bert[0], batch_norm_text[0]


# ===========================================================================
# 模型加载
# ===========================================================================

# 为兼容旧 checkpoint 中 `import utils` 的写法
import sys as _sys
_sys.modules['utils'] = gsv_utils

head2version = {
    b"01": "v2",
    b"05": "v2Pro",
    b"06": "v2ProPlus",
}
hash_pretrained_dict = {
    "dc3c97e17592963677a4a1681f30c653": "v2",
    "6642b37f3dbb1f76882b69937c95a5f3": "v2",
    "c7e9fce2223f3db685cdfa1e6368728a": "v2Pro",
    "66b313e39455b57ab1b0bc0b239c9d0a": "v2ProPlus",
}


class Sovits:
    """SoVITS 模型包装"""
    def __init__(self, vq_model: SynthesizerTrn, hps):
        self.vq_model = vq_model
        self.hps = hps


class Gpt:
    """GPT 模型包装"""
    def __init__(self, t2s_model: Text2SemanticDecoder, config):
        self.t2s_model = t2s_model
        self.config = config


def _get_hash_from_file(sovits_path):
    with open(sovits_path, "rb") as f:
        data = f.read(8192)
    hash_md5 = hashlib.md5()
    hash_md5.update(data)
    return hash_md5.hexdigest()


def _load_sovits(sovits_path):
    hash_val = _get_hash_from_file(sovits_path)
    f = open(sovits_path, "rb")
    meta = f.read(2)
    version = head2version.get(meta)
    if version is None:
        version = hash_pretrained_dict.get(hash_val)
    if meta != b"PK":
        data = b"PK" + f.read()
        bio = BytesIO()
        bio.write(data)
        bio.seek(0)
        return torch.load(bio, map_location="cpu", weights_only=False), version
    return torch.load(sovits_path, map_location="cpu", weights_only=False), version


def get_sovits_weights(sovits_path, tts_config: Config) -> Sovits:
    """加载 SoVITS 模型权重"""
    if os.path.isdir(sovits_path):
        with open(os.path.join(sovits_path, "hps.json"), "r") as f:
            hps = json.load(f)
        hps = gsv_utils.DictToAttrRecursive(hps)

        with torch.device("meta"):
            vq_model = SynthesizerTrn(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                **vars(hps.model),
            )
        vq_model.dec.remove_weight_norm()
        vq_model = vq_model.to_empty(device=tts_config.device)
        vq_model = vq_model.to(tts_config.dtype)
        load_model(vq_model, os.path.join(sovits_path, "model.safetensors"))
    else:
        dict_s2, version = _load_sovits(sovits_path)
        hps = gsv_utils.DictToAttrRecursive(dict_s2["config"])
        hps.model.semantic_frame_rate = "25hz"
        if version is None:
            assert getattr(hps.model, 'version', None) in ["v2", "v2Pro", "v2ProPlus"], \
                "The Sovits model is not the v2/v2pro/v2proplus version."
        else:
            hps.model.version = version

        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **vars(hps.model),
        )
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.dec.remove_weight_norm()
        vq_model.to(tts_config.device, tts_config.dtype)

    vq_model.eval()
    vq_model.initialize_runtime(tts_config.dtype, tts_config.device, tts_config.sovits_cache)
    return Sovits(vq_model, hps)


def get_gpt_weights(gpt_path, tts_config: Config) -> Gpt:
    """加载 GPT 模型权重"""
    if os.path.isdir(gpt_path):
        with open(os.path.join(gpt_path, "config.json"), "r") as f:
            config = json.load(f)

        with torch.device("meta"):
            if tts_config.use_flash_attn:
                from src.GPT.t2s_model_flash_attn import Text2SemanticDecoder as Text2SemanticDecoder_flash_attn
                t2s_model = Text2SemanticDecoder_flash_attn(config)
            else:
                t2s_model = Text2SemanticDecoder(config)

        t2s_model = t2s_model.to_empty(device=tts_config.device)
        t2s_model = t2s_model.to(tts_config.dtype)
        load_model(t2s_model, os.path.join(gpt_path, "model.safetensors"))
    else:
        dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
        config = dict_s1["config"]

        w_key_map = [
            ['self_attn.in_proj_weight', 'qkv.weight'],
            ['self_attn.in_proj_bias', 'qkv.bias'],
            ['self_attn.out_proj.weight', 'out_proj.weight'],
            ['self_attn.out_proj.bias', 'out_proj.bias'],
            ['linear1.weight', 'mlp.0.weight'],
            ['linear1.bias', 'mlp.0.bias'],
            ['linear2.weight', 'mlp.2.weight'],
            ['linear2.bias', 'mlp.2.bias'],
            ['norm1.weight', 'norm1.weight'],
            ['norm1.bias', 'norm1.bias'],
            ['norm2.weight', 'norm2.weight'],
            ['norm2.bias', 'norm2.bias'],
        ]

        for i in range(config["model"]["n_layer"]):
            original_l_key = f'model.h.layers.{i}.'
            new_l_key = f't2s_transformer.blocks.{i}.'
            for original_w_key, new_w_key in w_key_map:
                dict_s1["weight"][new_l_key + new_w_key] = dict_s1["weight"].pop(original_l_key + original_w_key)

        dict_s1["weight"] = {
            k.replace("model.", "", 1) if k.startswith("model.") else k: v
            for k, v in dict_s1["weight"].items()
        }

        if tts_config.use_flash_attn:
            from src.GPT.t2s_model_flash_attn import Text2SemanticDecoder as Text2SemanticDecoder_flash_attn
            t2s_model = Text2SemanticDecoder_flash_attn(config)
        else:
            t2s_model = Text2SemanticDecoder(config)

        t2s_model.load_state_dict(dict_s1["weight"])
        t2s_model = t2s_model.to(tts_config.device, tts_config.dtype)

    t2s_model.eval()
    t2s_model.initialize_runtime(tts_config.dtype, tts_config.device, tts_config.gpt_cache)
    return Gpt(t2s_model, config)


# ===========================================================================
# TTSEngine — 核心推理引擎
# ===========================================================================

class TTSEngine:
    """
    TTS 流式推理引擎，直接对接 GPT_SoVITS。
    管理 GPT/SoVITS 模型生命周期、参考音频缓存、流式推理管线。
    """

    def __init__(
        self,
        models_dir: str,
        gpt_cache: list[tuple[int, int]] = [(1, 512), (1, 768), (1, 1024), (4, 512), (4, 1024)],
        sovits_cache: list[int] = [50],
        device: str = None,
        dtype: str = None,
        use_flash_attn: bool = False,
        auto_bert: bool = True,
        always_load_cnhubert: bool = False,
        always_load_sv: bool = False,
    ):
        self.tts_config = Config()

        if device is not None:
            self.tts_config.device = torch.device(device)
            if self.tts_config.device.type in ["mps", "cpu"]:
                self.tts_config.dtype = torch.float32
                sovits_cache = []
        if dtype is not None:
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            self.tts_config.dtype = dtype_map.get(dtype.lower(), torch.float32)

        self.always_load_cnhubert = always_load_cnhubert
        self.always_load_sv = always_load_sv
        self.auto_bert = auto_bert

        self.models_dir = models_dir
        if global_config.models_dir is None:
            global_config.models_dir = models_dir
        # Auto-download missing auxiliary models at startup
        from model_downloader import ensure_models
        ensure_models(self.models_dir)

        self.tts_config.use_flash_attn = use_flash_attn
        self.tts_config.gpt_cache = gpt_cache
        self.tts_config.sovits_cache = sovits_cache

        # 模型存储
        self.gpt_models: dict[str, Gpt] = {}
        self.sovits_models: dict[str, Sovits] = {}

        # 音频缓存（带LRU淘汰机制，防止内存泄漏）
        self.max_cache_size = 50  # 缓存上限
        self.spk_audio_cache = {}
        self.prompt_audio_cache = {}

        # 重采样/频谱变换缓存
        self.resample_transform_dict = {}
        self.spectrogram_transform_dict = {}

        # 辅助模型（懒加载）
        self.cnhubert_model = None
        self.sv_model = None
        self._bert_loaded = False

        # 模型默认路径
        self.cnhubert_path = Path(self.models_dir) / "chinese-hubert-base"
        self.cnroberta_path = Path(self.models_dir) / "chinese-roberta-wwm-ext-large"
        self.sv_path = Path(self.models_dir) / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"
        self.default_gpt_path = Path(self.models_dir) / "s1v3.ckpt"
        self.default_sovits_path = Path(self.models_dir) / "s2Gv2ProPlus.pth"

        # 常量
        self.punctuation = tuple(Pause.pause_map.keys())
        self.samplerate = 32000
        self.gpt_hz = 25
        self.sovits_hz = 50

        # 推理锁（保证同一时刻只有一个推理线程）
        self._infer_lock = threading.Lock()

        # 缓存访问顺序记录（用于LRU淘汰）
        self._spk_cache_access_order = []
        self._prompt_cache_access_order = []

        logger.info(f"Device: {self.tts_config.device}, dtype: {self.tts_config.dtype}")

    # ---- 模型加载 ----

    def load_gpt_model(self, *model_paths: str):
        if not model_paths:
            model_paths = (str(self.default_gpt_path),)
        for model_path in model_paths:
            self.gpt_models[model_path] = get_gpt_weights(model_path, self.tts_config)
            logger.info(f'Loaded GPT model: {model_path}')

    def load_sovits_model(self, *model_paths: str):
        if not model_paths:
            model_paths = (str(self.default_sovits_path),)
        for model_path in model_paths:
            self.sovits_models[model_path] = get_sovits_weights(model_path, self.tts_config)
            logger.info(f'Loaded SoVITS model: {model_path}')

    # ---- 参考音频缓存 ----

    @torch.inference_mode()
    def cache_spk_audio(self, *spk_audio_paths: str, sovits_model: str = None):
        """缓存说话人音频的 speaker embedding（带LRU淘汰，防止内存泄漏）"""
        try:
            if not self.sovits_models:
                logger.error('No SoVITS models loaded! Cannot cache speaker audio.')
                return

            if sovits_model is None:
                sovits_model = next(iter(self.sovits_models))
            if sovits_model not in self.sovits_models:
                logger.error(f'SoVITS model {sovits_model} not loaded!')
                return

            model = self.sovits_models[sovits_model]

            if self.sv_model is None:
                self.sv_model = ERes2Net(self.sv_path, self.tts_config)

            for spk_audio_path in spk_audio_paths:
                refers, audio_tensor = self._get_spec(model.hps, spk_audio_path)
                if spk_audio_path not in self.spk_audio_cache:
                    # LRU淘汰：超过上限时删除最久未访问的条目
                    self._evict_cache(self.spk_audio_cache, self._spk_cache_access_order)
                    sv_emb = self.sv_model.compute_embedding3(audio_tensor)
                    ge = model.vq_model.get_ge(refers, sv_emb)
                    self.spk_audio_cache[spk_audio_path] = {
                        "ge": {sovits_model: ge},
                        "sv_emb": sv_emb,
                    }
                elif sovits_model not in self.spk_audio_cache[spk_audio_path]["ge"]:
                    ge = model.vq_model.get_ge(refers, self.spk_audio_cache[spk_audio_path]["sv_emb"])
                    self.spk_audio_cache[spk_audio_path]["ge"][sovits_model] = ge
                # 更新访问顺序
                self._touch_cache_key(self._spk_cache_access_order, spk_audio_path)
                logger.info(f'Cached speaker audio: {spk_audio_path}')

            if not self.always_load_sv:
                self.sv_model = None
        finally:
            self._empty_cache()

    @torch.inference_mode()
    def cache_prompt_audio(self, prompt_audio_paths: str | list[str], prompt_audio_texts: str | list[str]):
        """缓存提示音频的 BERT/phoneme/prompt 特征（带LRU淘汰，防止内存泄漏）"""
        try:
            if not self.sovits_models:
                logger.error('No SoVITS models loaded! Cannot cache prompt audio.')
                return

            model = self.sovits_models[next(iter(self.sovits_models))]

            if self.cnhubert_model is None:
                self.cnhubert_model = CNHubert(self.cnhubert_path, self.tts_config)

            if isinstance(prompt_audio_paths, str):
                prompt_audio_paths = [prompt_audio_paths]
            if isinstance(prompt_audio_texts, str):
                prompt_audio_texts = [prompt_audio_texts] * len(prompt_audio_paths)

            for prompt_audio_path, prompt_audio_text in zip(prompt_audio_paths, prompt_audio_texts):
                if not prompt_audio_text or not prompt_audio_text.strip():
                    raise ValueError(
                        "Prompt audio text is empty. "
                        "Please provide the text transcription for the reference audio."
                    )
                # LRU淘汰：超过上限时删除最久未访问的条目
                self._evict_cache(self.prompt_audio_cache, self._prompt_cache_access_order)
                prompt = self._get_prompt(self.cnhubert_model, model, prompt_audio_path)
                phones1, _, bert1, _ = get_phones_and_bert(prompt_audio_text, self.tts_config)
                self.prompt_audio_cache[prompt_audio_path] = {
                    "prompt": prompt,
                    "phones1": phones1,
                    "bert1": bert1,
                }
                # 更新访问顺序
                self._touch_cache_key(self._prompt_cache_access_order, prompt_audio_path)
                logger.info(f'Cached prompt audio: {prompt_audio_path}')

            if not self.always_load_cnhubert:
                self.cnhubert_model = None
        finally:
            self._empty_cache()

    # ---- 流式推理 ----

    @torch.inference_mode()
    def infer_stream(
        self,
        spk_audio_path: str | dict,
        prompt_audio_path: str,
        prompt_audio_text: str,
        text: str,
        is_cut_text: bool = True,
        cut_minlen: int = 10,
        cut_mute: float = 0.4,
        cut_mute_scale_map: dict = None,
        stream_mode: Literal["token", "sentence"] = "token",
        stream_chunk: int = 25,
        overlap_len: int = 5,
        boost_first_chunk: bool = True,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        noise_scale: float = 0.5,
        speed: float = 1.0,
        gpt_model: str = None,
        sovits_model: str = None,
    ):
        """
        同步流式推理生成器，每次 yield (audio_data: np.ndarray, samplerate: int)。
        audio_data 为 float32 numpy 数组，samplerate 固定 32000。
        """
        if cut_mute_scale_map is None:
            cut_mute_scale_map = {
                "…": 2.0, ".": 1.5, "。": 1.5, "?": 1.5, "？": 1.5,
                "!": 1.5, "！": 1.5, ",": 1.0, "，": 1.0, ":": 1.0,
                "：": 1.0, ";": 1.0, "；": 1.0, "~": 1.0, "、": 0.8, "・": 0.8,
            }

        try:
            if self._contains_chinese(text):
                self._ensure_bert_loaded()

            if not self._check_pause(text):
                text += "."

            if len(text) > 20:
                logger.info(f"Starting Stream inference for text: '{text[:20]}...'")
            else:
                logger.info(f"Starting Stream inference for text: '{text}'")

            if stream_mode == "sentence":
                stream_chunk = 10000
            if not is_cut_text:
                cut_minlen = 10000
            cut_mute = cut_mute / speed

            if gpt_model is None:
                if len(self.gpt_models) > 0:
                    gpt_model = list(self.gpt_models.keys())[0]
                else:
                    gpt_model = str(self.default_gpt_path)
            if sovits_model is None:
                if len(self.sovits_models) > 0:
                    sovits_model = list(self.sovits_models.keys())[0]
                else:
                    sovits_model = str(self.default_sovits_path)

            logger.info(f"Using GPT model: {gpt_model}")
            logger.info(f"Using SoVITS model: {sovits_model}")

            sovits, ge = self._prepare_sovits_resources(sovits_model, spk_audio_path)
            gpt, prompt, phones1, bert1 = self._prepare_gpt_resources(gpt_model, prompt_audio_path, prompt_audio_text)
            t2s_model = gpt.t2s_model
            vq_model = sovits.vq_model

            overlap_samples = overlap_len * vq_model.samples_per_frame

            text_cuts = cut_text(text, cut_minlen)
            for i, text_cut in enumerate(text_cuts):
                logger.info(f"Processing segment {i+1}/{len(text_cuts)}: '{text_cut}'")

                phones2, word2ph, bert2, norm_text = get_phones_and_bert(text_cut, self.tts_config)

                curr_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.tts_config.device).unsqueeze(0)
                curr_bert = torch.cat([bert1, bert2]).unsqueeze(0)

                generator = t2s_model.infer_stream(
                    curr_phoneme_ids,
                    prompt,
                    curr_bert,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    stream_chunk=stream_chunk,
                    boost_first_chunk=boost_first_chunk if i == 0 else False,
                )

                phones2_tensor = torch.LongTensor(phones2).to(self.tts_config.device).unsqueeze(0)

                last_overlap_audio = None
                valid_start_idx = 0
                chunk_idx = 0
                for pred_semantic, is_final in generator:
                    audio, attn = vq_model.decode(
                        pred_semantic,
                        phones2_tensor,
                        ge,
                        noise_scale=noise_scale,
                        speed=speed,
                        stream_mode=True,
                        valid_start_idx=valid_start_idx,
                        overlap_len=overlap_len,
                    )

                    if last_overlap_audio is not None:
                        audio, offset = self._sola_algorithm(last_overlap_audio, audio, overlap_samples)
                    last_overlap_audio = audio[:, :, -overlap_samples:].clone()

                    if not is_final:
                        audio = audio[:, :, :-overlap_samples]
                        valid_start_idx = attn.shape[1]

                    audio = audio[0, 0, :]

                    # 去除开头静音
                    if chunk_idx == 0:
                        head_offset = self._find_head_threshold_offsets(audio)
                        audio = audio[head_offset:]

                    # 句尾静音
                    if is_final:
                        if text_cut[-1] in cut_mute_scale_map:
                            cut_mute_scale = cut_mute_scale_map[text_cut[-1]]
                        elif "…" in cut_mute_scale_map and text_cut[-3:] in ["...", "。。。"]:
                            cut_mute_scale = cut_mute_scale_map["…"]
                        else:
                            cut_mute_scale = 1.0
                        silence = torch.zeros(
                            (int(cut_mute * cut_mute_scale * self.samplerate),),
                            dtype=audio.dtype, device=audio.device,
                        )
                        audio = torch.concatenate([audio, silence])

                    audio = audio.float().cpu().numpy()
                    yield audio, self.samplerate

                    chunk_idx += 1

                vq_model.enc_p.y_overlap = None

            logger.info(f"Stream inference complete.")

        finally:
            self._empty_cache()

    async def infer_stream_async(
        self,
        spk_audio_path: str | dict,
        prompt_audio_path: str,
        prompt_audio_text: str,
        text: str,
        is_cut_text: bool = True,
        cut_minlen: int = 10,
        cut_mute: float = 0.4,
        cut_mute_scale_map: dict = None,
        stream_mode: Literal["token", "sentence"] = "token",
        stream_chunk: int = 25,
        overlap_len: int = 5,
        boost_first_chunk: bool = True,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        noise_scale: float = 0.5,
        speed: float = 1.0,
        gpt_model: str = None,
        sovits_model: str = None,
        executor: ThreadPoolExecutor = None,
    ):
        """异步流式推理包装器，在线程池中运行同步 infer_stream"""
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        def _stream_wrapper():
            exc = None
            try:
                with self._infer_lock:
                    for audio_data, samplerate in self.infer_stream(
                        spk_audio_path=spk_audio_path,
                        prompt_audio_path=prompt_audio_path,
                        prompt_audio_text=prompt_audio_text,
                        text=text,
                        is_cut_text=is_cut_text,
                        cut_minlen=cut_minlen,
                        cut_mute=cut_mute,
                        cut_mute_scale_map=cut_mute_scale_map,
                        stream_mode=stream_mode,
                        stream_chunk=stream_chunk,
                        overlap_len=overlap_len,
                        boost_first_chunk=boost_first_chunk,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        noise_scale=noise_scale,
                        speed=speed,
                        gpt_model=gpt_model,
                        sovits_model=sovits_model,
                    ):
                        loop.call_soon_threadsafe(queue.put_nowait, (audio_data, samplerate))
            except Exception as e:
                # 线程内异常不可静默丢失，传递给消费端
                exc = e
                logger.exception("infer_stream 线程执行异常")
            finally:
                if exc is not None:
                    loop.call_soon_threadsafe(queue.put_nowait, ("__error__", exc))
                else:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

        if executor is None:
            loop.run_in_executor(None, _stream_wrapper)
        else:
            loop.run_in_executor(executor, _stream_wrapper)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            marker, exc = chunk[0], chunk[1]
            if marker == "__error__":
                raise RuntimeError(f"流式推理线程异常: {exc}") from exc
            audio_data, samplerate = chunk
            yield chunk

    # ---- 内部辅助方法 ----

    def _prepare_gpt_resources(self, gpt_model, prompt_audio_path, prompt_audio_text):
        if gpt_model not in self.gpt_models:
            self.load_gpt_model(gpt_model)
        if prompt_audio_path not in self.prompt_audio_cache:
            self.cache_prompt_audio(prompt_audio_paths=prompt_audio_path, prompt_audio_texts=prompt_audio_text)
        else:
            # 缓存命中，更新LRU访问顺序
            self._touch_cache_key(self._prompt_cache_access_order, prompt_audio_path)
        prompt = self.prompt_audio_cache[prompt_audio_path]["prompt"]
        phones1 = self.prompt_audio_cache[prompt_audio_path]["phones1"]
        bert1 = self.prompt_audio_cache[prompt_audio_path]["bert1"]
        gpt = self.gpt_models[gpt_model]
        return gpt, prompt, phones1, bert1

    def _prepare_sovits_resources(self, sovits_model, spk_audio_path):
        if sovits_model not in self.sovits_models:
            self.load_sovits_model(sovits_model)
        if isinstance(spk_audio_path, dict):
            weight_sum = sum(spk_audio_path.values())
            ge = None
            for audio_path, weight in spk_audio_path.items():
                if (audio_path not in self.spk_audio_cache) or (sovits_model not in self.spk_audio_cache[audio_path]["ge"]):
                    self.cache_spk_audio(audio_path, sovits_model=sovits_model)
                else:
                    # 缓存命中，更新LRU访问顺序
                    self._touch_cache_key(self._spk_cache_access_order, audio_path)
                if ge is None:
                    ge = self.spk_audio_cache[audio_path]["ge"][sovits_model] * (weight / weight_sum)
                else:
                    ge += self.spk_audio_cache[audio_path]["ge"][sovits_model] * (weight / weight_sum)
        else:
            if (spk_audio_path not in self.spk_audio_cache) or (sovits_model not in self.spk_audio_cache[spk_audio_path]["ge"]):
                self.cache_spk_audio(spk_audio_path, sovits_model=sovits_model)
            else:
                # 缓存命中，更新LRU访问顺序
                self._touch_cache_key(self._spk_cache_access_order, spk_audio_path)
            ge = self.spk_audio_cache[spk_audio_path]["ge"][sovits_model]
        sovits = self.sovits_models[sovits_model]
        return sovits, ge

    def _contains_chinese(self, text: str) -> bool:
        segments = LangSegment.getTexts(text)
        for segment in segments:
            if segment['lang'] == 'zh':
                return True
        return False

    def _ensure_bert_loaded(self):
        if self._bert_loaded:
            return
        if not self.auto_bert:
            return
        if os.path.exists(self.cnroberta_path):
            self.tts_config.cnroberta = CNRoberta(self.cnroberta_path, self.tts_config)
            self._bert_loaded = True
            logger.info("BERT model loaded lazily for Chinese text")
        else:
            logger.warning(f"BERT 模型路径不存在: {self.cnroberta_path}，跳过自动加载")

    def _check_pause(self, text: str) -> bool:
        return text.endswith(self.punctuation) or text[-3:] in ["...", "。。。"]

    def _get_prompt(self, cnhubert_model: CNHubert, sovits_model: Sovits, audio_path: str):
        wav, sr = self._load_audio(audio_path)
        wav = wav.to(self.tts_config.device)
        wav16k = self._resample(wav, sr, 16000).mean(dim=0)
        wav16k = wav16k.to(self.tts_config.dtype)
        silence = torch.zeros(int(16000 * 0.3), device=wav16k.device, dtype=wav16k.dtype)
        wav16k = torch.cat([wav16k, silence])
        ssl_content = cnhubert_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = sovits_model.vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0).to(self.tts_config.device)
        return prompt

    def _resample(self, audio_tensor, sr0, sr1):
        key = "%s-%s" % (sr0, sr1)
        if key not in self.resample_transform_dict:
            self.resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(self.tts_config.device)
        return self.resample_transform_dict[key](audio_tensor)

    def _get_spec(self, hps, filename):
        sr1 = int(hps.data.sampling_rate)
        audio, sr0 = self._load_audio(filename)
        audio = audio.to(self.tts_config.device).float()
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
        if sr0 != sr1:
            audio = self._resample(audio, sr0, sr1)
        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(2, maxx)

        key = "%s-%s-%s" % (hps.data.filter_length, hps.data.hop_length, hps.data.win_length)
        if key not in self.spectrogram_transform_dict:
            self.spectrogram_transform_dict[key] = torchaudio.transforms.Spectrogram(
                n_fft=hps.data.filter_length,
                win_length=hps.data.win_length,
                hop_length=hps.data.hop_length,
                center=True,
                pad_mode="reflect",
                power=1.0,
            ).to(self.tts_config.device)
        spectrogram_torch = self.spectrogram_transform_dict[key]
        spec = spectrogram_torch(audio)
        spec = spec.to(self.tts_config.dtype)
        audio = self._resample(audio, sr1, 16000)
        audio = audio.to(self.tts_config.dtype)
        return spec, audio

    def _sola_algorithm(self, f1_overlap, f2, overlap_len, search_len: int = 320):
        query = f1_overlap
        key = f2[:, :, :overlap_len + search_len]
        corr = F.conv1d(key, query)
        ones_kernel = torch.ones_like(query)
        energy = F.conv1d(key**2, ones_kernel) + 1e-8
        norm_corr = corr / torch.sqrt(energy)
        offset = norm_corr.argmax(dim=-1)
        f2_aligned = f2[:, :, offset.item():]
        alpha = torch.linspace(0, 1, overlap_len, device=self.tts_config.device, dtype=self.tts_config.dtype).view(1, 1, -1)
        f2_overlap = f2_aligned[:, :, :overlap_len]
        f_faded = f1_overlap * (1 - alpha) + f2_overlap * alpha
        f2_real = torch.cat([f_faded, f2_aligned[:, :, overlap_len:]], dim=-1)
        return f2_real, offset

    def _find_head_threshold_offsets(self, audio, threshold=0.02, frame_length=512, hop_length=256, search_len=64000, margin=3200):
        search_audio_head = audio[:search_len]
        frames_head = search_audio_head.unfold(0, frame_length, hop_length)
        rms_head = torch.sqrt(torch.mean(frames_head**2, dim=1))
        head_mask = rms_head > threshold
        head_indices = torch.nonzero(head_mask)
        if head_indices.numel() > 0:
            head_frame_idx = head_indices[0].item()
            head_offset = head_frame_idx * hop_length
            head_offset = max(0, head_offset - margin)
        else:
            head_offset = search_audio_head.shape[0]
        return head_offset

    @staticmethod
    def _load_audio(audio_path):
        with av.open(audio_path) as container:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format='flt', layout='mono', rate=stream.rate)
            frames = []
            for frame in container.decode(stream):
                for resampled_frame in resampler.resample(frame):
                    frames.append(resampled_frame.to_ndarray())
            audio_data = np.concatenate(frames, axis=1)
            return torch.from_numpy(audio_data), stream.rate

    def _touch_cache_key(self, access_order: list, key: str):
        """更新缓存访问顺序（LRU），将key移到末尾表示最近访问"""
        try:
            access_order.remove(key)
        except ValueError:
            pass
        access_order.append(key)

    def _evict_cache(self, cache_dict: dict, access_order: list):
        """LRU淘汰：当缓存超过上限时，删除最久未访问的条目"""
        while len(cache_dict) >= self.max_cache_size and access_order:
            oldest_key = access_order.pop(0)
            if oldest_key in cache_dict:
                del cache_dict[oldest_key]
                logger.info(f'LRU evicted cache entry: {oldest_key}')

    def _empty_cache(self):
        """清理GPU显存碎片（保留模型和音频缓存以保证性能）"""
        try:
            gc.collect()
            if self.tts_config.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.tts_config.device.type == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass
