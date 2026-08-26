"""
nori-tts 流式合成服务端

直接对接 GPT_SoVITS 推理引擎，无中间层依赖。
提供 OpenAI 协议风格的 WS 流式合成端点，兼容 vLLM-Omni (TTS)。
支持两种模式：
  1. CustomVoice 模式 - 使用本地配置的预设音色名称和参考音频
  2. 声音克隆模式 - 客户端实时传入 ref_audio (data: URI base64) 和 ref_text

协议流程（OpenAI 风格，兼容 vLLM-Omni TTS）：
  Client → Server:
    1. {"type": "session.config", ...}     ← 会话配置（必须首先发送）
    2. {"type": "input.text", "text": "..."}  ← 文本片段（可多次发送）
    3. {"type": "input.done"}              ← 输入结束

  Server → Client:
    1. {"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "pcm", "sample_rate": 24000}
    2. <binary frame: PCM 16bit/24kHz/mono>
    3. {"type": "audio.done", "sentence_index": 0}
    4. {"type": "session.done", "total_sentences": N}
"""

from __future__ import annotations

import sys
import argparse
import asyncio
import base64
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import re
import struct
import time
from contextlib import asynccontextmanager

import numpy as np
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import scipy.signal as signal

# 目标输出采样率（默认 24000），启动时从配置文件覆盖
TARGET_SAMPLE_RATE: int = 24000

# OpenAI 兼容的模型 ID，启动时从配置文件覆盖，缺省 "gsv"
MODEL_ID: str = "gsv"

# 将服务根目录加入 sys.path，使 config.py 和 src/ 可被找到
sys.path.insert(0, str(Path(__file__).parent))

from tts_engine import TTSEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("nori_tts")

# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------
tts_engine: Optional[TTSEngine] = None
voices_config: dict = {}  # 从 voices.yaml 加载的预设音色配置
temp_dir = tempfile.mkdtemp(prefix="nori_tts_ref_")

# 参考音频大小限制（防止超大 base64 payload 占满内存/磁盘）
MAX_REF_AUDIO_BYTES = 20 * 1024 * 1024  # 解码后上限 20MB
MAX_REF_AUDIO_B64_LEN = MAX_REF_AUDIO_BYTES * 4 // 3  # 对应 base64 文本长度上限
ui_enabled: bool = False  # 是否启用 WebUI
ui_tmp_dir: str = ""  # WebUI 合成音频保存目录
voices_config_path: str = ""  # voices.yaml 路径（用于运行时写入）

# ---------------------------------------------------------------------------
# 应用生命周期（lifespan，替代已弃用的 on_event）
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载模型，关闭时清理临时目录"""
    await startup_event()
    yield
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info("nori-tts 服务已关闭，临时目录已清理")


app = FastAPI(
    title="nori-tts 流式合成服务",
    description=(
        "OpenAI 协议风格的流式 TTS 服务端，兼容 vLLM-Omni (TTS)。\n\n"
        "## 端点\n"
        "- **WebSocket**: `/v1/audio/speech/stream` — WS 流式合成\n"
        "- **HTTP POST**: `/v1/audio/speech` — 流式/非流式合成（OpenAI 兼容）\n"
        "- **HTTP GET**: `/v1/models` — 模型列表\n"
        "- **HTTP GET**: `/v1/audio/voices` — 预设音色列表\n"
    ),
    version="2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic 请求模型（兼容 OpenAI /v1/audio/speech 协议）
# ---------------------------------------------------------------------------
class SpeechRequest(BaseModel):
    model: str = "gsv"  # 默认值，startup 时会被 MODEL_ID 覆盖
    input: str
    voice: str = ""
    response_format: str = "wav"
    speed: float = 1.0
    stream: bool = False
    language_type: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    instructions: str | None = None
    # nori-tts 扩展参数
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    noise_scale: float = 0.5
    stream_chunk: int = 25
    overlap_len: int = 5


class UISynthesizeRequest(BaseModel):
    """WebUI 合成请求"""
    voice: str = ""  # 预制音色名
    ref_audio: str = ""  # base64 编码的参考音频（克隆模式）
    ref_text: str = ""  # 参考文本（克隆模式）
    text: str = ""  # 待合成文本


class UIAddVoiceRequest(BaseModel):
    """WebUI 添加预制音色请求"""
    voice_name: str
    ref_audio: str  # base64 编码的参考音频
    ref_text: str  # 参考文本


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------
def load_voices_config(config_path: str) -> dict:
    """加载预设音色配置文件 (YAML)"""
    p = Path(config_path)
    if not p.exists():
        logger.warning(f"音色配置文件不存在: {config_path}，将使用空配置")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    logger.info(f"已加载音色配置: {config_path}，共 {len(cfg.get('voices', {}))} 个预设音色")
    return cfg


def resolve_voice(voice_name: str) -> dict:
    """根据音色名称查找预设配置，返回 {ref_audio_path, ref_text, ...}"""
    voices = voices_config.get("voices", {})
    if voice_name in voices:
        return voices[voice_name]
    return {}


# ---------------------------------------------------------------------------
# 音频工具
# ---------------------------------------------------------------------------
def float32_to_pcm16(data: np.ndarray) -> bytes:
    """将 float32 音频数据转为 PCM 16bit LE 字节流"""
    audio = np.asarray(data, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    return pcm.tobytes()


def resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    将音频数据从 orig_sr 重采样到 target_sr。
    使用 scipy.signal.resample_poly 进行高质量有理数重采样，
    避免频谱混叠，适合 32kHz → 24kHz (比率 3:4) 等场景。
    """
    if orig_sr == target_sr:
        return data
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = signal.resample_poly(data, up, down, padtype="edge")
    return resampled.astype(np.float32)


def decode_base64_audio(ref_audio_b64: str) -> str:
    """
    解码 base64 参考音频（支持 data:audio/wav;base64,... 格式），
    保存为临时文件并返回路径。
    """
    if ref_audio_b64.startswith("data:"):
        _, payload = ref_audio_b64.split(",", 1)
    else:
        payload = ref_audio_b64

    if len(payload) > MAX_REF_AUDIO_B64_LEN:
        raise ValueError(
            f"参考音频过大: base64 长度 {len(payload)} 超过上限 {MAX_REF_AUDIO_B64_LEN}"
            f"（约 {MAX_REF_AUDIO_BYTES // (1024 * 1024)}MB 原始数据）"
        )
    try:
        raw = base64.b64decode(payload)
    except Exception as e:
        raise ValueError(f"参考音频 base64 解码失败: {e}")
    if len(raw) > MAX_REF_AUDIO_BYTES:
        raise ValueError(f"参考音频过大: {len(raw)} bytes 超过上限 {MAX_REF_AUDIO_BYTES} bytes")
    tmp_path = os.path.join(temp_dir, f"ref_{uuid.uuid4().hex}.wav")
    with open(tmp_path, "wb") as f:
        f.write(raw)
    logger.info(f"参考音频已保存到: {tmp_path} ({len(raw)} bytes)")
    return tmp_path


# ---------------------------------------------------------------------------
# WAV / 音频编码工具（HTTP 端点专用）
# ---------------------------------------------------------------------------
def make_wav_header(sample_rate: int, bits_per_sample: int = 16, channels: int = 1) -> bytes:
    """生成 44 字节 WAV 头（用于流式场景，RIFF size 和 data size 填 0xFFFFFFFF 占位）"""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    # 占位大小：0xFFFFFFFF
    riff_size = 0xFFFFFFFF
    data_size = 0xFFFFFFFF
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,   # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header


def make_wav_bytes(pcm_data: bytes, sample_rate: int, bits_per_sample: int = 16, channels: int = 1) -> bytes:
    """生成完整 WAV 文件（含正确 size 字段，用于非流式场景）"""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_data)
    riff_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_data


def get_content_type(fmt: str) -> str:
    """根据 response_format 返回 Content-Type"""
    content_types = {
        "pcm": "audio/pcm",
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "flac": "audio/flac",
    }
    return content_types.get(fmt, "audio/wav")


# ---------------------------------------------------------------------------
# HTTP 端点辅助函数
# ---------------------------------------------------------------------------
def _resolve_http_voice(request: SpeechRequest) -> tuple[str, str, str, str | None]:
    """
    解析 HTTP 请求中的 voice/ref_audio 模式。
    返回 (spk_audio_path, prompt_audio_path, ref_text, temp_ref_file)。
    temp_ref_file 为克隆模式下需要清理的临时文件路径，否则为 None。
    """
    temp_ref_file = None

    if request.ref_audio:
        # ---- 声音克隆模式 ----
        if request.ref_audio.startswith("data:"):
            temp_ref_file = decode_base64_audio(request.ref_audio)
            ref_audio_path = temp_ref_file
        else:
            # 安全限制：仅接受 data: URI（base64），拒绝本地路径/URL，防止任意文件读取
            raise ValueError("克隆模式的 ref_audio 仅支持 data: URI（base64 编码）格式")
        ref_text = request.ref_text or ""
        spk_audio_path = ref_audio_path
        prompt_audio_path = ref_audio_path
        logger.info(f"HTTP 克隆模式: ref_audio={ref_audio_path}, ref_text={ref_text}")
    elif request.voice:
        # ---- CustomVoice 模式 ----
        voice_cfg = resolve_voice(request.voice)
        if not voice_cfg:
            raise ValueError(f"未找到预设音色: {request.voice}")

        ref_audio_path = voice_cfg.get("ref_audio_path", "")
        ref_text = voice_cfg.get("ref_text", "")
        emotion_audio_path = voice_cfg.get("emotion_audio_path", ref_audio_path)

        if not ref_audio_path:
            raise ValueError(f"预设音色 '{request.voice}' 缺少 ref_audio_path 配置")

        spk_audio_path = ref_audio_path
        prompt_audio_path = emotion_audio_path
        logger.info(
            f"HTTP CustomVoice 模式: voice={request.voice}, "
            f"ref_audio={ref_audio_path}, emotion={emotion_audio_path}"
        )
    else:
        raise ValueError("必须提供 voice（预设音色）或 ref_audio（克隆模式）")

    return spk_audio_path, prompt_audio_path, ref_text, temp_ref_file


async def _stream_pcm_generator(
    request: SpeechRequest,
    spk_audio_path: str,
    prompt_audio_path: str,
    ref_text: str,
    temp_ref_file: str | None,
):
    """
    流式 PCM 音频生成器，用于 StreamingResponse。
    逐 chunk yield 裸 PCM16 字节流（WAV 模式先 yield 44 字节 WAV 头）。
    voice/ref_audio 解析由调用方完成，此处仅负责推理和编码。
    """
    try:
        # WAV 流式：首包为 44 字节 WAV 头（size 占位 0xFFFFFFFF）
        if request.response_format == "wav":
            yield make_wav_header(TARGET_SAMPLE_RATE)

        async for audio_data, samplerate in tts_engine.infer_stream_async(
            spk_audio_path=spk_audio_path,
            prompt_audio_path=prompt_audio_path,
            prompt_audio_text=ref_text,
            text=request.input,
            is_cut_text=True,
            stream_mode="token",
            stream_chunk=max(1, request.stream_chunk),
            overlap_len=max(1, request.overlap_len),
            boost_first_chunk=True,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            noise_scale=request.noise_scale,
            speed=request.speed,
        ):
            resampled = resample_audio(audio_data, samplerate, TARGET_SAMPLE_RATE)
            pcm_bytes = float32_to_pcm16(resampled)
            yield pcm_bytes

    finally:
        # 清理克隆模式的临时文件
        if temp_ref_file and os.path.exists(temp_ref_file):
            try:
                os.remove(temp_ref_file)
                logger.info(f"已清理临时文件: {temp_ref_file}")
            except Exception:
                pass


def _encode_audio(pcm_data: bytes, fmt: str, sample_rate: int) -> tuple[bytes, str]:
    """
    根据格式编码音频数据，返回 (encoded_bytes, content_type)。
    不支持的格式回退到 wav。
    """
    if fmt == "pcm":
        return pcm_data, "audio/pcm"
    elif fmt == "wav":
        return make_wav_bytes(pcm_data, sample_rate), "audio/wav"
    else:
        # 尝试使用 pydub 编码 mp3/opus/flac
        try:
            from pydub import AudioSegment
            from io import BytesIO as _BytesIO

            audio = AudioSegment(
                data=pcm_data,
                sample_width=2,  # 16-bit
                frame_rate=sample_rate,
                channels=1,
            )
            buf = _BytesIO()
            if fmt == "mp3":
                audio.export(buf, format="mp3")
                return buf.getvalue(), "audio/mpeg"
            elif fmt == "opus":
                audio.export(buf, format="opus")
                return buf.getvalue(), "audio/opus"
            elif fmt == "flac":
                audio.export(buf, format="flac")
                return buf.getvalue(), "audio/flac"
            else:
                # 未知格式回退到 wav
                return make_wav_bytes(pcm_data, sample_rate), "audio/wav"
        except ImportError:
            logger.warning(f"pydub 未安装，无法编码 {fmt} 格式，回退到 wav")
            return make_wav_bytes(pcm_data, sample_rate), "audio/wav"
        except Exception as e:
            logger.warning(f"编码 {fmt} 格式失败: {e}，回退到 wav")
            return make_wav_bytes(pcm_data, sample_rate), "audio/wav"


# ---------------------------------------------------------------------------
# WS 协议处理
# ---------------------------------------------------------------------------
class WSSession:
    """管理单个 WebSocket 连接的会话状态"""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.configured = False
        self.session_config: dict = {}
        # CustomVoice 模式参数
        self.voice: str = ""
        self.ref_audio_path: Optional[str] = None
        self.ref_text: Optional[str] = None
        # 克隆模式临时文件（需要清理）
        self._temp_ref_file: Optional[str] = None
        # 推理参数
        self.speed: float = 1.0
        self.top_k: int = 15
        self.top_p: float = 1.0
        self.temperature: float = 1.0
        self.repetition_penalty: float = 1.35
        self.noise_scale: float = 0.5
        self.stream_chunk: int = 25
        self.is_cut_text: bool = True
        self.cut_minlen: int = 10
        self.cut_mute: float = 0.4
        self.overlap_len: int = 5
        # 句子计数
        self.sentence_index: int = 0
        # 文本缓冲区
        self._text_buffer: str = ""
        # 切分粒度：sentence (按 。?!．。？！) 或 clause (额外按 ，；,;)
        self.split_granularity: str = "sentence"

    # 句子结束标点（用于流式切分）
    _SENTENCE_PUNCT = set("。！？!?…\n")
    _CLAUSE_PUNCT = set("。！？!?…\n，；,;:：")

    async def cleanup(self):
        """清理会话资源（删除临时参考音频文件）"""
        if self._temp_ref_file and os.path.exists(self._temp_ref_file):
            try:
                os.remove(self._temp_ref_file)
                logger.info(f"已清理临时文件: {self._temp_ref_file}")
            except Exception:
                pass

    def _apply_config(self, config: dict):
        """应用 session.config 参数"""
        self.voice = config.get("voice", "")
        self.speed = float(config.get("speed", 1.0))
        self.stream_chunk = int(config.get("stream_chunk", 25))
        self.is_cut_text = bool(config.get("is_cut_text", True))
        self.cut_minlen = int(config.get("cut_minlen", 10))
        self.cut_mute = float(config.get("cut_mute", 0.4))
        self.overlap_len = int(config.get("overlap_len", 5))
        self.top_k = int(config.get("top_k", 15))
        self.top_p = float(config.get("top_p", 1.0))
        self.temperature = float(config.get("temperature", 1.0))
        self.repetition_penalty = float(config.get("repetition_penalty", 1.35))
        self.noise_scale = float(config.get("noise_scale", 0.5))

        # 切分粒度
        self.split_granularity = config.get("split_granularity", "sentence")
        if self.split_granularity not in ("sentence", "clause"):
            self.split_granularity = "sentence"

        # 判断模式：克隆模式 vs CustomVoice 模式
        ref_audio_val = config.get("ref_audio")
        ref_text_val = config.get("ref_text")

        if ref_audio_val:
            # ---- 声音克隆模式 ----
            if ref_audio_val.startswith("data:"):
                self._temp_ref_file = decode_base64_audio(ref_audio_val)
                self.ref_audio_path = self._temp_ref_file
            else:
                # 安全限制：仅接受 data: URI（base64），拒绝本地路径/URL，防止任意文件读取
                raise ValueError("克隆模式的 ref_audio 仅支持 data: URI（base64 编码）格式")

            self.ref_text = ref_text_val or ""
            logger.info(f"声音克隆模式: ref_audio={self.ref_audio_path}, ref_text={self.ref_text}")
        elif self.voice:
            # ---- CustomVoice 模式 ----
            voice_cfg = resolve_voice(self.voice)
            if not voice_cfg:
                raise ValueError(f"未找到预设音色: {self.voice}")

            self.ref_audio_path = voice_cfg.get("ref_audio_path", "")
            self.ref_text = voice_cfg.get("ref_text", "")

            # 情感参考音频（可选，若未配置则与音色参考相同）
            self.emotion_audio_path = voice_cfg.get("emotion_audio_path", self.ref_audio_path)

            if not self.ref_audio_path:
                raise ValueError(f"预设音色 '{self.voice}' 缺少 ref_audio_path 配置")

            logger.info(
                f"CustomVoice 模式: voice={self.voice}, "
                f"ref_audio={self.ref_audio_path}, ref_text={self.ref_text}"
            )
        else:
            raise ValueError("必须提供 voice（预设音色）或 ref_audio（克隆模式）")

        self.configured = True

    async def _synthesize_and_send(self, text: str):
        """对一段文本执行流式合成，并通过 WS 发送 PCM 音频帧"""
        if not text.strip():
            return

        # 发送 audio.start，固定输出 24kHz（重采样后）
        await self.ws.send_json({
            "type": "audio.start",
            "sentence_index": self.sentence_index,
            "sentence_text": text,
            "format": "pcm",
            "sample_rate": TARGET_SAMPLE_RATE,
        })

        # 在 CustomVoice 模式下，spk_audio 和 prompt_audio 可以不同
        spk_audio = self.ref_audio_path
        prompt_audio = getattr(self, "emotion_audio_path", self.ref_audio_path)

        # 流式推理并逐 chunk 发送 PCM 二进制帧
        try:
            async for audio_data, samplerate in tts_engine.infer_stream_async(
                spk_audio_path=spk_audio,
                prompt_audio_path=prompt_audio,
                prompt_audio_text=self.ref_text,
                text=text,
                is_cut_text=self.is_cut_text,
                cut_minlen=self.cut_minlen,
                cut_mute=self.cut_mute,
                stream_mode="token",
                stream_chunk=max(1, self.stream_chunk),
                overlap_len=max(1, self.overlap_len),
                boost_first_chunk=True,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                noise_scale=self.noise_scale,
                speed=self.speed,
            ):
                # audio_data: np.ndarray float32, samplerate: 32000
                # 重采样到 24kHz 后再转为 PCM16
                resampled = resample_audio(audio_data, samplerate, TARGET_SAMPLE_RATE)
                pcm_bytes = float32_to_pcm16(resampled)
                await self.ws.send_bytes(pcm_bytes)

        except WebSocketDisconnect:
            # 客户端在流式合成中途断开，不再 send，向上抛出由 handle 统一处理
            raise
        except Exception as e:
            logger.error(f"流式推理失败: {e}", exc_info=True)
            try:
                await self.ws.send_json({
                    "type": "error",
                    "message": f"推理失败: {str(e)}",
                })
            except WebSocketDisconnect:
                pass  # 发送错误消息时客户端已断开，忽略
            return

        # 发送 audio.done
        await self.ws.send_json({
            "type": "audio.done",
            "sentence_index": self.sentence_index,
        })
        self.sentence_index += 1

    def _extract_complete_sentences(self) -> list[str]:
        """
        从 _text_buffer 中切出所有"完整句子"（以句末标点结尾），
        剩余的不完整片段保留在缓冲区中。
        """
        punct = self._CLAUSE_PUNCT if self.split_granularity == "clause" else self._SENTENCE_PUNCT
        sentences: list[str] = []
        last_cut = 0
        for i, ch in enumerate(self._text_buffer):
            if ch in punct:
                segment = self._text_buffer[last_cut : i + 1].strip()
                if segment:
                    sentences.append(segment)
                last_cut = i + 1
        self._text_buffer = self._text_buffer[last_cut:]
        return sentences

    async def handle(self):
        """主消息循环"""
        try:
            while True:
                raw = await self.ws.receive()

                if raw.get("type") == "websocket.disconnect":
                    logger.info(f"客户端主动断开 (code={raw.get('code')})")
                    break

                if "text" in raw and raw["text"] is not None:
                    try:
                        msg = json.loads(raw["text"])
                    except json.JSONDecodeError:
                        await self.ws.send_json({"type": "error", "message": "无效的 JSON"})
                        continue

                    msg_type = msg.get("type", "")

                    if msg_type == "session.config":
                        try:
                            self._apply_config(msg)
                            logger.info(
                                f"会话配置完成: voice={self.voice}, "
                                f"clone={'yes' if self._temp_ref_file else 'no'}, "
                                f"split={self.split_granularity}"
                            )
                        except ValueError as e:
                            await self.ws.send_json({"type": "error", "message": str(e)})

                    elif msg_type == "input.text":
                        if not self.configured:
                            await self.ws.send_json({"type": "error", "message": "请先发送 session.config"})
                            continue
                        text = msg.get("text", "")
                        self._text_buffer += text
                        for sentence in self._extract_complete_sentences():
                            await self._synthesize_and_send(sentence)

                    elif msg_type == "input.done":
                        if not self.configured:
                            await self.ws.send_json({"type": "error", "message": "请先发送 session.config"})
                            continue
                        tail = self._text_buffer.strip()
                        self._text_buffer = ""
                        if tail:
                            await self._synthesize_and_send(tail)
                        await self.ws.send_json({
                            "type": "session.done",
                            "total_sentences": self.sentence_index,
                        })

                    else:
                        await self.ws.send_json({
                            "type": "error",
                            "message": f"未知的消息类型: {msg_type}",
                        })

                elif "bytes" in raw and raw["bytes"] is not None:
                    await self.ws.send_json({"type": "error", "message": "不支持二进制输入"})

        except WebSocketDisconnect:
            logger.info("客户端断开连接")
        except Exception as e:
            logger.error(f"会话异常: {e}", exc_info=True)
        finally:
            await self.cleanup()


# ---------------------------------------------------------------------------
# FastAPI 路由
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    voices = voices_config.get("voices", {})
    return {
        "service": "nori-tts 流式合成服务",
        "version": "2.0",
        "protocol": "OpenAI TTS Streaming (vLLM-Omni 兼容)",
        "ws_endpoint": "/v1/audio/speech/stream",
        "http_endpoint": "/v1/audio/speech",
        "preset_voices": list(voices.keys()),
        "clone_mode": True,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "tts_loaded": tts_engine is not None}


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """
    OpenAI 兼容的 TTS 合成端点。
    - stream=true: 返回 chunked 裸 PCM/WAV 二进制流（非 SSE）
    - stream=false: 返回完整音频文件
    """
    if tts_engine is None:
        return JSONResponse(status_code=503, content={"error": "TTS 引擎未加载"})

    if not request.input or not request.input.strip():
        return JSONResponse(status_code=400, content={"error": "input 文本不能为空"})

    try:
        # 在创建 StreamingResponse 之前解析 voice/ref_audio，
        # 这样 ValueError 可以被外层 try/except 捕获并返回 400
        spk_audio_path, prompt_audio_path, ref_text, temp_ref_file = _resolve_http_voice(request)

        if request.stream:
            # 流式：返回 chunked 裸二进制流
            return StreamingResponse(
                _stream_pcm_generator(
                    request, spk_audio_path, prompt_audio_path, ref_text, temp_ref_file
                ),
                media_type=get_content_type(request.response_format),
            )
        else:
            # 非流式：收集所有音频 chunk，编码后返回
            try:
                audio_chunks = []
                async for audio_data, samplerate in tts_engine.infer_stream_async(
                    spk_audio_path=spk_audio_path,
                    prompt_audio_path=prompt_audio_path,
                    prompt_audio_text=ref_text,
                    text=request.input,
                    is_cut_text=True,
                    stream_mode="sentence",
                    stream_chunk=max(1, request.stream_chunk),
                    overlap_len=max(1, request.overlap_len),
                    boost_first_chunk=True,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    temperature=request.temperature,
                    repetition_penalty=request.repetition_penalty,
                    noise_scale=request.noise_scale,
                    speed=request.speed,
                ):
                    resampled = resample_audio(audio_data, samplerate, TARGET_SAMPLE_RATE)
                    audio_chunks.append(resampled)

                if not audio_chunks:
                    return JSONResponse(status_code=500, content={"error": "合成结果为空"})

                full_audio = np.concatenate(audio_chunks)
                pcm_bytes = float32_to_pcm16(full_audio)
                encoded, content_type = _encode_audio(pcm_bytes, request.response_format, TARGET_SAMPLE_RATE)

                return Response(content=encoded, media_type=content_type)

            finally:
                if temp_ref_file and os.path.exists(temp_ref_file):
                    try:
                        os.remove(temp_ref_file)
                        logger.info(f"已清理临时文件: {temp_ref_file}")
                    except Exception:
                        pass

    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"HTTP 合成失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"合成失败: {str(e)}"})


@app.get("/v1/models")
async def list_models():
    """OpenAI 兼容的模型列表端点"""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mogu-home",
            }
        ],
    }


@app.get("/v1/audio/voices")
async def list_voices():
    """预设音色列表端点"""
    voices = voices_config.get("voices", {})
    return {
        "voices": [
            {"voice_id": name, "type": "preset"} for name in voices.keys()
        ]
    }


# ---------------------------------------------------------------------------
# WebUI API 路由（--ui 启用时可用）
# ---------------------------------------------------------------------------
@app.get("/ui/api/voices")
async def ui_list_voices():
    """WebUI: 获取预制音色列表（含参考文本）"""
    voices = voices_config.get("voices", {})
    return {
        "voices": [
            {
                "voice_id": name,
                "ref_text": cfg.get("ref_text", ""),
            }
            for name, cfg in voices.items()
        ]
    }


@app.post("/ui/api/synthesize")
async def ui_synthesize(request: UISynthesizeRequest):
    """WebUI: 离线合成音频，保存到 tmp/ 目录并返回"""
    if tts_engine is None:
        return JSONResponse(status_code=503, content={"error": "TTS 引擎未加载"})
    if not ui_enabled:
        return JSONResponse(status_code=404, content={"error": "WebUI 未启用"})
    if not request.text or not request.text.strip():
        return JSONResponse(status_code=400, content={"error": "合成文本不能为空"})

    # 克隆模式前置校验
    if request.ref_audio and not request.voice:
        if not request.ref_text or not request.ref_text.strip():
            return JSONResponse(status_code=400, content={"error": "克隆模式需提供参考文本"})
        # 校验参考音频是否为合法音频文件
        try:
            import soundfile as sf
            if request.ref_audio.startswith("data:"):
                _, payload = request.ref_audio.split(",", 1)
            else:
                payload = request.ref_audio
            raw = base64.b64decode(payload)
            if len(raw) < 44:
                return JSONResponse(status_code=400, content={"error": "参考音频文件过小，可能不是有效的音频文件"})
            # 写入临时文件后用 soundfile 校验
            _tmp_check = os.path.join(temp_dir, f"check_{uuid.uuid4().hex}.wav")
            with open(_tmp_check, "wb") as f:
                f.write(raw)
            info = sf.info(_tmp_check)
            if info.duration < 0.1:
                return JSONResponse(status_code=400, content={"error": "参考音频时长过短（至少0.1秒）"})
            if info.frames < 1:
                return JSONResponse(status_code=400, content={"error": "参考音频无效，无有效帧数据"})
            # 清理临时校验文件
            try:
                os.remove(_tmp_check)
            except Exception:
                pass
        except Exception as e:
            # 清理可能残留的临时文件
            for _f in [_tmp_check] if '_tmp_check' in dir() else []:
                try: os.remove(_f)
                except: pass
            err_msg = str(e)
            if "Invalid" in err_msg or "Not a valid" in err_msg or "unrecognized" in err_msg.lower() or "not recognised" in err_msg.lower() or "Format not recognised" in err_msg:
                return JSONResponse(status_code=400, content={"error": "参考音频格式无效，请上传合法的音频文件（wav/mp3/ogg）"})
            return JSONResponse(status_code=400, content={"error": "参考音频校验失败，请确认文件有效"})
    elif not request.voice and not request.ref_audio:
        return JSONResponse(status_code=400, content={"error": "请选择预制音色或上传参考音频"})

    try:
        # 构造内部 SpeechRequest 复用现有推理逻辑
        inner = SpeechRequest(
            input=request.text,
            voice=request.voice,
            ref_audio=request.ref_audio or None,
            ref_text=request.ref_text or None,
            response_format="wav",
            stream=False,
        )
        spk_audio_path, prompt_audio_path, ref_text, temp_ref_file = _resolve_http_voice(inner)

        # 推理
        audio_chunks = []
        async for audio_data, samplerate in tts_engine.infer_stream_async(
            spk_audio_path=spk_audio_path,
            prompt_audio_path=prompt_audio_path,
            prompt_audio_text=ref_text,
            text=request.text,
            is_cut_text=True,
            stream_mode="sentence",
            stream_chunk=25,
            overlap_len=5,
            boost_first_chunk=True,
        ):
            resampled = resample_audio(audio_data, samplerate, TARGET_SAMPLE_RATE)
            audio_chunks.append(resampled)

        # 清理临时参考音频
        if temp_ref_file and os.path.exists(temp_ref_file):
            try:
                os.remove(temp_ref_file)
            except Exception:
                pass

        if not audio_chunks:
            return JSONResponse(status_code=500, content={"error": "合成结果为空"})

        full_audio = np.concatenate(audio_chunks)
        pcm_bytes = float32_to_pcm16(full_audio)
        wav_bytes = make_wav_bytes(pcm_bytes, TARGET_SAMPLE_RATE)

        # 保存到 tmp/ 目录
        os.makedirs(ui_tmp_dir, exist_ok=True)
        filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
        filepath = os.path.join(ui_tmp_dir, filename)
        with open(filepath, "wb") as f:
            f.write(wav_bytes)
        logger.info(f"WebUI 合成音频已保存: {filepath} ({len(wav_bytes)} bytes)")

        return Response(content=wav_bytes, media_type="audio/wav",
                        headers={"X-Audio-Filename": filename})

    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"WebUI 合成失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"合成失败: {str(e)}"})


@app.get("/ui/api/audio/{filename}")
async def ui_get_audio(filename: str):
    """WebUI: 获取已合成的音频文件"""
    if not ui_enabled:
        return JSONResponse(status_code=404, content={"error": "WebUI 未启用"})
    # 安全检查：只允许文件名，不允许路径遍历
    safe_name = os.path.basename(filename)
    filepath = os.path.join(ui_tmp_dir, safe_name)
    if not os.path.exists(filepath):
        return JSONResponse(status_code=404, content={"error": "音频文件不存在"})
    return FileResponse(filepath, media_type="audio/wav", filename=safe_name)


@app.post("/ui/api/voices/add")
async def ui_add_voice(request: UIAddVoiceRequest):
    """WebUI: 添加预制音色（保存参考音频到 examples/，写入 voices.yaml）"""
    global voices_config
    if not ui_enabled:
        return JSONResponse(status_code=404, content={"error": "WebUI 未启用"})
    if not request.voice_name or not request.voice_name.strip():
        return JSONResponse(status_code=400, content={"error": "音色名称不能为空"})
    if not request.ref_audio:
        return JSONResponse(status_code=400, content={"error": "参考音频不能为空"})
    if not request.ref_text or not request.ref_text.strip():
        return JSONResponse(status_code=400, content={"error": "参考文本不能为空"})

    voice_name = request.voice_name.strip()

    # 安全校验：音色名只允许字母/数字/下划线/连字符/中日文，防止路径注入
    if not re.fullmatch(r"[\w\u4e00-\u9fff-]+", voice_name):
        return JSONResponse(
            status_code=400,
            content={"error": "音色名称只能包含字母、数字、下划线、连字符或中日文字符"},
        )

    # 检查是否已存在
    existing = voices_config.get("voices", {})
    if voice_name in existing:
        return JSONResponse(status_code=400, content={"error": f"音色 '{voice_name}' 已存在"})

    try:
        # 解码 base64 音频（统一走 decode_base64_audio，含大小限制与异常处理）
        try:
            tmp_decoded = decode_base64_audio(request.ref_audio)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        try:
            with open(tmp_decoded, "rb") as f:
                raw = f.read()
        finally:
            try:
                os.remove(tmp_decoded)
            except OSError:
                pass

        # 校验音频合法性
        import soundfile as sf
        if len(raw) < 44:
            return JSONResponse(status_code=400, content={"error": "音频文件过小，可能不是有效的音频文件"})
        _tmp_check = os.path.join(temp_dir, f"check_{uuid.uuid4().hex}.wav")
        with open(_tmp_check, "wb") as f:
            f.write(raw)
        try:
            info = sf.info(_tmp_check)
            if info.duration < 0.1:
                return JSONResponse(status_code=400, content={"error": "参考音频时长过短（至少0.1秒）"})
        except Exception as e:
            try: os.remove(_tmp_check)
            except: pass
            err_msg = str(e)
            if "Invalid" in err_msg or "Not a valid" in err_msg or "unrecognized" in err_msg.lower() or "not recognised" in err_msg.lower() or "Format not recognised" in err_msg:
                return JSONResponse(status_code=400, content={"error": "音频格式无效，请上传合法的音频文件"})
            return JSONResponse(status_code=400, content={"error": "音频校验失败，请确认文件有效"})
        finally:
            try: os.remove(_tmp_check)
            except: pass

        # 保存到 examples/ 目录
        examples_dir = os.path.join(os.path.dirname(voices_config_path) or ".", "examples")
        os.makedirs(examples_dir, exist_ok=True)
        audio_filename = f"{voice_name}.wav"
        audio_path = os.path.join(examples_dir, audio_filename)
        with open(audio_path, "wb") as f:
            f.write(raw)
        logger.info(f"参考音频已保存: {audio_path} ({len(raw)} bytes)")

        # 更新 voices_config 并写入 YAML
        rel_audio_path = f"./examples/{audio_filename}"
        if "voices" not in voices_config:
            voices_config["voices"] = {}
        voices_config["voices"][voice_name] = {
            "ref_audio_path": rel_audio_path,
            "ref_text": request.ref_text.strip(),
        }

        # 写回 YAML 文件
        with open(voices_config_path, "w", encoding="utf-8") as f:
            yaml.dump(voices_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        logger.info(f"已添加预制音色 '{voice_name}' 到 {voices_config_path}")

        # 预热新音色的参考音频缓存
        if tts_engine is not None:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda p=audio_path, t=request.ref_text.strip(): tts_engine.cache_prompt_audio(
                        prompt_audio_paths=p, prompt_audio_texts=t
                    ),
                )
                await loop.run_in_executor(
                    None,
                    lambda p=audio_path: tts_engine.cache_spk_audio(p),
                )
                logger.info(f"新音色 '{voice_name}' 预热完成")
            except Exception as e:
                logger.warning(f"新音色 '{voice_name}' 预热失败（运行时会延迟首次推理）: {e}")

        return {"ok": True, "voice_id": voice_name}

    except Exception as e:
        logger.error(f"添加预制音色失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"添加失败: {str(e)}"})


@app.websocket("/v1/audio/speech/stream")
async def ws_tts_stream(websocket: WebSocket):
    """
    WebSocket 流式 TTS 端点，OpenAI 协议风格，兼容 vLLM-Omni (TTS)。
    """
    await websocket.accept()
    session = WSSession(websocket)
    await session.handle()


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
async def startup_event():
    global tts_engine, TARGET_SAMPLE_RATE, MODEL_ID
    if tts_engine is not None:
        return

    # 从配置文件读取 target_sample_rate
    cfg_sr = voices_config.get("target_sample_rate")
    if cfg_sr:
        TARGET_SAMPLE_RATE = int(cfg_sr)
        logger.info(f"输出采样率: {TARGET_SAMPLE_RATE} Hz (来自配置文件)")
    else:
        logger.info(f"输出采样率: {TARGET_SAMPLE_RATE} Hz (默认)")

    # 从配置文件读取 model_id
    cfg_model_id = voices_config.get("model_id")
    if cfg_model_id:
        MODEL_ID = str(cfg_model_id)
    logger.info(f"模型 ID: {MODEL_ID}")

    # 同步 Pydantic SpeechRequest 的 model 默认值
    SpeechRequest.model_fields["model"].default = MODEL_ID

    logger.info("正在加载 nori-tts 模型...")
    logger.info(f"  models_dir = {models_dir}")

    max_cache_len = 1024
    batch_sizes = [1, 4, 8]
    cache_lens = []
    length = 512
    while length <= max_cache_len:
        cache_lens.append(length)
        length *= 2
    gpt_cache = [(b, c) for b in batch_sizes for c in cache_lens]

    tts_engine = TTSEngine(
        models_dir=models_dir,
        gpt_cache=gpt_cache,
        sovits_cache=[50],
    )

    # 预加载 GPT / SoVITS 模型权重
    gpt_model_name = voices_config.get("gpt_model")
    sovits_model_name = voices_config.get("sovits_model")
    gpt_path = str(Path(models_dir) / gpt_model_name) if gpt_model_name else None
    sovits_path = str(Path(models_dir) / sovits_model_name) if sovits_model_name else None

    logger.info(f"正在预加载模型权重...")
    logger.info(f"  GPT    = {gpt_path or '(默认)'}")
    logger.info(f"  SoVITS = {sovits_path or '(默认)'}")
    tts_engine.load_gpt_model(gpt_path) if gpt_path else tts_engine.load_gpt_model()
    tts_engine.load_sovits_model(sovits_path) if sovits_path else tts_engine.load_sovits_model()

    # 预热所有预设音色
    preset_voices = voices_config.get("voices", {})
    if preset_voices:
        logger.info(f"正在预热 {len(preset_voices)} 个预设音色的参考音频缓存...")
        loop = asyncio.get_running_loop()
        for voice_name, voice_cfg in preset_voices.items():
            ref_audio_path = voice_cfg.get("ref_audio_path")
            ref_text = voice_cfg.get("ref_text", "")
            emotion_audio_path = voice_cfg.get("emotion_audio_path", ref_audio_path)
            if not ref_audio_path or not Path(ref_audio_path).exists():
                logger.warning(f"  [{voice_name}] 参考音频不存在，跳过预热: {ref_audio_path}")
                continue
            try:
                await loop.run_in_executor(
                    None,
                    lambda p=emotion_audio_path, t=ref_text: tts_engine.cache_prompt_audio(
                        prompt_audio_paths=p, prompt_audio_texts=t
                    ),
                )
                await loop.run_in_executor(
                    None,
                    lambda p=ref_audio_path: tts_engine.cache_spk_audio(p),
                )
                logger.info(f"  [{voice_name}] 预热完成: spk={ref_audio_path}, emotion={emotion_audio_path}")
            except Exception as e:
                logger.warning(f"  [{voice_name}] 预热失败（运行时会延迟首次推理）: {e}")

    logger.info("nori-tts 模型加载与预热完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nori-tts 流式合成服务")
    parser.add_argument(
        "--models_dir", type=str, default=None,
        help="预训练模型目录（覆盖配置文件中的 models_dir）"
    )
    parser.add_argument(
        "--voices_config",
        type=str,
        default=str(Path(__file__).parent / "voices.yaml"),
        help="配置文件路径 (YAML)",
    )
    parser.add_argument(
        "-p", "--port", type=int, default=8091, help="服务监听端口 (默认 8091)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="服务监听地址"
    )
    parser.add_argument(
        "--ui", action="store_true", default=False,
        help="启用 WebUI 管理界面（访问 http://ip:port/ui/）"
    )
    args = parser.parse_args()

    voices_config = load_voices_config(args.voices_config)

    if args.models_dir:
        models_dir = args.models_dir
    elif voices_config.get("models_dir"):
        models_dir = voices_config["models_dir"]
    else:
        models_dir = "models"

    # --ui 模式：设置全局变量，挂载静态文件
    if args.ui:
        ui_enabled = True
        voices_config_path = args.voices_config
        ui_tmp_dir = str(Path(__file__).parent / "tmp")
        os.makedirs(ui_tmp_dir, exist_ok=True)

        # 挂载 WebUI 静态文件
        ui_static_dir = str(Path(__file__).parent / "ui")
        if os.path.isdir(ui_static_dir):
            app.mount("/ui", StaticFiles(directory=ui_static_dir, html=True), name="ui")
            logger.info(f"WebUI 已启用: http://{args.host}:{args.port}/ui/")
        else:
            logger.warning(f"WebUI 静态文件目录不存在: {ui_static_dir}，WebUI 不可用")

    uvicorn.run(app, host=args.host, port=args.port)
