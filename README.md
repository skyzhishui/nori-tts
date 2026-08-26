# nori-tts 流式合成服务

基于 GPT-SoVITS 的语音合成服务，提供 OpenAI 协议风格的 WebSocket 和 HTTP 流式合成端点，兼容 vLLM-Omni (TTS)。

支持两种合成模式：
- **CustomVoice 模式** — 使用本地预设音色名称
- **声音克隆模式** — 客户端实时传入参考音频和文本

## 架构

```
客户端 (WS / HTTP)
    │
    ▼
tts_server.py ─── 协议层：WS/HTTP 路由、消息路由、PCM 重采样、音频编码
    │
    ▼
tts_engine.py ─── 推理层：模型加载/缓存、文本处理、流式推理
    │
    ▼
src/ ─── 模型层：GPT/SoVITS 推理、G2P、特征提取、说话人验证
```

## 目录结构

```
├── tts_server.py        # 服务端（FastAPI + WebSocket + HTTP）
├── tts_engine.py        # 推理引擎（TTSEngine 类 + 文本处理 + 模型加载）
├── config.py            # 设备检测 + Config / GlobalConfig
├── lang_segment.py      # 多语言识别与分段
├── model_downloader.py  # 启动时自动下载缺失模型
├── voices.yaml          # 模型路径 + 预设音色配置
├── requirements.txt     # Python 依赖
├── README.md
└── src/                 # 模型定义
    ├── Featurizer/      # CNHubert / CNRoberta 特征提取
    ├── G2P/             # 多语言文本→音素转换
    ├── GPT/             # Text2SemanticDecoder
    ├── SoVITS/          # SynthesizerTrn 声码器
    ├── SV/              # ERes2Net 说话人验证
    └── utils.py
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# torch / torchaudio 需根据 CUDA 版本选择对应索引
# 详见 https://pytorch.org/get-started/locally/
```

额外依赖（文本处理）：
```bash
pip install pysbd py3langid
```

### 2. 模型自动下载（可选）

自动下载仅补充缺失文件，已有则跳过。下载源：ModelScope 首选。
可通过 HTTP_PROXY 环境变量配置代理。



### 3. 配置

编辑 `voices.yaml`，设置模型路径和预设音色：

```yaml
models_dir: /path/to/models
gpt_model: s1v3.ckpt
sovits_model: s2Gv2ProPlus.pth
target_sample_rate: 24000

voices:
  my_voice:
    ref_audio_path: /path/to/ref.wav
    ref_text: "参考音频对应的文本"
```

### 4. 启动服务

```bash
python tts_server.py
# 默认监听 ws://0.0.0.0:8091

# 自定义参数
python tts_server.py --port 8091 --host 0.0.0.0 --voices_config voices.yaml --models_dir /path/to/models
```

启动时会自动加载模型并预缓存所有预设音色的参考音频特征。

## API 端点总览

| 方法   | 路径                            | 用途                            |
|--------|-------------------------------|---------------------------------|
| WS     | `/v1/audio/speech/stream`      | WebSocket 流式 TTS（双向，支持增量文本输入） |
| POST   | `/v1/audio/speech`            | HTTP TTS（流式 chunked / 非流式，OpenAI 兼容） |
| GET    | `/v1/models`                  | 模型列表（OpenAI 兼容）             |
| GET    | `/v1/audio/voices`            | 预设音色列表                       |
| GET    | `/health`                     | 健康检查                          |
| GET    | `/`                           | 服务信息                          |

## HTTP 端点：POST /v1/audio/speech

OpenAI 兼容的 TTS 合成端点，同时支持流式和非流式。

### 请求体

```json
{
  "model": "gsv",
  "input": "你好，世界",
  "voice": "yaoyao",
  "response_format": "wav",
  "speed": 1.0,
  "stream": false
}
```

### 参数表

| 字段              | 类型          | 默认         | 说明                                                        |
|-----------------|-------------|------------|------------------------------------------------------------|
| `model`         | string      | `"gsv"`    | 模型名（兼容字段，由配置文件 `model_id` 决定）                              |
| `input`         | string      | **必填**     | 要合成的文本                                                     |
| `voice`         | string      | `""`       | 预设音色 ID（CustomVoice 模式）                                   |
| `response_format` | string    | `"wav"`    | `wav` / `pcm` / `mp3` / `opus` / `flac`                    |
| `speed`         | float       | `1.0`      | 语速倍率（**流式必须为 1.0**）                                       |
| `stream`        | bool        | `false`    | 是否启用流式输出                                                   |
| `language_type`  | string \| null | `null`  | `zh` / `en` / `ja` / `ko` 等，留空自动检测                          |
| `ref_audio`      | string \| null | `null`  | **克隆模式**：参考音频。支持 `data:audio/wav;base64,...` 格式          |
| `ref_text`       | string \| null | `null`  | **克隆模式**：参考音频对应的原文（必须严格对应）                          |
| `instructions`   | string \| null | `null`  | OpenAI 兼容字段（暂未实现语义）                                    |
| `top_k`         | int         | `15`       | nori-tts 扩展：Top-K 采样                                    |
| `top_p`         | float       | `1.0`      | nori-tts 扩展：Top-P 采样                                    |
| `temperature`   | float       | `1.0`      | nori-tts 扩展：温度                                          |
| `repetition_penalty` | float  | `1.35`     | nori-tts 扩展：重复惩罚                                        |
| `noise_scale`   | float       | `0.5`      | nori-tts 扩展：噪声尺度                                        |
| `stream_chunk`  | int         | `25`       | nori-tts 扩展：流式 token 块大小                               |
| `overlap_len`   | int         | `5`        | nori-tts 扩展：重叠帧数                                       |

> `voice` 和 `ref_audio`+`ref_text` 二选一。两者都提供时，`ref_audio` 优先（克隆模式）。

### 响应

**非流式（`stream: false`）**：直接返回完整音频文件，`Content-Type` 对应 `response_format`。

**流式（`stream: true`）**：返回 **chunked 裸 PCM 二进制流**（**非 SSE**）：

```http
HTTP/1.1 200 OK
Content-Type: audio/wav
Transfer-Encoding: chunked
```

- `response_format=wav`：首包为 44 字节 WAV 头（RIFF/data size 填 `0xFFFFFFFF` 占位），后续为裸 PCM chunks
- `response_format=pcm`：直接发送裸 PCM 数据（24kHz/16bit/mono/S16_LE）
- 无 `data:` 前缀、无 JSON 包装、无 base64 编码、无 `[DONE]` 结束标记
- 结束信号为 HTTP chunked 0-length chunk

### curl 示例

**非流式（保存 WAV）：**
```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gsv",
    "input": "今天天气真好",
    "voice": "yaoyao",
    "response_format": "wav"
  }' --output out.wav
```

**流式（PCM）：**
```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gsv",
    "input": "你好世界",
    "voice": "yaoyao",
    "response_format": "pcm",
    "stream": true,
    "speed": 1.0
  }' --output out.pcm
```

**克隆模式（流式）：**
```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gsv",
    "input": "要合成的目标文本",
    "ref_audio": "data:audio/wav;base64,UklGRi...",
    "ref_text": "参考音频对应的原文",
    "response_format": "pcm",
    "stream": true
  }' --output cloned.pcm
```

### Python 示例

```python
import requests

# 非流式
resp = requests.post(
    "http://localhost:8091/v1/audio/speech",
    json={
        "model": "gsv",
        "input": "你好，我是语音合成",
        "voice": "yaoyao",
        "response_format": "wav",
    },
    timeout=60,
)
resp.raise_for_status()
with open("out.wav", "wb") as f:
    f.write(resp.content)

# 流式 PCM
with requests.post(
    "http://localhost:8091/v1/audio/speech",
    json={
        "model": "gsv",
        "input": "你好，流式合成测试",
        "voice": "yaoyao",
        "response_format": "pcm",
        "stream": True,
        "speed": 1.0,
    },
    stream=True, timeout=60,
) as resp:
    resp.raise_for_status()
    with open("out.pcm", "wb") as f:
        for chunk in resp.iter_content(chunk_size=None):
            if chunk:
                f.write(chunk)
```

### OpenAI SDK 示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8091/v1",
    api_key="EMPTY",
)

# 非流式
resp = client.audio.speech.create(
    model="gsv",
    voice="yaoyao",
    input="你好，世界",
    response_format="wav",
)
resp.write_to_file("out.wav")

# 流式（chunked PCM）
with client.audio.speech.with_streaming_response.create(
    model="gsv",
    voice="yaoyao",
    input="你好，世界",
    response_format="pcm",
    extra_body={"stream": True, "speed": 1.0},
) as resp:
    with open("out.pcm", "wb") as f:
        for chunk in resp.iter_bytes():
            f.write(chunk)
```

## WebSocket 端点

端点：`ws://<host>:<port>/v1/audio/speech/stream`

### CustomVoice 模式

```jsonc
// 1. 会话配置
→ {"type": "session.config", "voice": "my_voice", "output_format": "pcm"}

// 2. 发送文本
→ {"type": "input.text", "text": "你好世界"}
→ {"type": "input.text", "text": "这是第二句"}

// 3. 输入结束
→ {"type": "input.done"}

// 4. 接收音频
← {"type": "audio.start", "sentence_index": 0, "sentence_text": "你好世界", "format": "pcm", "sample_rate": 24000}
← <binary: PCM 16bit 24kHz mono>
← {"type": "audio.done", "sentence_index": 0}
← {"type": "session.done", "total_sentences": 2}
```

### 声音克隆模式

```jsonc
→ {
    "type": "session.config",
    "ref_audio": "data:audio/wav;base64,UklGRi...",
    "ref_text": "参考音频对应的文本",
    "output_format": "pcm"
  }
```

### 消息类型

| 方向 | type | 说明 |
|------|------|------|
| → | `session.config` | 会话配置（必须首先发送） |
| → | `input.text` | 文本片段（可多次发送） |
| → | `input.done` | 输入结束 |
| ← | `audio.start` | 句子音频开始 |
| ← | binary frame | PCM 16bit / 24kHz / mono 裸流 |
| ← | `audio.done` | 句子音频结束 |
| ← | `session.done` | 整个会话结束 |
| ← | `error` | 错误信息 |

### session.config 字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `voice` | string | 否* | 预设音色名称 |
| `ref_audio` | string | 否* | 参考音频 base64（`data:audio/wav;base64,...`） |
| `ref_text` | string | 否* | 参考音频对应文本 |
| `output_format` | string | 否 | 输出格式，目前仅 `pcm` |
| `split_granularity` | string | 否 | 分句粒度：`sentence`（默认）/ `paragraph` |

> *`voice` 和 `ref_audio`+`ref_text` 二选一

## 配置说明

### voices.yaml

```yaml
# 模型配置
models_dir: /data/models        # 预训练模型根目录
gpt_model: s1v3.ckpt            # GPT 模型文件名（相对于 models_dir）
sovits_model: s2Gv2ProPlus.pth  # SoVITS 模型文件名
target_sample_rate: 24000       # 输出重采样目标采样率
model_id: gsv                   # OpenAI 兼容的模型 ID

# 预设音色
voices:
  voice_name:
    ref_audio_path: /path/to/ref.wav     # 参考音频（必填）
    ref_text: "参考音频对应文本"            # 参考文本（必填）
    emotion_audio_path: /path/to/emo.wav # 情感音频（可选）
```

### 设备自动检测

`config.py` 自动检测最佳计算设备：
1. CUDA GPU（优先选最高 SM 版本 + 最大显存）
2. Apple MPS
3. CPU（fallback）


## 音频输出

- 模型原始输出：32kHz float32
- 服务端自动重采样至 `target_sample_rate`（默认 24kHz）
- 最终输出：PCM 16bit LE / 24kHz / mono

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 监听地址 |
| `-p` / `--port` | `8091` | 监听端口 |
| `--voices_config` | `voices.yaml` | 配置文件路径 |
| `--models_dir` | 配置文件中的值 | 覆盖模型目录 |

## 致谢

感谢以下项目：

- [chinokikiss/GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite)
- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

## License

MIT
