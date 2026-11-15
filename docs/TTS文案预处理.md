# TTS 文案预处理

`scripts/prepare_tts_text.py` 把混有 emoji、颜文字、弹幕标记和不规范标点的中文文案整理成纯文本朗读稿。输出稿要作为语音合成的输入；生成音频后，也把同一个文件作为字幕强制对齐脚本的 `--transcript-file`。这样配音和字幕使用同一份 ground truth。

## 离线清理

默认 `--mode clean` 不访问网络、不改写句子，只删除 emoji 和常见表情贴纸，规整标点；emoji 原本夹在两个词之间时会留下轻停顿。有意写出的“啊啊啊”等语气词会保留。

```sh
python3 scripts/prepare_tts_text.py raw.txt \
  --mode clean \
  --output scripts/tts稿.txt
```

## 长文切分

默认 `--split-mode auto --max-chars 220`：先尊重空行分出的段落，把相邻短段合并到长度上限；超长段落再沿句号、问号、感叹号和分句标点拆开。也可以用 `paragraph` 保留每段独立，或用 `sentence` 每句单独成块。没有清楚标点的超长句会在上限附近硬切并补一个逗号，个别情况下可能切到词中间；这种稿子建议先用 `rewrite` 模式补好断句，或手动加入标点。

```sh
python3 scripts/prepare_tts_text.py raw.txt \
  --mode clean \
  --split-mode auto \
  --max-chars 220 \
  --output scripts/tts稿.txt \
  --chunks-dir outputs/tts稿_chunks
```

总稿会以空行分隔各块；`--chunks-dir` 会额外写出 `chunk-001.txt`、`chunk-002.txt` 等编号文件，便于逐块合成。输出路径旁会自动保存 `.chunks.json` 清单，记录每块文本和字符数。清单中的编号文件是合成、拼接音频时的顺序；每块生成音频后，用对应的 chunk 文本做字幕对齐。

## 模型改写

`--mode rewrite` 会把原文发给显式配置的 OpenAI-compatible Chat Completions 接口，让模型把标点和断句改得更适合朗读，同时保留事实、人物口吻、搞笑和夸张。项目当前没有通用文本改写模型；没有配置接口时不会自动联网。

```sh
export TTS_TEXT_API_BASE='https://你的服务地址/v1'
export TTS_TEXT_MODEL='你的模型名'
export TTS_TEXT_API_KEY='你的服务密钥'

python3 scripts/prepare_tts_text.py raw.txt \
  --mode rewrite \
  --output scripts/tts稿.txt
```

本机运行的兼容服务可以省略 `TTS_TEXT_API_KEY`。也可以通过 `--style` 指定口吻，例如 `古风、克制、保留一本正经的荒诞感`。调用远端服务时，输入文案会发送给该服务。

## 后续生成和字幕对齐

先确认 `tts稿.txt`，再用它生成语音。若之后修改稿子，需要重新合成音频并重新对齐字幕。

```sh
python3 scripts/embed_chinese_subtitles.py \
  --image videos/静帧.jpg \
  --audio outputs/配音.wav \
  --transcript-file scripts/tts稿.txt \
  --output videos/成片.mp4
```
