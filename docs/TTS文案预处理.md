# TTS 文案预处理

`scripts/prepare_tts_text.py` 把混有 emoji、颜文字、弹幕标记和不规范标点的中文文案整理成纯文本朗读稿。原文放在 `inputs/原始文本/`，整理后的稿子放在 `inputs/台本/`。输出稿要作为语音合成的输入；生成音频后，也把同一个文件作为字幕强制对齐脚本的 `--transcript-file`。这样配音和字幕使用同一份 ground truth。

## 离线清理

默认 `--mode clean` 不访问网络、不改写句子，只删除 emoji 和常见表情贴纸，规整标点；emoji 原本夹在两个词之间时会留下轻停顿。有意写出的“啊啊啊”等语气词会保留。

```sh
python3 scripts/prepare_tts_text.py inputs/原始文本/原文.txt \
  --mode clean \
  --output inputs/台本/tts稿.txt
```

## 从连续文案池取段

长期积累的原文放在 `inputs/原始文本/文案池.txt`。用 `@@@ id=唯一ID status=待取 @@@` 和 `@@@ end @@@` 明确界定每一段；取用时按 ID 整段选取，不从段落中按句子或语义抽取。运行 `python3 scripts/take_copy.py list` 查看状态，再运行：

```sh
python3 scripts/take_copy.py take copy-001 --max-chars 220
```

它会把选中的完整段落送入本页的格式化脚本，默认清理 emoji 和标点，再写入 `inputs/台本/copy-001.txt`。格式化成功后，文案池中的该段才会标记为 `已取`；失败时仍保持 `待取`。超长段落在清理后按固定字符数硬切，完全不寻找句号、分句或意义边界；分块文件和清单保存在 `outputs/文案切块/copy-001/` 与台本旁的 `.chunks.json`。

## 长文切分

默认 `--split-mode auto --max-chars 220`：先尊重空行分出的段落，把相邻短段合并到长度上限；超长段落再沿句号、问号、感叹号和分句标点拆开。也可以用 `paragraph` 保留每段独立，或用 `sentence` 每句单独成块。`fixed` 模式会先规整空白，再按字符数硬切，不参考任何标点和语义；适合严格按文案池原段取用、但单段过长的情况。`none` 则不切块。

```sh
python3 scripts/prepare_tts_text.py inputs/原始文本/原文.txt \
  --mode clean \
  --split-mode auto \
  --max-chars 220 \
  --output inputs/台本/tts稿.txt \
  --chunks-dir outputs/tts稿_chunks
```

总稿会以空行分隔各块；`--chunks-dir` 会额外写出 `chunk-001.txt`、`chunk-002.txt` 等编号文件，便于逐块合成。输出路径旁会自动保存 `.chunks.json` 清单，记录每块文本和字符数。清单中的编号文件是合成、拼接音频时的顺序；每块生成音频后，用对应的 chunk 文本做字幕对齐。

## 模型改写

`--mode rewrite` 会把原文发给显式配置的 OpenAI-compatible Chat Completions 接口，让模型把标点和断句改得更适合朗读，同时保留事实、人物口吻、搞笑和夸张。项目当前没有通用文本改写模型；没有配置接口时不会自动联网。

```sh
export TTS_TEXT_API_BASE='https://你的服务地址/v1'
export TTS_TEXT_MODEL='你的模型名'
export TTS_TEXT_API_KEY='你的服务密钥'

python3 scripts/prepare_tts_text.py inputs/原始文本/原文.txt \
  --mode rewrite \
  --output inputs/台本/tts稿.txt
```

本机运行的兼容服务可以省略 `TTS_TEXT_API_KEY`。也可以通过 `--style` 指定口吻，例如 `古风、克制、保留一本正经的荒诞感`。调用远端服务时，输入文案会发送给该服务。

## 后续生成和字幕对齐

先确认 `inputs/台本/tts稿.txt`，再用它生成语音。若之后修改稿子，需要重新合成音频并重新对齐字幕。

```sh
python3 scripts/embed_chinese_subtitles.py \
  --image videos/静帧.jpg \
  --audio outputs/配音.wav \
  --transcript-file inputs/台本/tts稿.txt \
  --output videos/成片.mp4
```
