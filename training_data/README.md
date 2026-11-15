# 训练素材索引

本页说明本地 `training_data/` 中已有的素材和文本。远程微调正在使用的具体路径以当前远端任务为准；本地文件名保持不变，避免影响引用。

## `video_style_28sec/`

来源视频保存在本地 `视频音频下载/BV1XJpB6pEvu/`，不会提交到 Git。视频长 28.14 秒，整理出的可用人声约 27.5 秒；此数据包没有循环或填充到一分钟。

| 内容 | 文件 |
| --- | --- |
| 28 秒源音频 | `BV1XJpB6pEvu_28sec.wav` |
| 人工校对的完整字幕 | `BV1XJpB6pEvu_字幕人工校对.txt` |
| 六段字幕对齐音频 | `subtitle_chunk_01.wav` 至 `subtitle_chunk_06.wav` |
| 六段音频和文本清单 | `metadata_segments_v2.list` |
| 三段较粗切片和旧清单 | `clip_01.wav` 至 `clip_03.wav`；`metadata.list` |
| 5 秒结尾哭腔参考 | `BV1XJpB6pEvu_5sec_cry_reference.wav` |
| 参考音频对应文字 | `5sec_cry_reference_prompt.txt` |

`metadata_segments_v2.list` 对应六段字幕切片和人工校对稿。旧的 `metadata.list` 仍保留，但与新清单的切分及转写范围不同；新建训练任务前先确认远端任务使用哪份清单，不要将旧清单当作完整字幕。

## TTS 测试文本

- `scripts/视频急促哭腔_测试文本.txt` 是本地待合成目标稿，不提交到 Git。
- `outputs/archieve/20261116_cleanup/试听试验/视频急促哭腔_测试文本.txt` 是同一文本的输出目录副本；目前两份内容一致。脚本的源文本以 `scripts/` 下的版本为准。
- `5sec_cry_reference_prompt.txt` 是 5 秒参考音频的提示文字，和目标测试台词用途不同。
- 新文案的清理和朗读稿流程见[文案预处理说明](../docs/TTS文案预处理.md)。

## 相关成品与流程

同一测试台词的 GSV few-shot、28 秒微调和后接 RVC 版本见[快速开始中的视频风格实验](../docs/快速开始.md#视频风格-few-shot-与短数据微调)。数据说明、模型权重路径和参数都集中在该章节。
