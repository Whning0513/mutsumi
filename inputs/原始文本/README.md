# 原始文本

放收到或撰写后尚未整理的原文，保留原始内容。清理、改写或切分后的朗读稿另存到 `../台本/`，不要覆盖原文。

## 连续文案池

在本地 `文案池.txt` 中持续追加文案。每条文案用唯一 ID 和显式开始/结束标记包起来；块内可以保留换行，块外只留空行：

```text
@@@ id=copy-001 status=待取 @@@
这里粘贴一整段原始文案。
@@@ end @@@

@@@ id=copy-002 status=待取 @@@
下一段原始文案。
@@@ end @@@
```

先列出段落状态，再按 ID 取一整块：

```sh
python3 scripts/take_copy.py list
python3 scripts/take_copy.py take copy-001 --max-chars 220
```

取用命令只接受完整块 ID，不按句号或语义搜索/截取。它调用 `scripts/prepare_tts_text.py` 清理选中内容；成功后才把该块的 `status` 改为 `已取`。默认台本写入 `inputs/台本/copy-001.txt`，固定长度分块写入 `outputs/文案切块/copy-001/`。

超长块在清理标点和 emoji 后，按 Unicode 字符数每 220 字硬切；不寻找句号、分句或语义边界，可能切在词中或标点处。需要别的长度时传 `--max-chars`。原文池保持原样，只修改已取状态。
