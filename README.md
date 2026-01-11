# FSDP-Communication-Optimization

## 生成 Wikipedia 英文训练数据集（可复现）

本项目的 `WikipediaDataset` 期望输入为 **JSON 数组**，每条样本包含：`id/title/text`。

你可以用脚本 `download_wikipedia_en.py` 直接下载并导出。

### 生成一个约 10MB 的小数据集

```bash
python download_wikipedia_en.py \
	--config 20220301.en \
	--output /root/llama-7b/datasets/wikipedia_en_10mb_new.json \
	--max_mb 10
```

### 生成一个约 1000 条样本的数据集（用于快速调试）

```bash
python download_wikipedia_en.py \
	--config 20220301.en \
	--output /root/llama-7b/datasets/wikipedia_en_1k.json \
	--max_samples 1000
```

说明：`--max_mb` 是近似控制（按输出文件写入字节数判断），不同文本长度会导致样本数不同。

## 验证与对话

训练结束后会在 `${output_dir}/final_model/` 下保存：

- `pytorch_model.bin`：模型权重
- tokenizer 文件：`vocab.json` / `merges.txt` / `tokenizer_config.json` 等

### 1) 离线验证（loss / perplexity）

```bash
python validate_model.py \
	--model_dir /root/llama-7b/fsdp_output/tmp_save_check/final_model \
	--data_path /root/llama-7b/datasets/wikipedia_en_10mb.json \
	--max_length 128 \
	--batch_size 2
```

### 2) 与训练后的模型对话（交互式）

```bash
python chat_model.py \
	--model_dir /root/llama-7b/fsdp_output/final_model \
	--max_new_tokens 128 \
	--temperature 0.8 \
	--top_p 0.95
```

输入 `/exit` 退出。
