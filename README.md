# FSDP-Communication-Optimization

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
