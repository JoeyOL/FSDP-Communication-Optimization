from logger import logger
import json
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.distributed as dist
import os
import torch
from pathlib import Path
import ijson
import time

class WikipediaDataset(Dataset):
    """Wikipedia 数据集类"""
    
    def __init__(
        self,
        data_path,
        tokenizer,
        max_length=512,
        cache_dir: str | None = None,
        shard_size: int = 2000,
        max_samples: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        # 预分词缓存：默认放在与数据集同目录下，避免每次启动都重复 tokenize
        if cache_dir is None:
            cache_dir = os.path.dirname(os.path.abspath(data_path))
        os.makedirs(cache_dir, exist_ok=True)
        cache_prefix = os.path.join(cache_dir, f"tokenized_{os.path.basename(data_path)}_len{max_length}")
        cache_meta_path = f"{cache_prefix}.meta.pt"
        
        logger.info(f"加载并预分词数据集: {data_path}")

        def _atomic_save(obj: object, path: str) -> None:
            tmp = f"{path}.tmp"
            torch.save(obj, tmp)
            os.replace(tmp, path)

        def _is_compatible_meta(meta: dict) -> bool:
            return (
                meta.get("data_path") == str(data_path)
                and int(meta.get("max_length", -1)) == int(self.max_length)
                and int(meta.get("shard_size", -1)) == int(shard_size)
            )

        def _compat_mismatch_detail(meta: dict) -> str:
            pairs = [
                (
                    "data_path",
                    str(data_path),
                    str(meta.get("data_path")),
                ),
                (
                    "max_length",
                    str(int(self.max_length)),
                    str(meta.get("max_length")),
                ),
                (
                    "shard_size",
                    str(int(shard_size)),
                    str(meta.get("shard_size")),
                ),
            ]
            diffs = []
            for k, cur, old in pairs:
                if cur != old:
                    diffs.append(f"- {k}: current={cur} | meta={old}")
            if not diffs:
                return "(fields look equal, but meta was still considered incompatible)"
            return "\n".join(diffs)

        def _load_shards_from_meta(meta: dict) -> list:
            shards = meta.get("shards", [])
            all_enc = []
            for sp in shards:
                all_enc.extend(torch.load(sp))
            return all_enc

        # 如果缓存存在，直接加载（所有 rank 都可读）
        if os.path.exists(cache_meta_path):
            if rank == 0:
                logger.info(f"发现预分词缓存，直接加载: {cache_meta_path}")
            if is_dist:
                dist.barrier()
            meta = torch.load(cache_meta_path)
            if not meta.get("finalized", True):
                # 兼容旧 meta：没有 finalized 字段时视为已完成
                if rank == 0:
                    logger.warning("检测到未完成的预分词 meta，将尝试断点续作...")
            if not _is_compatible_meta(meta):
                raise ValueError(
                    "缓存 meta 与当前参数不一致："
                    f"{cache_meta_path}\n"
                    f"差异如下：\n{_compat_mismatch_detail(meta)}\n"
                    "建议：删除旧缓存后重跑，或保持训练参数与缓存一致。"
                )
            if not meta.get("finalized", True):
                # rank0 会走续作分支；其他 rank 等待 rank0 完成后再加载
                pass
            self.encodings = _load_shards_from_meta(meta)
            if is_dist:
                dist.barrier()
            return

        # rank0 做预处理并写缓存；其他 rank 等待后加载
        if rank != 0:
            # 不能直接 barrier：rank0 可能还在很长时间的预分词/写 shard 阶段，
            # 过早进入 NCCL barrier 可能触发 store 超时（默认 600s）。
            # 先轮询等待 meta 文件出现，再进入 barrier 做同步加载。
            wait_start = time.time()
            last_log = 0.0
            while True:
                if os.path.exists(cache_meta_path):
                    meta = torch.load(cache_meta_path)
                    # 必须等到 finalized 才能认为 shard 列表稳定
                    if meta.get("finalized", True):
                        break
                time.sleep(2.0)
                elapsed = time.time() - wait_start
                if elapsed - last_log > 30:
                    logger.info(f"Rank{rank} 等待预分词缓存生成中... 已等待 {elapsed:.0f}s")
                    last_log = elapsed
                # 1 小时超时，避免无限等待
                if elapsed > 3600:
                    raise TimeoutError(f"等待缓存超时: {cache_meta_path}")

            if is_dist:
                dist.barrier()
            meta = torch.load(cache_meta_path)
            self.encodings = _load_shards_from_meta(meta)
            if is_dist:
                dist.barrier()
            return

        # rank0 负责生成缓存（流式解析 JSON 数组，分片写入，避免 OOM；支持断点续作）
        shard_paths: list[str] = []
        shard_encodings: list[dict] = []

        resume_from = 0
        if os.path.exists(cache_meta_path):
            meta = torch.load(cache_meta_path)
            if _is_compatible_meta(meta) and not meta.get("finalized", False):
                shard_paths = list(meta.get("shards", []))
                resume_from = int(meta.get("processed_samples", meta.get("num_samples", 0)))
                logger.info(
                    f"检测到未完成缓存，断点续作：已处理 {resume_from} 条，已有分片 {len(shard_paths)}"
                )
            else:
                # 不兼容/已完成：不做续作
                resume_from = 0

        # 初始化/更新 meta（in_progress）
        meta = {
            "data_path": str(data_path),
            "max_length": int(self.max_length),
            "shard_size": int(shard_size),
            "processed_samples": int(resume_from),
            "num_samples": int(resume_from),
            "shards": shard_paths,
            "finalized": False,
        }
        _atomic_save(meta, cache_meta_path)

        n = 0

        # 用 ijson 流式解析最外层数组（假设数据格式为 [ {..}, {..}, ... ] ）
        with open(data_path, 'rb') as f:
            items = ijson.items(f, 'item')
            progress_bar = tqdm(desc="预分词中(流式)", disable=False)
            for idx, item in enumerate(items):
                # 跳过已处理样本，实现断点续作
                if idx < resume_from:
                    continue
                text = item.get('title', '') + "\n" + item.get('text', '')
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors=None,
                )
                shard_encodings.append(encoding)
                n += 1
                progress_bar.update(1)

                if shard_size > 0 and (n % shard_size == 0):
                    shard_path = f"{cache_prefix}.shard{len(shard_paths):06d}.pt"
                    torch.save(shard_encodings, shard_path)
                    shard_paths.append(shard_path)
                    shard_encodings = []

                    # 写入进度 meta（原子更新），支持中断恢复
                    meta["processed_samples"] = int(resume_from + n)
                    meta["num_samples"] = int(resume_from + n)
                    meta["shards"] = shard_paths
                    _atomic_save(meta, cache_meta_path)

                if max_samples and max_samples > 0 and n >= max_samples:
                    break

            progress_bar.close()

        # 写最后一个 shard
        if shard_encodings:
            shard_path = f"{cache_prefix}.shard{len(shard_paths):06d}.pt"
            torch.save(shard_encodings, shard_path)
            shard_paths.append(shard_path)

        # finalize
        total = int(resume_from + n)
        meta["processed_samples"] = total
        meta["num_samples"] = total
        meta["shards"] = shard_paths
        meta["finalized"] = True
        _atomic_save(meta, cache_meta_path)
        logger.info(f"预分词缓存 meta 已写入(完成): {cache_meta_path}")
        logger.info(f"预分词完成，有效数据集大小: {total} 个样本，分片数: {len(shard_paths)}")

        # 当前 rank0 也直接加载回内存，保持 Dataset 行为一致
        self.encodings = _load_shards_from_meta(meta)

        if is_dist:
            dist.barrier()  # 通知其他 rank 可以读取缓存
            # 让其他 rank 读取结束后再继续，避免部分 rank 先进入训练
            dist.barrier()
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        # 直接返回已经分词好的数据
        return self.encodings[idx]