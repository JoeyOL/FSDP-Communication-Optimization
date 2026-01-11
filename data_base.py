from logger import logger
import json
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.distributed as dist
import os
import torch

class WikipediaDataset(Dataset):
    """Wikipedia 数据集类"""
    
    def __init__(self, data_path, tokenizer, max_length=512, cache_dir: str | None = None):  # 减少到256
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        # 预分词缓存：默认放在与数据集同目录下，避免每次启动都重复 tokenize
        if cache_dir is None:
            cache_dir = os.path.dirname(os.path.abspath(data_path))
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"tokenized_{os.path.basename(data_path)}_len{max_length}.pt")
        
        logger.info(f"加载并预分词数据集: {data_path}")

        # 如果缓存存在，直接加载（所有 rank 都可读）
        if os.path.exists(cache_path):
            if rank == 0:
                logger.info(f"发现预分词缓存，直接加载: {cache_path}")
            if is_dist:
                dist.barrier()
            self.encodings = torch.load(cache_path)
            if is_dist:
                dist.barrier()
            return

        # rank0 做预处理并写缓存；其他 rank 等待后加载
        if rank != 0:
            if is_dist:
                dist.barrier()
            self.encodings = torch.load(cache_path)
            if is_dist:
                dist.barrier()
            return

        # rank0 负责生成缓存
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 在初始化时就进行分词
        progress_bar = tqdm(raw_data, desc="预分词中", disable=False)
        for item in progress_bar:
            text = item.get('title', '') + "\n" + item.get('text', '')
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",  # 确保所有输出长度一致
                return_tensors=None,   # 返回 list，而不是 tensor
            )
            self.encodings.append(encoding)
            
        logger.info(f"预分词完成，有效数据集大小: {len(self.encodings)} 个样本")

        torch.save(self.encodings, cache_path)
        logger.info(f"预分词缓存已写入: {cache_path}")

        if is_dist:
            dist.barrier()  # 通知其他 rank 可以读取缓存
            # 让其他 rank 读取结束后再继续，避免部分 rank 先进入训练
            dist.barrier()
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        # 直接返回已经分词好的数据
        return self.encodings[idx]