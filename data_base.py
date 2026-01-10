from logger import logger
import json
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.distributed as dist

class WikipediaDataset(Dataset):
    """Wikipedia 数据集类"""
    
    def __init__(self, data_path, tokenizer, max_length=512):  # 减少到256
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        
        logger.info(f"加载并预分词数据集: {data_path}")
        for _ in range(5):
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        
        # 在初始化时就进行分词
        progress_bar = tqdm(raw_data, desc="预分词中", disable=(dist.get_rank() != 0))
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
        dist.barrier()  # 确保所有进程都完成初始化
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        # 直接返回已经分词好的数据
        return self.encodings[idx]