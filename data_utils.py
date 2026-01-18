import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokenizer import get_tokenizer

def SFTCollator(model_name="answerdotai/ModernBERT-base"):
    
    tokenizer = get_tokenizer(model_name)
    eos_token = tokenizer.eos_token_id

    def collate_fn(batch):
        inputs = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
        query_masks = [torch.tensor(b["query_mask"]) for b in batch]

        inputs = torch.nn.utils.rnn.pad_sequence(inputs, padding_value=eos_token, batch_first=True)
        query_masks = torch.nn.utils.rnn.pad_sequence(query_masks, padding_value=1, batch_first=True)

        return {"input_ids": inputs, "query_mask": query_masks}
    
    return collate_fn