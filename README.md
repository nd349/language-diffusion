# Language Diffusion Model

This folder contains code and utilities for training and evaluating a diffusion-style language model implemented in PyTorch. The implementation is intended for learning purposes and includes a training script (`pretrain.py`) that uses the Hugging Face `datasets` format and `accelerate` for multi-GPU / multi-node execution.

## Contents

- `pretrain.py` — pre-training script (training loop, evaluation, logging, checkpointing)
- `finetune_sft.py` - finetuning script (training loop, evaluation, logging, checkpointing)
- `tokenizer/` or tokenizer helper — tokenizer utilities used by the code
- `inference.py` - inference script
- other helper modules (data collate, model, criterion) used by the training scripts

## Quick start

```bash
python -m pip install --upgrade pip
pip install torch torchvision transformers datasets accelerate tqdm numpy
```

## Data

- The training script expects preprocessed/tokenized data stored using the Hugging Face `datasets` library and accessed via `load_from_disk(...)`. By default the code reads `args.path_to_prepped_data` — prepare your dataset and save with `dataset.save_to_disk(path)` before training.

## Pretraining

- The pre-training script `pretrain.py` uses `accelerate` and accepts CLI args (see `parse_args()` in the script).

```bash
accelerate launch pretrain.py \
    --experiment_name "" \
    --working_directory "" \
    --hf_model_name "" \
    --path_to_prepped_data "" \
    --num_training_steps 100000 \
    --log_wandb
```
## Finetuning
Finetuning only masks answer tokens and don't mask initial prompt tokens.
```
accelerate launch finetune_sft.py \
    --experiment_name "" \
    --path_to_pretrained_checkpoint "/" \
    --working_directory "" \
    --hf_model_name "" \
    --path_to_prepped_data ""
```

## References
1. Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., ... & Li, C. (2025). Large language diffusion models. arXiv preprint arXiv:2502.09992.
2. https://github.com/gumran/language-diffusion/tree/master 
3. https://github.com/priyammaz/PyTorch-Adventures/tree/main
