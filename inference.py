import logging
import argparse
from transformers import AutoModelForMaskedLM
import torch
from rich.live import Live
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.text import Text
from tokenizer import get_tokenizer
from safetensors.torch import load_file

def load_model_and_tokenizer(path_to_weights, hf_model_name, device="cuda"):

    tokenizer = get_tokenizer(hf_model_name)

    model = AutoModelForMaskedLM.from_pretrained(hf_model_name, device_map=device)
    model.resize_token_embeddings(len(tokenizer))

    state_dict = load_file(path_to_weights)
    model.load_state_dict(state_dict, strict=False)
    model.tie_weights()
    model.eval()
    
    return model, tokenizer

def prepare_unconditional_tokens_for_inference(seq_len, mask_token_id, device="cuda"):

    input_tokens = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

    return input_tokens, mask, attention_mask

def prepare_conditional_tokens_for_inference(seq_len, tokenizer, prompt, device="cuda"):
    chat_template = [
        {"role": "user", "context": prompt}
    ]
    tokenized = tokenizer.apply_chat_template(
        chat_template,
        tokenize=True,
        add_special_tokens=True,
        add_generation_prompt=True
    )

    prompt_tokens = torch.tensor(tokenized).to(device)

    input_tokens, mask, attention_mask = prepare_unconditional_tokens_for_inference(
        seq_len, tokenizer.mask_token_id, device
    )

    input_tokens[0, :len(prompt_tokens)] = prompt_tokens
    mask[0, :len(prompt_tokens)] = False # Unpadding prompt tokens (they don't need mask)

    return input_tokens, mask, attention_mask

def format_display_for_qa(user_text, assistant_text):
    output = Text()
    output.append("USER: ", style="bold green")
    output.append(user_text + "\n\n")
    output.append("ASSISTANT: ", style="bold cyan")
    output.append(assistant_text, style="white")
    return output

def format_display_for_unconditional(gen_text):
    output = Text()
    output.append("Unconditional Generation: \n\n", style="bold green")
    output.append(gen_text, style="white")
    return output

def clean_text(raw_text: str) -> str:
    return (
        raw_text.replace("user", "")
        .replace("assistant", "")
        .strip()
    )

@torch.no_grad()
def inference(
    input_tokens,
    mask,
    attention_mask,
    num_steps,
    device="cuda",
    prompt=None,
    show_mask=True
):
    # Use separate Console instances for Progress and Live to avoid
    # "Only one live display may be active at once" errors when
    # Progress creates its own live renderable internally.
    progress_console = Console(highlight=False)
    live_console = Console(highlight=False)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=progress_console,
        transient=True
    ) as progress:
        task = progress.add_task("Generating...", total=num_steps)
         
        times = torch.linspace(1, 0, num_steps+1, device=device)

    with Live("", refresh_per_second=5, console=live_console) as live:
            for t, s in zip(times[:-1], times[1:]):
                logits = model(input_tokens, attention_mask=attention_mask).logits
                probs = torch.softmax(logits[mask], dim=-1)
                input_tokens[mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)

                remask_probs = torch.rand_like(mask, dtype=torch.float, device=device)
                remask_probs = (remask_probs < s/t)

                mask = mask & remask_probs
                input_tokens[mask] = tokenizer.mask_token_id
            
            if show_mask:
                decoded_tokens = tokenizer.convert_ids_to_tokens(input_tokens[0])
                cleaned_tokens = []
                for tok in decoded_tokens:
                    if tok == tokenizer.mask_token:
                        cleaned_tokens.append(tok)
                    elif tok in tokenizer.all_special_tokens:
                        continue
                    else:
                        cleaned_tokens.append(tok)
                decoded_after = tokenizer.convert_tokens_to_string(cleaned_tokens)
            else:
                decoded_after = tokenizer.batch_decode(input_tokens, skip_special_tokens=True)[0]
            print(decoded_after)
            if prompt is None:
                format_text = format_display_for_unconditional(decoded_after)
            else:
                assistant_text = decoded_after.replace(prompt, "").strip()
                assistant_text = clean_text(assistant_text)
                format_text = format_display_for_qa(prompt, assistant_text)
            live.update(format_text)
            progress.update(task, advance=1)

if __name__=="__main__":
    parser = argparse.ArgumentParser("Inference LDM")
    parser.add_argument("--safetensors_path", required=True, type=str)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=512)
    parser.add_argument("--strategy", type=str, default="random", choices=["random", "low_confidence"])
    parser.add_argument("--hf_model_name", type=str, default="distilbert/distilroberta-base")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Load Model
    model, tokenizer = load_model_and_tokenizer(args.safetensors_path, 
                                                args.hf_model_name, 
                                                args.device)


    if args.prompt is None:
        # Prepare Unconditional Inference Inputs
        input_tokens, mask, attention_mask = prepare_unconditional_tokens_for_inference(args.seq_len, 
                                                                                        mask_token_id=tokenizer.mask_token_id,
                                                                                        device=args.device)
    else:
        # Prepare Conditional Inference Inputs
        input_tokens, mask, attention_mask = prepare_conditional_tokens_for_inference(args.seq_len, 
                                                                                      tokenizer=tokenizer,
                                                                                      prompt=args.prompt,
                                                                                      device=args.device)

    
    inference(input_tokens, 
              mask, 
              attention_mask, 
              args.num_steps,
              device=args.device,
              prompt=args.prompt)