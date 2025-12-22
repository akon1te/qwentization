import argparse
import random
import torch
import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset


def load_mmlu_for_awq(tokenizer, n_samples=128):

    dataset = load_dataset("cais/mmlu", "all", split="validation", streaming=True)
    
    samples = []
    buffer = []

    for i, example in enumerate(dataset):
        buffer.append(example)
        if i >= 2000: break
    
    random.shuffle(buffer)
    selected_data = buffer[:n_samples]
    
    options = ["A", "B", "C", "D"]
    
    for ex in selected_data:

        question = ex['question']
        choices = ex['choices']
        
        prompt_text = f"{question}\n\nChoices:\n"
        for i, choice in enumerate(choices):
            prompt_text += f"{options[i]}. {choice}\n"
        prompt_text += "Answer:"
        
        if tokenizer.chat_template:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt_text

        samples.append(text)
        
    return samples


def main():
    parser = argparse.ArgumentParser(description="AWQ Quantization with MMLU Calibration")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the generic FP16/BF16 model")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the quantized model")
    parser.add_argument("--w_bit", type=int, default=4, help="Weight bit (default: 4)")
    parser.add_argument("--group_size", type=int, default=128, help="Quantization group size (default: 128)")
    parser.add_argument("--n_samples", type=int, default=128, help="Number of calibration samples (default: 128)")
    
    args = parser.parse_args()

    quant_config = { 
        "zero_point": True, 
        "q_group_size": args.group_size, 
        "w_bit": args.w_bit, 
        "version": "GEMM" 
    }

    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path, 
        **{"low_cpu_mem_usage": True, "use_cache": False}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)


    mmlu_samples = load_mmlu_for_awq(tokenizer, n_samples=args.n_samples)

    model.quantize(
        tokenizer, 
        quant_config=quant_config, 
        calib_data=mmlu_samples
    )

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    model.save_quantized(args.output_path)
    tokenizer.save_pretrained(args.output_path)


if __name__ == "__main__":
    main()
