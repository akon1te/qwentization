import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import prepare_model_for_kbit_training

def parse_args():
    parser = argparse.ArgumentParser(description="Train ONLY SVD lm_head on MMLU")
    
    parser.add_argument("--model_dir", type=str, required=True, 
                        help="Directory containing 'model.pt' and tokenizer files")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Where to save the finetuned model")
    
    # Dataset params
    parser.add_argument("--dataset_limit", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_acc", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4) 
    
    # Hub params
    parser.add_argument("--push_to_hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Repo ID")
    
    return parser.parse_args()

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}")

def main():
    args = parse_args()
    
    print(f"--- Configuration (Head Tuning) ---")
    
    # 1. Загрузка Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Датасет
    dataset = load_dataset("cais/mmlu", "all", split="auxiliary_train")

    def format_and_tokenize(example):
        options = ["A", "B", "C", "D"]
        prompt = f"{example['question']}\n\nChoices:\n"
        for i, choice in enumerate(example['choices']):
            prompt += f"{options[i]}. {choice}\n"
        prompt += "\nAnswer:"
        response = f" {options[example['answer']]}"
        
        # Chat template
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=512,
            padding=False,
            add_special_tokens=False
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print(f"Processing {args.dataset_limit} samples...")
    dataset = dataset.select(range(min(len(dataset), args.dataset_limit)))
    tokenized_dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)

    # 3. Загрузка Модели
    model_pt_path = os.path.join(args.model_dir, "model.pt")
    print(f"Loading SVD Model structure from {model_pt_path}...")
    
    model = torch.load(model_pt_path, weights_only=False)
    
    # Включаем Gradient Checkpointing и замораживаем всё
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # Важно для тренировки

    # --- FIX 1: Удаляем конфиг квантования (чтобы Trainer не ругался) ---
    print("Applying workaround for Trainer quantized check...")
    if hasattr(model.config, "quantization_config"):
        del model.config.quantization_config
    if hasattr(model, "is_quantized"):
        model.is_quantized = False

    # --- FIX 2: Исправляем TypeError: Object of type dtype is not JSON serializable ---
    print("Sanitizing config for JSON serialization...")
    # Проходимся по конфигу и превращаем torch.float16 -> "float16"
    for key, value in model.config.__dict__.items():
        if isinstance(value, torch.dtype):
            setattr(model.config, key, str(value).split('.')[-1])
            
    # Также проверяем стандартное поле torch_dtype
    if hasattr(model.config, "torch_dtype") and isinstance(model.config.torch_dtype, torch.dtype):
        model.config.torch_dtype = str(model.config.torch_dtype).split('.')[-1]

    # 4. Настройка весов головы
    print("Unfreezing lm_head parameters...")
    for param in model.lm_head.parameters():
        param.requires_grad = True
        # Приводим к FP16/BF16, если вдруг они были в другом формате
        if param.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
             param.data = param.data.to(torch.float16)

    print_trainable_parameters(model)

    # 5. Настройка Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True, 
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        optim="adamw_torch", 
        report_to="tensorboard",
        remove_unused_columns=False,
        # Gradient checkpointing
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting Training (Head Only)...")
    trainer.train()

    print(f"Saving finetuned model to {args.output_dir}...")
    
    # Сохраняем модель целиком
    torch.save(model, os.path.join(args.output_dir, "model_finetuned.pt"))
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hub_model_id:
        print(f"Pushing to Hub: {args.hub_model_id}")
        tokenizer.push_to_hub(args.hub_model_id)
        
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=args.hub_model_id,
            repo_type="model",
            ignore_patterns=["checkpoint-*", "runs"]
        )

    print("Done!")

if __name__ == "__main__":
    main()
