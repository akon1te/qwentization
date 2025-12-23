import argparse
import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)


def format_and_tokenize(example):
        options = ["A", "B", "C", "D"]
        
        prompt = f"{example['question']}\n\nChoices:\n"
        for i, choice in enumerate(example['choices']):
            prompt += f"{options[i]}. {choice}\n"
        prompt += "\nAnswer:"
        
        response = f" {options[example['answer']]}"
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]

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


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune AWQ model with MMLU using LoRA")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base AWQ model")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the adapters")
    parser.add_argument("--dataset_limit", type=int, default=10000, help="Number of training samples to use")
    parser.add_argument("--batch_size", type=int, default=6, help="Per device training batch size")
    parser.add_argument("--grad_acc", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading Model from: {args.model_path}")
    print(f"Saving to: {args.output_dir}")


    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading MMLU dataset (auxiliary_train)...")
    dataset = load_dataset("cais/mmlu", "all", split="auxiliary_train")

    dataset = dataset.select(range(min(len(dataset), args.dataset_limit)))
    tokenized_dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)


    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        max_memory={0: "22GiB", 1: "18GiB"}, 
    )


    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        optim="adamw_torch",
        report_to="tensorboard",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
