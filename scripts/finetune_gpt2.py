"""Fine-tune a pretrained GPT-2 on the Shakespeare corpus with the HF Trainer.

This closes the loop on the project: after building a transformer from scratch,
we fine-tune a real pretrained model to see how a strong initialization changes
the generations. Requires the optional ``transformers`` and ``datasets`` packages.
"""

import argparse

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main():
    parser = argparse.ArgumentParser(description="fine-tune gpt-2 on shakespeare")
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--dataset", default="2nji/Shakespeare_Corpus")
    parser.add_argument("--output_dir", default="checkpoints/gpt2-shakespeare")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    dataset = load_dataset(args.dataset)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=50,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
