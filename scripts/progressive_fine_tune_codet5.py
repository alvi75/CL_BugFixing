import os
import pandas as pd
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

# Load dataset and prepare
def load_data(file_path):
    df = pd.read_csv(file_path)
    dataset = Dataset.from_pandas(df)
    return dataset

# Tokenization function
def tokenize(batch, tokenizer, max_length=256):
    inputs = tokenizer(batch['Buggy Code'], truncation=True, padding='max_length', max_length=max_length)
    labels = tokenizer(batch['Fixed Code'], truncation=True, padding='max_length', max_length=max_length)['input_ids']
    inputs['labels'] = labels
    return inputs

# Fine-tuning function
def fine_tune_model(batch_file, output_dir, tokenizer, model, epochs=3):
    dataset = load_data(batch_file)
    tokenized_dataset = dataset.map(lambda batch: tokenize(batch, tokenizer), batched=True)
    train_dataset = tokenized_dataset.train_test_split(test_size=0.1)['train']
    eval_dataset = tokenized_dataset.train_test_split(test_size=0.1)['test']

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,  # Reduced batch size
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # Simulate larger batch size
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        fp16=True,  # Mixed precision
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

# Load CodeT5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")

# Dataset batches for progressive fine-tuning
datasets = {
    "small": [
        "data/small/batches/batch_1_low.csv",
        "data/small/batches/batch_2_low_medium.csv",
        "data/small/batches/batch_3_medium_high.csv",
        "data/small/batches/batch_4_high.csv",
    ],
    "medium": [
        "data/medium/batches/batch_1_low.csv",
        "data/medium/batches/batch_2_low_medium.csv",
        "data/medium/batches/batch_3_medium_high.csv",
        "data/medium/batches/batch_4_high.csv",
    ],
    "small+medium": [
        "data/small+medium/batches/batch_1_low.csv",
        "data/small+medium/batches/batch_2_low_medium.csv",
        "data/small+medium/batches/batch_3_medium_high.csv",
        "data/small+medium/batches/batch_4_high.csv",
    ],
}

# progressive fine-tuning
for dataset_name, batches in datasets.items():
    print(f"Starting progressive fine-tuning for {dataset_name} dataset...")
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
    for i, batch_file in enumerate(batches, start=1):
        print(f"Fine-tuning on {dataset_name} batch {i}: {batch_file}")
        output_dir = os.path.join("models", f"{dataset_name}_batch_{i}")
        os.makedirs(output_dir, exist_ok=True)
        fine_tune_model(batch_file, output_dir, tokenizer, model, epochs=3)

        # Checkpoint validation
        checkpoint_path = os.path.join(output_dir, "pytorch_model.bin")
        if os.path.exists(checkpoint_path):
            model = T5ForConditionalGeneration.from_pretrained(output_dir)
        else:
            print(f"Warning: No checkpoint found in {output_dir}. Continuing without loading.")
