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

# Dynamic weighting using simplified lambda progression
def assign_weights(losses, lambda_param=1.0):
    return [max(0, 1 - loss / lambda_param) for loss in losses]

# Fine-tuning function with lightweight CL integration
def fine_tune_model(dataset, output_dir, tokenizer, model, lambda_value=1.0, lambda_step=0.1, epochs=3):
    tokenized_dataset = dataset.map(lambda batch: tokenize(batch, tokenizer), batched=True)
    train_dataset = tokenized_dataset.train_test_split(test_size=0.1)['train']
    eval_dataset = tokenized_dataset.train_test_split(test_size=0.1)['test']

    # Dynamic weights placeholder
    losses = []  # Placeholder for dynamic weights, replace with actual computed losses if needed
    weights = assign_weights(losses, lambda_value)

    # Training arguments optimized for efficiency
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save at the end of each epoch
        learning_rate=5e-5,
        per_device_train_batch_size=4,  # Moderate batch size for faster training
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # Simulate larger batch size
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,  # Log less frequently for faster runs
        save_total_limit=2,
        load_best_model_at_end=True,  # Ensure best model is saved
        metric_for_best_model="eval_loss",  # Lower validation loss indicates better model
        greater_is_better=False,
        fp16=True,  # Enable mixed precision for speed
        eval_accumulation_steps=4,  # Reduce memory usage during evaluation
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Update lambda value for the next batch
    lambda_value += lambda_step

    return lambda_value

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
}

# Progressive fine-tuning with lightweight CL
lambda_value = 1.0  # Initial lambda for dynamic weighting
lambda_step = 0.1   # Increment lambda gradually

for dataset_name, batches in datasets.items():
    print(f"Starting Curriculum Learning for {dataset_name} dataset...")
    for i, batch_file in enumerate(batches, start=1):
        print(f"Fine-tuning on {dataset_name} batch {i}: {batch_file}")
        output_dir = os.path.join("/home/ygong07/CL_BugFixing/models", f"{dataset_name}_batch_{i}")
        os.makedirs(output_dir, exist_ok=True)

        # Load dataset
        dataset = load_data(batch_file)

        # Fine-tune model with lightweight CL
        lambda_value = fine_tune_model(dataset, output_dir, tokenizer, model, lambda_value=lambda_value, lambda_step=lambda_step, epochs=3)

print("Curriculum Learning (CL) fine-tuning completed!")
