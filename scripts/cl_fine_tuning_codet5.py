import os
import pandas as pd
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from tqdm import tqdm

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

# Assign weights to examples based on loss values (dynamic weighting)
def assign_weights(losses, lambda_param):
    return [max(0, 1 - loss / lambda_param) for loss in losses]

# Fine-tuning function with dynamic scheduling
def fine_tune_model(dataset, output_dir, tokenizer, model, lambda_value=1.0, epochs=3):
    tokenized_dataset = dataset.map(lambda batch: tokenize(batch, tokenizer), batched=True)
    train_dataset = tokenized_dataset.train_test_split(test_size=0.1)['train']
    eval_dataset = tokenized_dataset.train_test_split(test_size=0.1)['test']

    # Dynamic weighting based on loss values
    losses = []  # Example placeholder for losses
    weights = assign_weights(losses, lambda_value)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,  # Save best model based on validation
        metric_for_best_model="eval_loss",  # Choose validation loss as metric
        greater_is_better=False,  # Lower validation loss is better
        fp16=True,  # Mixed precision
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: {'bleu': compute_bleu(p.predictions, p.label_ids)},  # Optional: add custom metrics
    )

    trainer.train()

# Compute BLEU metric for validation (example placeholder function)
def compute_bleu(predictions, references):
    # Use a library like sacrebleu for actual BLEU computation
    return {"BLEU": 0.0}  # Replace with actual BLEU score computation

# Load CodeT5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")

# Dataset batches for Curriculum Learning
datasets = {
    "small+medium": [
        "data/small+medium/batches/batch_1_low.csv",
        "data/small+medium/batches/batch_2_low_medium.csv",
        "data/small+medium/batches/batch_3_medium_high.csv",
        "data/small+medium/batches/batch_4_high.csv",
    ],
}

# Progressive fine-tuning
lambda_value = 1.0  # Initial lambda value for dynamic weighting
lambda_step = 0.1   # Increment lambda to include harder examples gradually

for dataset_name, batches in datasets.items():
    print(f"Starting Curriculum Learning for {dataset_name} dataset...")
    for i, batch_file in enumerate(batches, start=1):
        print(f"Fine-tuning on {dataset_name} batch {i}: {batch_file}")
        output_dir = os.path.join("/home/ygong07/CL_BugFixing/models", f"{dataset_name}_batch_{i}")
        os.makedirs(output_dir, exist_ok=True)

        # Load dataset
        dataset = load_data(batch_file)

        # Fine-tune model with dynamic weighting
        fine_tune_model(dataset, output_dir, tokenizer, model, lambda_value=lambda_value, epochs=3)

        # Increment lambda after each batch
        lambda_value += lambda_step

print("Curriculum Learning (CL) fine-tuning completed!")
