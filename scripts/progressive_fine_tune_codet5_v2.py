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
def fine_tune_model(dataset, output_dir, tokenizer, model, epochs=3):
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

# Dataset batches for progressive fine-tuning (Curriculum Learning)
datasets = {
    # "small": [
    #     "/home/ygong07/CL_BugFixing/data/small/batches/batch_1_low.csv",
    #     "/home/ygong07/CL_BugFixing/data/small/batches/batch_2_low_medium.csv",
    #     "/home/ygong07/CL_BugFixing/data/small/batches/batch_3_medium_high.csv",
    #     "/home/ygong07/CL_BugFixing/data/small/batches/batch_4_high.csv",
    # ],
    # "medium": [
    #     "/home/ygong07/CL_BugFixing/data_/medium/batches/batch_1_low.csv",
    #     "/home/ygong07/CL_BugFixing/data_/medium/batches/batch_2_low_medium.csv",
    #     "/home/ygong07/CL_BugFixing/data_/medium/batches/batch_3_medium_high.csv",
    #     "/home/ygong07/CL_BugFixing/data_/medium/batches/batch_4_high.csv",
    # ],
      "small+medium": [
        "data/small+medium/batches/batch_1_low.csv",
        "data/small+medium/batches/batch_2_low_medium.csv",
        "data/small+medium/batches/batch_3_medium_high.csv",
        "data/small+medium/batches/batch_4_high.csv",
    ],
}

# Progressive fine-tuning
for dataset_name, batches in datasets.items():
    print(f"Starting fine-tuning for {dataset_name} dataset...")
    for i, batch_file in enumerate(batches, start=1):
        print(f"Fine-tuning on {dataset_name} batch {i}: {batch_file}")
        output_dir = os.path.join("/home/ygong07/CL_BugFixing/models", f"{dataset_name}_batch_{i}")
        os.makedirs(output_dir, exist_ok=True)

        # Load dataset
        dataset = load_data(batch_file)

        # Fine-tune model
        fine_tune_model(dataset, output_dir, tokenizer, model, epochs=3)

        # After finishing all small batches, continue training the same model on the next dataset (medium)
        if dataset_name == "small" and i == len(batches):
            print("Finished fine-tuning on small dataset. Proceeding to medium dataset...")
            break  # Exit the loop after completing small batches

# Proceed to the medium dataset
for i, batch_file in enumerate(datasets["medium"], start=1):
    print(f"Fine-tuning on medium batch {i}: {batch_file}")
    output_dir = os.path.join("/home/ygong07/CL_BugFixing/models", f"medium_batch_{i}")
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = load_data(batch_file)

    # Fine-tune model
    fine_tune_model(dataset, output_dir, tokenizer, model, epochs=3)

print("Progressive fine-tuning (CL) completed!")
