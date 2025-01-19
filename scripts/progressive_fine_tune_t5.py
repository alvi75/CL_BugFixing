import os
import pandas as pd
from transformers import RobertaTokenizer, T5ForConditionalGeneration, T5Config, Trainer, TrainingArguments
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
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
tokenizer.add_tokens(["<SATD_START>", "<SATD_END>"])

# Define T5Config with custom settings
t5_config = T5Config(
    decoder_start_token_id=tokenizer.convert_tokens_to_ids(['<pad>'])[0],
    d_model=768,
    d_ff=3072,
    d_kv=64,
    num_heads=12,
    num_layers=12,
    num_decoder_layers=12,
    eos_token_id=2,
    n_positions=512,
    torch_dtype="float32",
    bos_token_id=1,
    gradient_checkpointing=False,
    output_past=True
)

# Initialize T5 model with custom configuration
model = T5ForConditionalGeneration(config=t5_config)

# Resize token embeddings to include new tokens
model.resize_token_embeddings(len(tokenizer))

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

# Progressive fine-tuning
for dataset_name, batches in datasets.items():
    print(f"Starting progressive fine-tuning for {dataset_name} dataset...")
    for i, batch_file in enumerate(batches, start=1):
        print(f"Fine-tuning on {dataset_name} batch {i}: {batch_file}")
        output_dir = os.path.join("/home/ygong07/CL_BugFixing/models", f"{dataset_name}_batch_{i}")
        os.makedirs(output_dir, exist_ok=True)
        fine_tune_model(batch_file, output_dir, tokenizer, model, epochs=3)

        # Checkpoint validation
        checkpoint_path = os.path.join(output_dir, "pytorch_model.bin")
        if os.path.exists(checkpoint_path):
            model = T5ForConditionalGeneration.from_pretrained(output_dir)
        else:
            print(f"Warning: No checkpoint found in {output_dir}. Continuing without loading.")
