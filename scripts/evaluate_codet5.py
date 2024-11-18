import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Config
)
from safetensors.torch import load_file
from tqdm import tqdm

# Load model and tokenizer
def load_model_and_tokenizer(model_dir):
    print(f"Loading tokenizer from Salesforce/codet5-base")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base", use_fast=False)
    print(f"Looking for checkpoints in {model_dir}")

    # Find the latest checkpoint
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint")]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {model_dir}")

    latest_checkpoint = os.path.join(model_dir, sorted(checkpoints)[-1])
    print(f"Loading model from checkpoint: {latest_checkpoint}")

    config = T5Config.from_pretrained(latest_checkpoint)

    # Load model weights from safetensors
    state_dict = load_file(os.path.join(latest_checkpoint, "model.safetensors"))

    # Initialize a new model instance
    model = T5ForConditionalGeneration(config)

    # Filter the state_dict to match the model keys
    model_keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

    # Load the filtered state_dict into the model
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    print("Model loaded successfully")
    return tokenizer, model

# Evaluate the model
def evaluate_model(model_dir, test_file):
    tokenizer, model = load_model_and_tokenizer(model_dir)

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load test data
    test_data = pd.read_csv(test_file)
    buggy_code = test_data['Buggy Code'].tolist()
    fixed_code = test_data['Fixed Code'].tolist()

    # Generate predictions with progress bar
    predictions = []
    print("Generating predictions...")
    for code in tqdm(buggy_code, desc="Evaluating"):
        inputs = tokenizer(code, return_tensors="pt", truncation=True, padding="max_length", max_length=256)

        # Move inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = model.generate(inputs['input_ids'], max_length=256)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)

    # Save predictions and calculate metrics
    results = pd.DataFrame({
        'Buggy Code': buggy_code,
        'Ground Truth': fixed_code,
        'Prediction': predictions
    })
    results.to_csv(os.path.join(model_dir, "evaluation_results.csv"), index=False)
    print(f"Evaluation completed. Results saved to {os.path.join(model_dir, 'evaluation_results.csv')}")

datasets = {
    "small": "data/small/test_with_complexity.csv",
    "medium": "data/medium/test_medium_with_complexity.csv",
    "small+medium": "data/small+medium/test_combined_with_complexity.csv",
}

models = {
    "small": "models/small_batch_4",
    "medium": "models/medium_batch_4",
    "small+medium": "models/small+medium_batch_4",
}

# Evaluate each dataset
for dataset_name, test_file in datasets.items():
    print(f"Evaluating on {dataset_name} dataset...")
    model_dir = models[dataset_name]
    evaluate_model(model_dir, test_file)
