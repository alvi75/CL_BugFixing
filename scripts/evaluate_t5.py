import os
import pandas as pd
import torch
from transformers import (
    RobertaTokenizer,
    T5ForConditionalGeneration,
    T5Config
)
from safetensors.torch import load_file
from tqdm import tqdm

# Load model and tokenizer
def load_model_and_tokenizer(model_dir):
    print(f"Loading tokenizer from custom configuration")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    tokenizer.add_tokens(["<SATD_START>", "<SATD_END>"])
    print(f"Looking for checkpoints in {model_dir}")

    # Find the latest checkpoint
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint")]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {model_dir}")

    latest_checkpoint = os.path.join(model_dir, sorted(checkpoints)[-1])
    print(f"Loading model from checkpoint: {latest_checkpoint}")

    # Define the custom T5Config
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
    model.resize_token_embeddings(len(tokenizer))  # Ensure the model matches the tokenizer size

    # Load the weights from safetensors
    state_dict_path = os.path.join(latest_checkpoint, "model.safetensors")
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"Model weights file not found: {state_dict_path}")

    # Load state_dict and filter out incompatible keys
    state_dict = load_file(state_dict_path)
    model_state_dict = model.state_dict()
    filtered_state_dict = {
        k: v for k, v in state_dict.items() if k in model_state_dict
    }
    missing_keys = [k for k in model_state_dict.keys() if k not in filtered_state_dict]
    print(f"Missing keys will be initialized: {missing_keys}")

    # Load the filtered state_dict into the model
    model.load_state_dict(filtered_state_dict, strict=False)

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
    results_file = os.path.join(model_dir, "evaluation_results.csv")
    results.to_csv(results_file, index=False)
    print(f"Evaluation completed. Results saved to {results_file}")

# Specify datasets and models
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
