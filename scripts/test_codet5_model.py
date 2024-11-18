import os
import csv
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Function to save predictions to a CSV file
def save_prediction(input_code, predicted_code, output_file="test_results.csv"):
    # Check if the file exists, if not, write headers
    write_headers = not os.path.exists(output_file)

    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if write_headers:
            writer.writerow(["Buggy Code", "Predicted Fixed Code"])
        writer.writerow([input_code, predicted_code])

# Function to test the model on a single input and save the result
def test_model_and_save(tokenizer, model, input_code, output_file="test_results.csv"):
    # GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    inputs = tokenizer(input_code, truncation=True, padding='max_length', max_length=256, return_tensors="pt")
    
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Generate prediction
    outputs = model.generate(
        input_ids=inputs["input_ids"], 
        attention_mask=inputs["attention_mask"], 
        max_length=256
    )
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print the results
    print("Buggy Code:")
    print(input_code)
    print("\nPredicted Fixed Code:")
    print(prediction)

    # Save the prediction
    save_prediction(input_code, prediction, output_file)
    print(f"Saved prediction to {output_file}")

# Function to test the model on multiple examples
def test_multiple_examples(model_dir, examples, output_file="test_results.csv"):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    model = T5ForConditionalGeneration.from_pretrained(model_dir)

    # Remove the output file if it exists, to avoid appending to an old file
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Test the model on each example
    for input_code in examples:
        test_model_and_save(tokenizer, model, input_code, output_file)

# Example usage
if __name__ == "__main__":
    model_dir = "models/small+medium_batch_4/checkpoint-8355"

    examples = [
        """
        public int divide(int a, int b) {
            if (b == 0) {
                return -1;
            }
            return a / b;
        }
        """,
        """
        public int add(int a, int b) {
            return a - b; // Incorrect logic
        }
        """,
        """
        public int max(int a, int b) {
            if (a > b) {
                return b; // Incorrect return
            }
            return a;
        }
        """
    ]

    # Run the tests
    test_multiple_examples(model_dir, examples, output_file="test_results.csv")
