import os
import csv
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

def save_prediction(input_code, predicted_code, output_file="test_results.csv"):
    write_headers = not os.path.exists(output_file)

    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if write_headers:
            writer.writerow(["Buggy Code", "Predicted Fixed Code"])
        writer.writerow([input_code, predicted_code])

def test_model_and_save(tokenizer, model, input_code, output_file="test_results.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(input_code, truncation=True, padding='max_length', max_length=256, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        input_ids=inputs["input_ids"], 
        attention_mask=inputs["attention_mask"], 
        max_length=256
    )
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Buggy Code:")
    print(input_code)
    print("\nPredicted Fixed Code:")
    print(prediction)

    save_prediction(input_code, prediction, output_file)
    print(f"Saved prediction to {output_file}")

def test_examples(model_dir, examples, output_file="test_results.csv"):
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    model = T5ForConditionalGeneration.from_pretrained(model_dir)

    if os.path.exists(output_file):
        os.remove(output_file)

    for input_code in examples:
        test_model_and_save(tokenizer, model, input_code, output_file)

if __name__ == "__main__":
    model_dir = "models/small_batch_4/checkpoint-2625"

    examples = [
        # Example 1
        """
        public int divide(int a, int b) {
            if (b == 0) {
                return -1;
            }
            return a / b;
        }
        """,
        # Example 2
        """
        public int add(int a, int b) {
            return a - b; // Incorrect logic
        }
        """,
        # Example 3
        """
        public int max(int a, int b) {
            if (a > b) {
                return b; // Incorrect return
            }
            return a;
        }
        """
    ]

    test_examples(model_dir, examples, output_file="test_results_small.csv")
