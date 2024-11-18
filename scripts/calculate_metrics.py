import os
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_metrics(reference_file, predictions_file):
    references = pd.read_csv(reference_file)['Fixed Code'].tolist()
    df = pd.read_csv(predictions_file)
    
    predictions = df['Prediction'].tolist()
    ground_truths = df['Ground Truth'].tolist()

    smoothie = SmoothingFunction().method4
    bleu_scores = [sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie) for ref, pred in zip(references, predictions)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    correct_predictions = sum([1 if gt.strip() == pred.strip() else 0 for gt, pred in zip(ground_truths, predictions)])
    accuracy = correct_predictions / len(ground_truths)

    print(f"Average BLEU-4 score: {avg_bleu:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    return avg_bleu, accuracy

datasets = {
    "small": "data/small/test_with_complexity.csv",
    "medium": "data/medium/test_medium_with_complexity.csv",
    "small+medium": "data/small+medium/test_combined_with_complexity.csv",
}

models = {
    "small": "models/small_batch_4/evaluation_results.csv",
    "medium": "models/medium_batch_4/evaluation_results.csv",
    "small+medium": "models/small+medium_batch_4/evaluation_results.csv",
}

results = []
for dataset_name in datasets.keys():
    print(f"Calculating metrics for {dataset_name} dataset...")
    reference_file = datasets[dataset_name]
    predictions_file = models[dataset_name]
    print(f"Reference file: {reference_file}")
    print(f"Predictions file: {predictions_file}")
    bleu, accuracy = calculate_metrics(reference_file, predictions_file)
    results.append((dataset_name, bleu, accuracy))

results_df = pd.DataFrame(results, columns=["Dataset", "BLEU-4", "Accuracy"])
results_csv_path = "metrics_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Metrics saved to {results_csv_path}")
