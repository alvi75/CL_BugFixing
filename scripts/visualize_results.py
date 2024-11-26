import matplotlib.pyplot as plt

datasets = ["Small", "Medium", "Small+Medium"]
bleu_scores = [69.69, 87.14, 78.02]  # BLEU scores as percentages

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(datasets, bleu_scores, color=['salmon', 'skyblue', 'lightgreen'], alpha=0.8)

# Add BLEU score text labels
for bar, score in zip(bars, bleu_scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() - 5,
        f"{score:.2f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        color="white",
        fontweight="bold"
    )

# Add additional annotations
plt.plot([0, 1, 2], bleu_scores, 'r--', label="Optimal BLEU Trend", linewidth=1.5)
plt.scatter([1], [87.14], color='teal', s=100, label="Proceed with Medium")

plt.text(0, 72, "Low training data", fontsize=10, color="red", ha="center")

plt.text(2, 82, "Combination benefits", fontsize=10, color="red", ha="center")


plt.legend(loc="upper left", fontsize=10)

plt.title("BLEU Scores for Small, Medium, and Combined Datasets during Progressive Curriculum Learning with Key Observations Highlighted", fontsize=10)
plt.xlabel("Dataset", fontsize=14)
plt.ylabel("BLEU Score (%)", fontsize=14)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save and display the plot
plt.tight_layout()
plt.savefig("bleu_scores_with_annotations.png", dpi=300)
plt.show()
