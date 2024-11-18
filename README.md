# CL for Bug Fixing

This project fine-tunes and evaluates the **CodeT5 model** to automatically fix buggy code. The workflow includes data preprocessing, model training, evaluation, and testing on real-world examples. Below, you'll find a detailed guide on the project's structure and usage.

---

## **Table of Contents**
1. [Features](#features)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
   - [Fine-Tuning](#fine-tuning)
   - [Evaluation](#evaluation)
   - [Testing](#testing)
5. [Results](#results)
6. [Contributing](#contributing)

---

## **Features**
- Fine-tune the CodeT5 model on custom datasets to fix buggy code.
- Evaluate the model using **BLEU** and **Accuracy** metrics.
- Test the model on custom buggy code examples.
- Visualize BLEU scores across datasets for deeper insights.

---

## **Installation**

1. Install dependencies:
   ```bash
   conda create -n csci_680 python=3.10
   conda activate csci_680
   pip install -r requirements.txt
   ```

2. Ensure **PyTorch** and **Transformers** are installed:
   ```bash
   pip install torch transformers safetensors pandas matplotlib
   ```

---

## **Project Structure**
```
CL_Bugfixing/
├── data/                   # Contains datasets for training, evaluation, and testing
├── logs/                   # Training logs
├── models/                 # Saved fine-tuned models and checkpoints
├── scripts/                # Python scripts for various tasks
│   ├── calculate_metrics.py    # Computes BLEU and Accuracy
│   ├── evaluate_codet5.py      # Evaluates the model
│   ├── test_codet5_model.py    # Tests the model on custom buggy code
│   ├── prepare_datasets.py     # Prepares datasets for training
│   ├── progressive_fine_tune_codet5.py # Fine-tunes the CodeT5 model
├── README.md            
```

---

## **Usage**

### **Fine-Tuning**
1. Place your training datasets in the `data/` directory. Organize them into batches if progressive fine-tuning is required.
2. Run the fine-tuning script:
   ```bash
   python scripts/progressive_fine_tune_codet5.py
   ```

### **Evaluation**
1. Evaluate the fine-tuned models on test datasets:
   ```bash
   python scripts/evaluate_codet5.py
   ```
2. Visualize the BLEU scores using the `calculate_metrics.py` script:
   ```bash
   python scripts/calculate_metrics.py
   ```

### **Testing**
To test the model on custom buggy code, modify the `input_code` variable in `test_codet5_model.py` and run:
```bash
python scripts/test_codet5_model.py
```

Example buggy code input:
```java
public int divide(int a, int b) {
    if (b == 0) {
        return -1;
    }
    return a / b;
}
```

---

## **Results**
- **BLEU Scores**:
  - Small Dataset: **69.69%**
  - Medium Dataset: **87.14%**
  - Small+Medium Dataset: **78.02%**


---