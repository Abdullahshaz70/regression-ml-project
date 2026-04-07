# 📊 Machine Learning from Scratch — Project 1

> Implementing **Linear Regression** and **Logistic Regression** from scratch using NumPy, with comparisons against scikit-learn.

---

## 📁 Project Structure

```
├── linear_regression.py       # Question 1: Linear Regression
├── logistic_regression.py     # Question 2: Logistic Regression
├── HousingData.csv            # Boston Housing Dataset
├── Breast_cancer_dataset.csv  # Wisconsin Breast Cancer Dataset
```

---

## 📌 Question 1 — Linear Regression

**Dataset:** Boston Housing Dataset (506 samples, 13 features)

### What's Implemented
- Manual 80/20 train-test split with shuffling
- Null value imputation using column medians
- Feature standardization (zero mean, unit variance)
- Squared error loss function
- Gradient descent with convergence check: `‖w_new − w_old‖₂ ≤ ε`
- Experiments across **5 learning rates**: `0.0001, 0.001, 0.01, 0.1, 1.0`
- Training loss curve plotted per learning rate
- Failure case analysis (predictions with error > 5)
- Comparison with `sklearn.linear_model.LinearRegression`
- Classification metrics (Accuracy, F1-Score, Confusion Matrix) via median thresholding

### Key Formula

```
Loss = (1/n) * Σ(target − prediction)²

∂L/∂w = (-2/n) * Xᵀ(y − ŷ)
∂L/∂b = (-2/n) * Σ(y − ŷ)
```

---

## 📌 Question 2 — Logistic Regression (Binary Classification)

**Dataset:** [Wisconsin Breast Cancer Diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (M = Malignant, B = Benign)

### What's Implemented
- Sigmoid activation function
- Binary cross-entropy loss
- Gradient descent with convergence criteria
- Training loss curve
- Full evaluation: Accuracy, Precision, Recall, F1-Score
- **5-Fold Cross Validation** with per-fold and average metrics
- Comparison with `sklearn.linear_model.LogisticRegression`
- **Threshold tuning** across `[0.3, 0.4, 0.5, 0.6, 0.7]`
- Metrics vs. threshold plot

### Key Formula

```
L = -(1/n) * Σ [y·log(ŷ) + (1−y)·log(1−ŷ)]

ŷ = sigmoid(Xw + b) = 1 / (1 + e^(−z))
```

---

## ⚙️ Setup & Usage

### Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Run Linear Regression

```bash
python linear_regression.py
```

### Run Logistic Regression

```bash
python logistic_regression.py
```

> Make sure `HousingData.csv` and `Breast_cancer_dataset.csv` are in the same directory as the scripts.

---

## 📈 Results Summary

| Model | Metric | Custom Implementation | Sklearn |
|---|---|---|---|
| Linear Regression | MSE (lr=0.1) | 18.9384 | 18.9389 |
| Logistic Regression | Accuracy | 0.9802 | 0.9737 |
| Logistic Regression | F1-Score | 0.9851 | 0.9677 |

> Fill in actual values from your run outputs.

---

## 🛠️ Dependencies

| Library | Purpose |
|---|---|
| `numpy` | Core math & matrix operations |
| `pandas` | Data loading & preprocessing |
| `matplotlib` | Loss curves & metric plots |
| `scikit-learn` | Baseline model comparison & metrics |

---

## ⚠️ Notes

- All models are implemented **from scratch** using NumPy — no ML library is used for training.
- Scikit-learn is used **only** for baseline comparison and evaluation metrics.
- Convergence is determined by `‖w_new − w_old‖₂ ≤ 1e-4`.
