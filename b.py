import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
ALPHA       = 0.1
EPSILON     = 1e-4
ITERATIONS  = 1000
K_FOLDS     = 5
THRESHOLD   = 0.5
THRESHOLDS  = [0.3, 0.4, 0.5, 0.6, 0.7]
TARGET_COL  = 'diagnosis'

# ─────────────────────────────────────────────
# 2. LOAD & CLEAN DATA
# ─────────────────────────────────────────────
df = pd.read_csv("Breast_cancer_dataset.csv")
df = df.drop(columns=['id'])

# ─────────────────────────────────────────────
# 3. FUNCTION DEFINITIONS
# ─────────────────────────────────────────────

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def shuffle_data(df):
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def train_test_split(df, target_col, split=0.8):
    n               = len(df)
    train_size      = int(split * n)
    train_df        = df[:train_size]
    test_df         = df[train_size:]

    X_train = train_df.drop(columns=target_col).to_numpy()
    Y_train = train_df[target_col].map({'M': 1, 'B': 0}).to_numpy()
    X_test  = test_df.drop(columns=target_col).to_numpy()
    Y_test  = test_df[target_col].map({'M': 1, 'B': 0}).to_numpy()

    return X_train, Y_train, X_test, Y_test


def normalize(X_train, X_test):
    mean    = X_train.mean(axis=0)
    std     = X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std


def initialize_params(n_features):
    w = np.zeros(n_features)
    b = 0
    return w, b


def forward_pass(X, w, b):
    z = X @ w + b
    return sigmoid(z)


def compute_loss(y, y_hat):
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    return (-1 / len(y)) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def compute_gradients(X, y, y_hat):
    n       = len(y)
    error   = y - y_hat
    dw      = (-1 / n) * X.T @ error
    db      = (-1 / n) * np.sum(error)
    return dw, db


def train(X, y, w, b, alpha=ALPHA, epsilon=EPSILON, iterations=ITERATIONS):
    losses = []
    for i in range(iterations):
        y_hat   = forward_pass(X, w, b)
        loss    = compute_loss(y, y_hat)
        losses.append(loss)

        dw, db  = compute_gradients(X, y, y_hat)
        w_old   = w.copy()
        w       = w - alpha * dw
        b       = b - alpha * db

        if np.linalg.norm(w - w_old) <= epsilon:
            print(f"  Converged at iteration {i}")
            break

    return w, b, losses


def predict(X, w, b, threshold=THRESHOLD):
    y_hat = forward_pass(X, w, b)
    return (y_hat >= threshold).astype(int)


def get_confusion_values(y, y_pred):
    TP = np.sum((y_pred == 1) & (y == 1))
    TN = np.sum((y_pred == 0) & (y == 0))
    FP = np.sum((y_pred == 1) & (y == 0))
    FN = np.sum((y_pred == 0) & (y == 1))
    return TP, TN, FP, FN


def get_scores(TP, TN, FP, FN):
    Accuracy  = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = (2 * Precision * Recall) / (Precision + Recall)
    return Accuracy, Precision, Recall, F1


def print_scores(Accuracy, Precision, Recall, F1):
    print(f"  Accuracy  : {Accuracy:.4f}")
    print(f"  Precision : {Precision:.4f}")
    print(f"  Recall    : {Recall:.4f}")
    print(f"  F1 Score  : {F1:.4f}")


def k_fold_cross_validation(X, y, k=K_FOLDS):
    n           = len(y)
    fold_size   = n // k
    accuracies  = []

    print(f"\n{'='*45}")
    print(f"  K-FOLD CROSS VALIDATION (k={k})")
    print(f"{'='*45}")

    for i in range(k):
        X_test_fold  = X[i * fold_size : (i + 1) * fold_size]
        y_test_fold  = y[i * fold_size : (i + 1) * fold_size]
        X_train_fold = np.concatenate([X[:i * fold_size], X[(i + 1) * fold_size:]])
        y_train_fold = np.concatenate([y[:i * fold_size], y[(i + 1) * fold_size:]])

        w, b        = initialize_params(X_train_fold.shape[1])
        w, b, _     = train(X_train_fold, y_train_fold, w, b)
        y_pred      = predict(X_test_fold, w, b)

        TP, TN, FP, FN              = get_confusion_values(y_test_fold, y_pred)
        Accuracy, Precision, Recall, F1 = get_scores(TP, TN, FP, FN)

        print(f"\n  Fold {i + 1}:")
        print_scores(Accuracy, Precision, Recall, F1)
        accuracies.append(Accuracy)

    print(f"\n  Average Accuracy: {np.mean(accuracies):.4f}")
    return accuracies


# ─────────────────────────────────────────────
# 4. MAIN EXECUTION
# ─────────────────────────────────────────────

# --- Data Preparation ---
df                              = shuffle_data(df)
X_train, Y_train, X_test, Y_test = train_test_split(df, TARGET_COL)
X_train, X_test                 = normalize(X_train, X_test)

# --- Train Model ---
print("\n" + "="*45)
print("  MODEL TRAINING")
print("="*45)
w, b        = initialize_params(X_train.shape[1])
w, b, losses = train(X_train, Y_train, w, b)

# --- Plot Training Loss ---
plt.figure()
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss vs Iterations")
plt.tight_layout()
plt.show()

# --- Evaluate on Test Set ---
print("\n" + "="*45)
print("  TEST SET EVALUATION (threshold=0.5)")
print("="*45)
y_pred                          = predict(X_test, w, b)
TP, TN, FP, FN                  = get_confusion_values(Y_test, y_pred)
Accuracy, Precision, Recall, F1 = get_scores(TP, TN, FP, FN)
print(f"\n  Confusion Matrix:")
print(f"  TP={TP}  FP={FP}")
print(f"  FN={FN}  TN={TN}")
print()
print_scores(Accuracy, Precision, Recall, F1)

# --- K-Fold Cross Validation ---
k_fold_cross_validation(X_train, Y_train)

# --- Sklearn Comparison ---
print("\n" + "="*45)
print("  SKLEARN COMPARISON")
print("="*45)
sk_model    = LogisticRegression(max_iter=1000)
sk_model.fit(X_train, Y_train)
sk_pred     = sk_model.predict(X_test)
print(f"\n  Sklearn Accuracy  : {accuracy_score(Y_test, sk_pred):.4f}")
print(f"  Sklearn Precision : {precision_score(Y_test, sk_pred):.4f}")
print(f"  Sklearn Recall    : {recall_score(Y_test, sk_pred):.4f}")
print(f"  Sklearn F1        : {f1_score(Y_test, sk_pred):.4f}")

# --- Threshold Tuning ---
print("\n" + "="*45)
print("  THRESHOLD TUNING")
print("="*45)
accuracies, precisions, recalls, f1s = [], [], [], []

for t in THRESHOLDS:
    print(f"\n  Threshold = {t}")
    y_pred                              = predict(X_test, w, b, threshold=t)
    TP, TN, FP, FN                      = get_confusion_values(Y_test, y_pred)
    Accuracy, Precision, Recall, F1     = get_scores(TP, TN, FP, FN)
    print_scores(Accuracy, Precision, Recall, F1)
    accuracies.append(Accuracy)
    precisions.append(Precision)
    recalls.append(Recall)
    f1s.append(F1)

# --- Plot Threshold Metrics ---
plt.figure()
plt.plot(THRESHOLDS, accuracies,  label='Accuracy')
plt.plot(THRESHOLDS, precisions,  label='Precision')
plt.plot(THRESHOLDS, recalls,     label='Recall')
plt.plot(THRESHOLDS, f1s,         label='F1 Score')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Metrics vs Threshold")
plt.legend()
plt.tight_layout()
plt.show()