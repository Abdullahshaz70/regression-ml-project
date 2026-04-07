import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, f1_score



df = pd.read_csv("HousingData.csv")


target = "MEDV"
feature_cols = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]

epsilon = 1e-4
alpha = [0.0001, 0.001, 0.01, 0.1, 1.0]


# Replace null values with median of each column
def remove_null_values():
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            median_value = df[column].median()
            df.fillna(median_value, inplace=True)


# Shuffle the dataset randomly
def shuffle_Data():
    global df
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# Split dataset manually into 80% train and 20% test
def training_testing_splitting():
    n = len(df)
    train_data_size = int(0.8 * n)

    train_df = df[:train_data_size]
    test_df = df[train_data_size:]

    X_train = train_df.drop(columns=target)
    Y_train = train_df[target]
    X_test = test_df.drop(columns=target)
    Y_test = test_df[target]

    return X_train, Y_train, X_test, Y_test


# Initialize weights to zero and bias to zero
def initillize_params():
    w = np.zeros(13)
    b = 0
    return w, b


# Linear prediction: Y = X @ w + b
def predict(X, w, b):
    return X @ w + b


# Squared error loss (MSE)
def compute_loss(y_true, y_pred):
    n = len(y_true)
    return (1/n) * np.sum((y_true - y_pred)**2)


# Gradient of loss with respect to w and b
def compute_gradients(X, y_true, y_pred):
    n = len(y_true)
    error = y_true - y_pred
    gradient_w = (-2/n) * X.T @ error
    gradient_b = (-2/n) * np.sum(error)
    return gradient_w, gradient_b


# Gradient descent loop with convergence check
def gradient_descent(X, y, w, b, learning_rate, epsilon):
    iterations = 1000
    losses = []

    for i in range(iterations):
        prediction = predict(X, w, b)
        loss = compute_loss(y, prediction)
        losses.append(loss)

        gradient_w, gradient_b = compute_gradients(X, y, prediction)

        w_new = w - (learning_rate * gradient_w)
        b_new = b - (learning_rate * gradient_b)

        w_old = w.copy()
        w = w_new
        b = b_new


        if np.linalg.norm(w_new - w_old) <= epsilon:
            break

    return w, b, losses



def evaluate(X_test, Y_test, w, b):
    prediction = predict(X_test, w, b)
    loss = compute_loss(Y_test, prediction)
    print(f"Test MSE: {loss:.4f}")

    print("\nSample Predictions vs Actual:")
    for i in range(10):
        print(f"Actual: {Y_test[i]:.2f}  |  Predicted: {prediction[i]:.2f}")

    print("\nFailure Cases (error > 5):")
    for i in range(len(Y_test)):
        if abs(Y_test[i] - prediction[i]) > 5:
            print(f"Actual: {Y_test[i]:.2f} | Predicted: {prediction[i]:.2f} | Error: {abs(Y_test[i] - prediction[i]):.2f}")



def classification_metrics(Y_test, y_pred):
    median = np.median(Y_test)
    Y_test_class = (Y_test >= median).astype(int)
    Y_pred_class = (y_pred >= median).astype(int)

    acc = accuracy_score(Y_test_class, Y_pred_class)
    cm = confusion_matrix(Y_test_class, Y_pred_class)
    f1 = f1_score(Y_test_class, Y_pred_class)

    print(f"Accuracy:         {acc:.4f}")
    print(f"F1-Score:         {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")


# ─── Main Loop: Experiment with 5 different learning rates ───────────────────

for i in range(len(alpha)):
    print(f"\n{'='*50}")
    print(f"Learning Rate: {alpha[i]}")
    print(f"{'='*50}")

    remove_null_values()
    shuffle_Data()
    X_train, Y_train, X_test, Y_test = training_testing_splitting()

    # Convert to numpy arrays
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    Y_train = Y_train.to_numpy()
    Y_test = Y_test.to_numpy()

    # Standardize features using training mean and std
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    w, b = initillize_params()
    w, b, losses = gradient_descent(X_train, Y_train, w, b, alpha[i], epsilon)

    print(f"Final Training Loss: {losses[-1]}")
    print(f"Learned b: {b:.4f}")
    print(f"Learned w: {w}")

    # Plot training loss vs iterations
    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss (MSE)")
    plt.title(f"Training Loss vs Iterations (lr={alpha[i]})")
    plt.grid(True)
    plt.show()

    # Evaluate on test data
    evaluate(X_test, Y_test, w, b)

    # Compare with sklearn
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, Y_train)
    sklearn_pred = sklearn_model.predict(X_test)

    your_loss = compute_loss(Y_test, predict(X_test, w, b))
    sklearn_loss = compute_loss(Y_test, sklearn_pred)

    print(f"\nYour Model MSE:    {your_loss:.4f}")
    print(f"Sklearn Model MSE: {sklearn_loss:.4f}")

    # Classification metrics
    y_pred_test = predict(X_test, w, b)
    classification_metrics(Y_test, y_pred_test)


