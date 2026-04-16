
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- 1. Helper Functions ---

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def l1_regularization(weights, lambda_val):
    """Returns cost and gradient for L1"""
    cost = lambda_val * np.sum(np.abs(weights))
    grad = lambda_val * np.sign(weights)
    return cost, grad

def l2_regularization(weights, lambda_val):
    """Returns cost and gradient for L2"""
    cost = lambda_val * np.sum(np.square(weights))
    grad = 2 * lambda_val * weights
    return cost, grad

# --- 2. Data Loading & Preprocessing ---

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Training Function ---

def train_model(reg_type, lambda_val=0.01):
    print(f"\n--- Training with {reg_type.upper()} Regularization ---")

    # Re-initialize weights for fair comparison
    np.random.seed(42)
    input_size = X_train.shape[1]
    hidden_size = 12
    output_size = 1

    w1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    w2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    learning_rate = 0.01
    epochs = 1000
    m = y_train.shape[0]

    for epoch in range(epochs):
        # Forward
        z1 = np.dot(X_train, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        output = sigmoid(z2)

        # Apply Regularization
        if reg_type == 'l1':
            cost_w1, grad_w1 = l1_regularization(w1, lambda_val)
            cost_w2, grad_w2 = l1_regularization(w2, lambda_val)
        else: # l2
            cost_w1, grad_w1 = l2_regularization(w1, lambda_val)
            cost_w2, grad_w2 = l2_regularization(w2, lambda_val)

        # Loss Calculation
        base_loss = -np.mean(y_train * np.log(output + 1e-8) + (1 - y_train) * np.log(1 - output + 1e-8))
        total_loss = base_loss + (cost_w1 + cost_w2) / m

        # Backward Pass
        error_output = output - y_train

        d_w2 = np.dot(a1.T, error_output) + grad_w2
        d_b2 = np.sum(error_output, axis=0, keepdims=True)

        error_hidden = np.dot(error_output, w2.T) * sigmoid_derivative(a1)

        d_w1 = np.dot(X_train.T, error_hidden) + grad_w1
        d_b1 = np.sum(error_hidden, axis=0, keepdims=True)

        # Update
        w2 -= learning_rate * (d_w2 / m)
        b2 -= learning_rate * (d_b2 / m)
        w1 -= learning_rate * (d_w1 / m)
        b1 -= learning_rate * (d_b1 / m)

    # Evaluation
    z1_test = np.dot(X_test, w1) + b1
    a1_test = sigmoid(z1_test)
    z2_test = np.dot(a1_test, w2) + b2
    predictions = sigmoid(z2_test)
    final_preds = (predictions > 0.5).astype(int)

    acc = accuracy_score(y_test, final_preds) * 100
    print(f"Final Loss: {total_loss:.4f}")
    print(f"Accuracy with {reg_type.upper()}: {acc:.2f}%")
    return acc

# --- 4. Execution ---

# Train first time with L1
acc_l1 = train_model('l1', lambda_val=0.01)

# Train second time with L2
acc_l2 = train_model('l2', lambda_val=0.01)

# Summary
print("\n--- Summary ---")
print(f"L1 Accuracy: {acc_l1:.2f}%")
print(f"L2 Accuracy: {acc_l2:.2f}%")
