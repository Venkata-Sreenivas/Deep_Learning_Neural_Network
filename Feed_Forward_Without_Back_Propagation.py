import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

#activation function
def sigmoid(x):
  return 1/(1+np.exp(-x))

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

#preprocessing
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values.reshape(-1, 1)
#scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

#depth and breadth
input_size = X_train.shape[1]
hidden_size = 12
output_size = 1

# random weights
np.random.seed(42)
w1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
w2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))


#feed forward
z1_test = np.dot(X_test, w1) + b1
a1_test = sigmoid(z1_test)
z2_test = np.dot(a1_test, w2) + b2
predictions = sigmoid(z2_test)

#results
final_preds = (predictions > 0.5).astype(int)

#comparisons
results_comparison = pd.DataFrame({
    'Actual': y_test.flatten(),
    'Predicted': final_preds.flatten(),
    'Confidence Score': np.round(predictions.flatten(), 4)
})

print("\nFirst 5 Predictions (Random Weights):")
print(results_comparison.head())

cm = confusion_matrix(y_test, final_preds)
tn, fp, fn, tp = cm.ravel()
accuracy = accuracy_score(y_test, final_preds) * 100
print(f"\nAccuracy: {accuracy:.2f}%")
print("\nConfusion Matrix:")
print(cm)
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
