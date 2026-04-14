#lib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
def sigmoid(x):
  return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
  return x*(1-x)
#load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)
#preproccesing
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values.reshape(-1, 1)
#scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
#splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Hyperparameters
input_size = X_train.shape[1]
hidden_size = 12
output_size = 1
learning_rate = 0.1
epochs = 10000
#initialize weights and bias
np.random.seed(42)
w1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
w2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))
history = []
#training
for i in range(epochs):
  # feed forward
  z1 = np.dot(X_train, w1) + b1
  a1 = sigmoid(z1)
  z2 = np.dot(a1, w2) + b2
  a2 = sigmoid(z2)
  #back propagation
  error = y_train - a2
  # Calculate loss
  loss = np.mean(np.square(error))
  history.append(loss)
  #gradients
  d_a2 = error * sigmoid_derivative(a2) #output layer
  d_a1 = d_a2.dot(w2.T) * sigmoid_derivative(a1) #hidden layer
  #update weights
  w2 += a1.T.dot(d_a2) * learning_rate
  b2 += np.sum(d_a2, axis=0, keepdims=True) * learning_rate
  w1 += X_train.T.dot(d_a1) * learning_rate
  b1 += np.sum(d_a1, axis=0, keepdims=True) * learning_rate
print(f"Epoch Final: Loss = {loss:.4f}")
#Testing
z1_test = np.dot(X_test, w1) + b1
a1_test = sigmoid(z1_test)
z2_test = np.dot(a1_test, w2) + b2
predictions = sigmoid(z2_test)
final_preds = (predictions > 0.5).astype(int)
#comparisons
results_comparison = pd.DataFrame({
    'Actual': y_test.flatten(),
    'Predicted': final_preds.flatten(),
    'Confidence Score': np.round(predictions.flatten(), 4)
})
print("\nFirst 5 Predictions:")
print(results_comparison.head())

cm = confusion_matrix(y_test, final_preds)
print("\nConfusion Matrix:")
print(cm)
tn, fp, fn, tp = cm.ravel()
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
accuracy = accuracy_score(y_test, final_preds)*100
print(f"Accuracy: {accuracy:.2f}%")
