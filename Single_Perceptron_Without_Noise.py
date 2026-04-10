#lib
import numpy as np
#activation function
def step_function(z):
  return 1 if z >= 0 else 0
#input
X = np.array([
    [0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1],
    [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1],
    [1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1],
    [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1]
])
#output
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
#weight & bias
w = np.zeros(4)
b = 0.0
learning_rate = 0.1
epochs = 20
#training
print(f"{'Epoch':<8} {'Error Count':<12} {'Bias':<10} {'Weights'}")
print("-" * 50)
for epoch in range(epochs):
    total_error = 0
    for i in range(len(X)):
      z = (w[0]*X[i][0]) + (w[1]*X[i][1]) + (w[2]*X[i][2]) + (w[3]*X[i][3]) + b
      y_pred = step_function(z)
      error = y[i] - y_pred

      if error != 0:
        total_error += 1
        # Update weights and bias
        w[0] += learning_rate * error * X[i][0]
        w[1] += learning_rate * error * X[i][1]
        w[2] += learning_rate * error * X[i][2]
        w[3] += learning_rate * error * X[i][3]
        b += learning_rate * error

    print(f"{epoch+1:<8} {total_error:<12} {b:<10.2f} {w}")

    if total_error == 0:
        print("-" * 50)
        print("Converged! Training Complete.")
        break
#Result
print("\nFinal Testing:")
print(f"Final Weights: {w}")
print(f"Final Bias: {b}")
#Test
test_input = [1, 1, 0, 1]
final_z = (w[0]*test_input[0]) + (w[1]*test_input[1]) + (w[2]*test_input[2]) + (w[3]*test_input[3]) + b
print(f"Input: {test_input} -> Prediction: {step_function(final_z)}")
