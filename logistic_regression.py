import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([0,0,0,0,1,1,1,1,1,1])

X_b = np.c_[np.ones((len(X),1)),X]

def sigmoid(z):
    return 1/(1+ np.exp(-z))

theta = np.zeros(X_b.shape[1])

def predict_probability(X, theta):
    z = X @ theta
    return sigmoid(z)

def compute_loss(X, y, theta):
    m = len(y)
    y_pred = predict_probability(X, theta)

    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -(1/m) * np.sum(y * np.log(y_pred) + (1-y) * np.log(1 - y_pred))
    return loss

def gradient_descent(X, y, theta, lr, epochs):
    m = len(y)
    loss_history = []

    for i in range(epochs):
        y_pred = predict_probability(X, theta)
        gradient = (1/m) * X.T @ (y_pred - y)
        theta = theta - lr * gradient
        loss_history.append(compute_loss(X, y, theta))

        if i % 100 == 0:
            print(f"Epoch : {i}, Loss: {loss_history[-1]}")
    return theta, loss_history

learning_rate = 0.1
epochs = 1000
theta, loss_history = gradient_descent(X_b, y, theta, learning_rate, epochs)
print("Final predictions:")

probabilities = predict_probability(X_b, theta)
print(probabilities)
def predict(X, theta, threshold=0.5):
    probabilities = predict_probability(X, theta)
    return (probabilities >= threshold).astype(int)

predictions = predict(X_b, theta)

accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy}")

X_new = [2, 4, 5, 7, 11]
X_new_b = np.c_[np.ones((5,1)),X_new]
new_probability = predict_probability(X_new_b, theta)
new_predictions = predict(X_new_b, theta)
print("New predictions:")
for ads_seen, probability, prediction in zip(X_new, new_probability, new_predictions):
    print(f"Ads seen: {ads_seen}",
        f"--> Probability of buying: {probability:.4f} "
        f"--> Prediction: {prediction}")

X_curve = np.linspace(1, 10, 100)
X_curve_b = np.c_[np.ones((len(X_curve), 1)), X_curve]

probability_curve = predict_probability(X_curve_b, theta)

plt.figure()

plt.scatter(X, y, label="Training data")
plt.plot(X_curve, probability_curve, label="Sigmoid curve")
plt.axhline(0.5, linestyle="--", label="Decision threshold")

plt.xlabel("Ads Seen")
plt.ylabel("Probability of Buying")
plt.title("Logistic Regression Decision Curve")
plt.legend()

plt.savefig("/Users/subekxya/Documents/ML/logistic-regression-from-scratch/results/logistic_curve.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Logistic Regression Training")

plt.savefig("/Users/subekxya/Documents/ML/logistic-regression-from-scratch/results/loss_convergence.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nLearned parameters:")
print(f"Intercept (theta_0): {theta[0]:.4f}")
print(f"Slope (theta_1): {theta[1]:.4f}")
print(f"Final Loss: {loss_history[-1]:.4f}")
print(f"Training Accuracy: {accuracy:.4f}")