import numpy as np
import matplotlib.pyplot as plt

# Hyper Parameters
input_size = 1
hidden_size = 32
output_size = 1
learning_rate = 1e-3
batch_size = 256

# Data Points
data_size = 1000
x_train = np.linspace(-5, 5, data_size).reshape(-1,1)
y_gt = np.power(x_train, 2)
y_train = y_gt + np.random.randn(y_gt.shape[0], y_gt.shape[1])

# Model Parameters
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(1, hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(1, output_size)

# Neural Network Model
def Predict(x_, w1, b1, w2, b2):
    n = x_.shape[0]
    a1 = x_.dot(W1) + np.ones((n,1)).dot(b1)
    h1 = np.maximum(a1,0)
    y_pred = h1.dot(W2) + np.ones((n,1)).dot(b2)
    return y_pred

# Training the Model
loss_rec = []
for i in range(20001):
    # Sample Data Batch
    batch_id = np.random.choice(data_size, batch_size)
    x_ = x_train[batch_id]
    y_ = y_train[batch_id]
    
    # Forward Propagation
    n = x_.shape[0]
    a1 = x_.dot(W1) + np.ones((n,1)).dot(b1)
    h1 = np.maximum(a1,0)
    y_pred = h1.dot(W2) + np.ones((n,1)).dot(b2)
    loss = 0.5 * np.square(y_pred - y_).mean()
    loss_rec.append(loss)

    if i%100 == 0:
        print("Iter:", i, ",Loss:", loss)

    # Backward Propagation
    grad_y_pred = y_pred - y_
    grad_b2 = np.ones((n, 1)).T.dot(grad_y_pred)
    grad_W2 = h1.T.dot(grad_y_pred)
    grad_h1 = grad_y_pred.dot(W2.T)
    grad_a1 = grad_h1.copy()
    grad_a1[a1 < 0] = 0
    grad_b1 = np.ones((n, 1)).T.dot(grad_a1)
    grad_W1 = x_.T.dot(grad_a1)

    # Update Parameters
    W1 = W1 - learning_rate * grad_W1 / n
    b1 = b1 - learning_rate * grad_b1 / n
    W2 = W2 - learning_rate * grad_W2 / n
    b2 = b2 - learning_rate * grad_b2 / n

y_pred = Predict(x_train, W1, b1, W2, b2)
total_loss = 0.5 * np.square(y_pred - y_train).mean()
print("Total Loss:", total_loss)

print("[Show Fitting Curve]")
plt.plot(x_train, y_train,'b.')
plt.plot(x_train, y_gt, 'g.')
plt.plot(x_train, y_pred,'r.')
plt.show()

print("[Show Loss Curve]")
plt.plot(loss_rec[100:])
plt.show()