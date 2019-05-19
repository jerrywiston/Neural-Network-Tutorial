import numpy as np
import matplotlib.pyplot as plt
import MNIST_tools
#%%
# Hyper Parameters
input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 1e-2
batch_size = 64
#%%
# Data Points
MNIST_tools.downloadMNIST(path='MNIST_data', unzip=True)
x_train, y_train = MNIST_tools.loadMNIST(dataset="training", path="MNIST_data")
x_test, y_test = MNIST_tools.loadMNIST(dataset="testing", path="MNIST_data")
x_train = x_train.astype(np.float32) / 255.
x_test = x_test.astype(np.float32) / 255.
data_size = x_train.shape[0]

# Model Parameters
W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
b1 = np.zeros([1, hidden_size])
W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
b2 = np.zeros([1, output_size])
#%%
# Util Function
def OneHot(y):
    y_one_hot = np.zeros([y.shape[0], 10], dtype=np.float32)
    for i in range(y.shape[0]):
        y_one_hot[i][y[i]] = 1
    return y_one_hot

def Softmax(s):
    out_logit = np.exp(s - np.max(s,1)[:, np.newaxis])
    out_prob = out_logit / np.sum(out_logit, 1, keepdims=True)
    return out_prob, out_logit

def CrossEntropyLoss(y,y_):
    temp = y_ * np.log(y + 1e-6)
    temp = -np.sum(temp, 1)
    temp = np.mean(temp)
    return temp

def Accuracy(y,y_):
    y_digit = np.argmax(y,1)
    y_digit_ = np.argmax(y_,1)
    temp = np.equal(y_digit, y_digit_).astype(np.float32)
    return np.sum(temp) / float(y_digit.shape[0])

# Neural Network Model
def Predict(x_, w1, b1, w2, b2):
    n = x_.shape[0]
    a1 = x_.dot(W1) + np.ones((n,1)).dot(b1)
    h1 = np.maximum(a1,0)
    a2 = h1.dot(W2) + np.ones((n,1)).dot(b2)
    y_prob, y_logit = Softmax(a2)
    return y_prob, y_logit

# Training the Model
loss_rec = []
for i in range(20001):
    # Sample Data Batch
    batch_id = np.random.choice(data_size, batch_size)
    x_ = x_train[batch_id]
    y_ = y_train[batch_id]
    y_ = OneHot(y_)
    
    # Forward Propagation
    n = x_.shape[0]
    a1 = x_.dot(W1) + np.ones((n,1)).dot(b1)
    h1 = np.maximum(a1,0)
    a2 = h1.dot(W2) + np.ones((n,1)).dot(b2)
    y_prob, y_logit = Softmax(a2)
    loss = CrossEntropyLoss(y_, y_prob)
    loss_rec.append(loss)

    # Accuracy
    batch_id = np.random.choice(y_test.shape[0], batch_size)
    x_test_ = x_test[batch_id]
    y_test_ = y_test[batch_id]
    y_test_prob, y_test_logit = Predict(x_test_, W1, b1, W2, b2)
    acc = Accuracy(y_test_prob, OneHot(y_test_))

    if i%100 == 0:
        print("Iter:", i, "Loss:", loss, "Acc:", acc)

    # Backward Propagation
    grad_a2 = y_prob - y_
    grad_b2 = np.ones((n, 1)).T.dot(grad_a2)
    grad_W2 = h1.T.dot(grad_a2)
    grad_h1 = grad_a2.dot(W2.T)
    grad_a1 = grad_h1.copy()
    grad_a1[a1 < 0] = 0
    grad_b1 = np.ones((n, 1)).T.dot(grad_a1)
    grad_W1 = x_.T.dot(grad_a1)

    # Update Parameters
    W1 = W1 - learning_rate * grad_W1 / n
    b1 = b1 - learning_rate * grad_b1 / n
    W2 = W2 - learning_rate * grad_W2 / n
    b2 = b2 - learning_rate * grad_b2 / n

y_prob, y_logit = Predict(x_test, W1, b1, W2, b2)
total_acc = Accuracy(y_prob, OneHot(y_test))
print("Total Accuracy:", total_acc)

print("[Show Loss Curve]")
plt.plot(loss_rec)
plt.show()

print("[Visualize Filters]")
for i in range(0):
    f = W1[:,i].reshape([28,28])
    plt.imshow(f, cmap='gray')
    plt.show()