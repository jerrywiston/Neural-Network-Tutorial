import numpy as np
import matplotlib.pyplot as plt
import MNIST_tools
import os
import matplotlib.gridspec as gridspec

# Hyper Parameters
input_size = 784
hidden_size = 128
output_size = 784
learning_rate = 1e-3
batch_size = 16

# Data Points
path = 'MNIST_data'
MNIST_tools.downloadMNIST(path=path, unzip=True)
x_train, y_train = MNIST_tools.loadMNIST(dataset="training", path=path)
x_test, y_test = MNIST_tools.loadMNIST(dataset="testing", path=path)
x_train = x_train.astype(np.float32) / 255.
x_test = x_test.astype(np.float32) / 255.
data_size = x_train.shape[0]

# Model Parameters
W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
b1 = 0.1 * np.ones([1, hidden_size])
W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
b2 = 0.1 * np.ones([1, output_size])

# Util Function
def PlotFig(samples, fig_size, samp_size):
	fig = plt.figure(figsize=(fig_size[0], fig_size[1]))
	gs = gridspec.GridSpec(fig_size[0], fig_size[1])
	gs.update(wspace=0.05, hspace=0.05)

	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
        
		plt.imshow(sample.reshape(samp_size), cmap='gray')
        
	return fig

def SaveFig(fname, samples, fig_size, samp_size):
 	fig = PlotFig(samples, fig_size, samp_size)
 	plt.savefig(fname, bbox_inches='tight')
 	plt.close(fig)

def Sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))

def CrossEntropyLoss(y,y_):
    temp = - y_ * np.log(y + 1e-6) - (1-y_) * np.log(1-y + 1e-6)
    temp = np.mean(temp)
    return temp

# Neural Network Model
def Predict(x_, w1, b1, w2, b2):
    n = x_.shape[0]
    a1 = x_.dot(W1) + np.ones((n,1)).dot(b1)
    h1 = np.maximum(a1,0)
    a2 = h1.dot(W2) + np.ones((n,1)).dot(b2)
    y_prob = Sigmoid(a2)
    return y_prob

# Training the Model
loss_rec = []
out_folder = 'out/'
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

for i in range(20001):
    # Sample Data Batch
    batch_id = np.random.choice(data_size, batch_size)
    x_ = x_train[batch_id]
    corrupt = np.random.choice(2, x_.shape, p=[0.9, 0.1]).astype(np.float32)
    x_ = np.maximum(x_, corrupt)
    y_ = x_train[batch_id]

    # Forward Propagation
    n = x_.shape[0]
    a1 = x_.dot(W1) + np.ones((n,1)).dot(b1)
    h1 = np.maximum(a1,0)
    drop = np.random.choice(2, h1.shape, p=[0.4, 0.6]).astype(np.float32)
    h1 = h1 * drop
    a2 = h1.dot(W2) + np.ones((n,1)).dot(b2)
    y_prob = Sigmoid(a2)
    loss = CrossEntropyLoss(y_, y_prob)
    loss_rec.append(loss)

    if i%100 == 0:
        print("Iter:", i, "Loss:", loss)
    
    if i%1000 == 0:
        print("Save Fig ...")
        x_fig = np.concatenate((x_[0:4], y_prob[0:4], x_[4:8], y_prob[4:8]), axis=0)
        samp_name = out_folder + str(int(i/1000)).zfill(4) + '.png'
        SaveFig(samp_name, x_fig, [4,4], [28,28])

        f_fig = W1[:,0:16].T
        filter_name = out_folder + str(int(i/1000)).zfill(4) + '_filter.png'
        SaveFig(filter_name, f_fig, [4,4], [28,28])

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

print("[Show Loss Curve]")
plt.plot(loss_rec)
plt.show()
    