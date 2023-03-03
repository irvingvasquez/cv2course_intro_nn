import numpy as np
from data_prep import features, targets, features_test, targets_test


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))
    
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))


np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

epochs = 1000
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        h = np.dot(x, weights)

        output = sigmoid(h)

        error = y - sigmoid(np.dot(x, weights))

        del_w += learnrate * (error*sigmoid_prime(h)) * x

    m = len(features.values)
    weights += 1/m * del_w


    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
