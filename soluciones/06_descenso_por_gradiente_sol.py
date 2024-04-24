# Attribution 4.0 International
# Juan Irving Vasquez
# jivg.org

import numpy as np
from data_prep import features, targets, features_test, targets_test
import matplotlib.pyplot as plt
from nni import sigmoid, sigmoid_prime

np.random.seed(42)

def main():
    n_records, n_features = features.shape
    last_loss = None

    # Weights initialization
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

    epochs = 200
    learnrate = 0.5

    History_loss = []

    for e in range(epochs):
        incremento_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            h = np.dot(x, weights)
            output = sigmoid(h)

            error = y - output
            delta = error*sigmoid_prime(h)
            incremento_w += delta * x

        m = len(features.values)
        weights += (learnrate/m) * incremento_w

        # para registro y visualización
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        History_loss.append(loss)
        if e % (epochs / 10) == 0:
            if last_loss and last_loss < loss:
                print("Epoch:", e, " Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Epoch:", e, " Train loss: ", loss)
            last_loss = loss

    tes_out = sigmoid(np.dot(features_test, weights))
    predictions = tes_out > 0.5
    accuracy = np.mean(predictions == targets_test)
    print("Prediction accuracy: {:.3f}".format(accuracy))

    plt.plot(History_loss)
    plt.title("Rendimiento del entrenamiento")
    plt.ylabel('Error')
    plt.xlabel('Época')
    plt.show()

if __name__ == "__main__":
    main()