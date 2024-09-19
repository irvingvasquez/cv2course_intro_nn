# Attribution 4.0 International
# Juan Irving Vasquez
# jivg.org

import numpy as np
import nni

x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = nni.sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = nni.sigmoid(output_layer_in)

## Backwards pass
## TODO: Calculate error
error = target-output

# TODO: Calculate error gradient for output layer
del_err_output = error * nni.sigmoid_prime(output_layer_in)

# TODO: Calculate error gradient for hidden layer
del_err_hidden = del_err_output * np.multiply(weights_hidden_output, nni.sigmoid_prime(hidden_layer_input))


# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * del_err_output * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate * (del_err_hidden * x[:,None])

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)


