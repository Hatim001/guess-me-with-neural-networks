import json
import numpy as np


# Neural Network Class
class DualModeNeuralNetwork:
    def __init__(
        self,
        num_questions,
        num_hidden1,
        num_hidden2,
        num_targets,
        learning_rate=0.01,
        activation_func="relu",
    ):
        self.num_questions = num_questions
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.num_targets = num_targets
        self.learning_rate = learning_rate

        # Initialize weights and biases for hidden and output layers
        self.hidden_weights1 = np.random.randn(num_hidden1, num_questions) * np.sqrt(
            2.0 / num_questions
        )
        self.hidden_bias1 = np.zeros(num_hidden1)
        self.hidden_weights2 = np.random.randn(num_hidden2, num_hidden1) * np.sqrt(
            2.0 / num_hidden1
        )
        self.hidden_bias2 = np.zeros(num_hidden2)
        self.output_weights = np.random.randn(num_targets, num_hidden2) * np.sqrt(
            2.0 / num_hidden2
        )
        self.output_bias = np.zeros(num_targets)
        self.prepare_activation_functions(activation_func)

    def prepare_activation_functions(self, activation_func):
        function_mapper = {
            "relu": {
                "activation": self.relu,
                "derivative": self.relu_derivative,
            },
            "sigmoid": {
                "activation": self.sigmoid,
                "derivative": self.sigmoid_derivative,
            },
            "tanh": {
                "activation": self.tanh,
                "derivative": self.tanh_derivative,
            },
            "softmax": {
                "activation": self.softmax,
                "derivative": self.softmax_derivative,
            },
        }

        self.activation_function = function_mapper[activation_func]["activation"]
        self.activation_function_derivative = function_mapper[activation_func][
            "derivative"
        ]

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def softmax_derivative(self, x):
        s = self.softmax(x)
        return s * (1 - s)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1.0 - np.tanh(x) ** 2

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feed_forward_questions_to_targets(self, inputs):
        hidden_layer_activation1 = (
            np.dot(self.hidden_weights1, inputs) + self.hidden_bias1
        )
        hidden_layer_output1 = self.activation_function(hidden_layer_activation1)

        hidden_layer_activation2 = (
            np.dot(self.hidden_weights2, hidden_layer_output1) + self.hidden_bias2
        )
        hidden_layer_output2 = self.activation_function(hidden_layer_activation2)

        output_layer_activation = (
            np.dot(self.output_weights, hidden_layer_output2) + self.output_bias
        )
        output = self.sigmoid(output_layer_activation)

        return output

    def feed_forward_targets_to_questions(self, target_inputs):
        hidden_layer_activation2 = (
            np.dot(self.output_weights.T, target_inputs) + self.output_bias
        )
        hidden_layer_output2 = self.activation_function(hidden_layer_activation2)

        hidden_layer_activation1 = (
            np.dot(self.hidden_weights2.T, hidden_layer_output2) + self.hidden_bias2
        )
        hidden_layer_output1 = self.activation_function(hidden_layer_activation1)

        question_layer_activation = (
            np.dot(self.hidden_weights1.T, hidden_layer_output1) + self.hidden_bias1
        )
        output = self.sigmoid(question_layer_activation)

        return output

    def backpropagate(self, inputs, expected_output):
        hidden_layer_activation1 = (
            np.dot(self.hidden_weights1, inputs) + self.hidden_bias1
        )
        hidden_layer_output1 = self.activation_function(hidden_layer_activation1)

        hidden_layer_activation2 = (
            np.dot(self.hidden_weights2, hidden_layer_output1) + self.hidden_bias2
        )
        hidden_layer_output2 = self.activation_function(hidden_layer_activation2)

        output_layer_activation = (
            np.dot(self.output_weights, hidden_layer_output2) + self.output_bias
        )
        output = self.sigmoid(output_layer_activation)

        # Calculate output layer error
        output_layer_error = expected_output - output
        output_layer_delta = output_layer_error * self.sigmoid_derivative(output)

        # Calculate hidden layer errors
        hidden_layer_error2 = np.dot(self.output_weights.T, output_layer_delta)
        hidden_layer_delta2 = hidden_layer_error2 * self.activation_function_derivative(
            hidden_layer_output2
        )

        hidden_layer_error1 = np.dot(self.hidden_weights2.T, hidden_layer_delta2)
        hidden_layer_delta1 = hidden_layer_error1 * self.activation_function_derivative(
            hidden_layer_output1
        )

        # Update the weights and biases
        self.output_weights += self.learning_rate * np.outer(
            output_layer_delta, hidden_layer_output2
        )
        self.output_bias += self.learning_rate * output_layer_delta

        self.hidden_weights2 += self.learning_rate * np.outer(
            hidden_layer_delta2, hidden_layer_output1
        )
        self.hidden_bias2 += self.learning_rate * hidden_layer_delta2

        self.hidden_weights1 += self.learning_rate * np.outer(
            hidden_layer_delta1, inputs
        )
        self.hidden_bias1 += self.learning_rate * hidden_layer_delta1

    def adjust_weights(self, inputs, question_index, target_index, answer):
        adjustment = answer * self.learning_rate
        hidden_layer_activation1 = (
            np.dot(self.hidden_weights1, inputs) + self.hidden_bias1
        )
        hidden_layer_output1 = self.activation_function(hidden_layer_activation1)

        hidden_layer_activation2 = (
            np.dot(self.hidden_weights2, hidden_layer_output1) + self.hidden_bias2
        )
        hidden_layer_output2 = self.activation_function(hidden_layer_activation2)

        self.output_weights[target_index, :] += adjustment * hidden_layer_output2
        self.output_bias[target_index] += adjustment

        self.hidden_weights2[question_index, :] += adjustment * hidden_layer_output1
        self.hidden_bias2[question_index] += adjustment

        self.hidden_weights1[question_index, :] += adjustment * inputs
        self.hidden_bias1[question_index] += adjustment

    def rank_targets(self, inputs):
        return self.feed_forward_questions_to_targets(inputs)

    def rank_questions(self, target_output):
        return self.feed_forward_targets_to_questions(target_output)

    def calculate_total_error(self, testing_sets):
        total_error = 0
        for input_vector, target_vector in testing_sets:
            output = self.feed_forward_questions_to_targets(np.array(input_vector))
            total_error += 0.5 * sum((np.array(target_vector) - output) ** 2)
        return total_error

    def save(self, filename):
        model = {
            "hidden_weights1": self.hidden_weights1.tolist(),
            "hidden_bias1": self.hidden_bias1.tolist(),
            "hidden_weights2": self.hidden_weights2.tolist(),
            "hidden_bias2": self.hidden_bias2.tolist(),
            "output_weights": self.output_weights.tolist(),
            "output_bias": self.output_bias.tolist(),
        }
        with open(filename, "w") as f:
            json.dump(model, f)

    def load(self, filename):
        with open(filename, "r") as f:
            model = json.load(f)
            self.hidden_weights1 = np.array(model["hidden_weights1"])
            self.hidden_bias1 = np.array(model["hidden_bias1"])
            self.hidden_weights2 = np.array(model["hidden_weights2"])
            self.hidden_bias2 = np.array(model["hidden_bias2"])
            self.output_weights = np.array(model["output_weights"])
            self.output_bias = np.array(model["output_bias"])
