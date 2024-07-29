import sys
import random
from zoo import Zoo
from game import Q20Game
from neural import DualModeNeuralNetwork
import json


SRC = "data/zoo.csv"

# Initialize Zoo Object
Z = Zoo(SRC)
qs = Z.questions
ts = Z.targets
num_qs = len(qs)
num_ts = len(ts)
input_size = len(qs)
output_size = len(ts)
question_limit = 8


# Training Function
def train(nn, training_size, question_limit=13):
    i = 0
    interval = int(training_size / 100)

    game = Q20Game(nn, qs, ts, question_limit)

    for _ in range(int(training_size / len(ts))):
        t = 0
        for target_vector in ts:
            input_vector = game.autoplay(t)
            i += 1
            t += 1

    return nn


# Validation Function
def validate(nn, num_games, question_limit):
    game = Q20Game(nn, qs, ts, question_limit)

    print("\n=======================================")
    print("VALIDATION SET")
    print("=======================================\n")

    wins = 0
    testing_sets = []

    for _ in range(num_games):
        t = random.randint(0, len(ts) - 1)
        target_vector = [0] * len(ts)
        target_vector[t] = 1
        win, input_vector = game.autoplay_smartq(t)
        testing_sets.append([input_vector, target_vector])

        if win == 1:
            wins += 1

    accuracy = float(wins) / float(num_games)
    error = nn.calculate_total_error(testing_sets)

    print("=======================================")
    print(f"{accuracy * 100}% accuracy, error={error}")
    print("=======================================\n")

    return error, accuracy


# Testing Function
def test(nn):
    game = Q20Game(nn, qs, ts, question_limit)
    while True:
        game.play()
        nn.save("saved/zoo.json")


def crossvalidate():
    learning_rate = 0.1

    hidden_size = 64  # Increased hidden layer size for more capacity

    num_weights = int(hidden_size * input_size + output_size * hidden_size)

    nn = DualModeNeuralNetwork(
        num_questions=input_size,
        num_hidden1=hidden_size,
        num_hidden2=hidden_size,
        num_targets=output_size,
        learning_rate=learning_rate,
        activation_func="relu",
    )

    print("\n=======================================")
    print("NETWORK SUMMARY")
    print("=======================================\n")

    print(f"#input = {input_size}")
    print(f"#output = {output_size}")
    print(f"#hidden = {hidden_size}")
    print(f"learning rate = {learning_rate}")
    print(f"#weights = {num_weights}")

    upper_epoch = min(1000000, round(num_weights * num_weights, 100))
    interval = int(upper_epoch / 10)
    validate_epoch = min(200, int(interval / 10))

    print(f"upper epoch = {upper_epoch}")
    print(f"interval = {interval}")
    print(f"#validation sets = {validate_epoch}")
    print(f"error breakpoint <= {validate_epoch / 13.3}")  # chaged to 13.3

    i = 0

    epoch = 0

    epoch_intervals = []
    epoch_errors = []
    epoch_accuracy = []

    while epoch < upper_epoch:
        epoch += interval

        print("\n=======================================")
        print(f"{i}. # TRAINING SETS: {epoch}")
        print("=======================================\n")

        train(nn, interval, question_limit)
        error, accuracy = validate(nn, validate_epoch, question_limit)

        epoch_intervals.append(epoch)
        epoch_errors.append(error)
        epoch_accuracy.append(accuracy)

        if error < validate_epoch / 13.3:  # changed to 13.3
            break

        i += 1

    nn.save("saved/zoo.json")
    return nn


def hyperparameter_training(lr, ls, question_limit, layer):
    print("\n=======================================")
    print("HYPERPARAMETER TRAINING")
    print("Learning Rate:", lr)
    print("Layer Size:", ls)
    print("No. of Questions:", question_limit)
    print("Activation Layer:", layer)
    print("=======================================\n")

    nn = DualModeNeuralNetwork(
        num_questions=input_size,
        num_hidden1=ls,
        num_hidden2=ls,
        num_targets=output_size,
        learning_rate=lr,
        activation_func=layer,
    )

    interval = 100000
    validate_epoch = 100

    train(nn, interval, question_limit)
    error, accuracy = validate(nn, validate_epoch, question_limit)

    if error < validate_epoch / 13.3:  # changed to 13.3
        return accuracy, error

    return accuracy, error


def hyperparameter_tuning():
    learning_rates = [0.1, 0.01, 0.001]
    layer_sizes = [32, 64, 128]
    question_limit_set = [20, 15, 10, 8]
    activation_layers = ["sigmoid", "relu", "tanh", "softmax"]

    hyperparameter_info = {}

    for lr in learning_rates:
        for ls in layer_sizes:
            for no_of_question in question_limit_set:
                for layer in activation_layers:

                    accuracy, error = hyperparameter_training(
                        lr, ls, no_of_question, layer
                    )

                    hyperparameter_info[lr] = {
                        ls: {
                            no_of_question: {
                                layer: {"accuracy": accuracy, "error": error}
                            }
                        }
                    }

    print(hyperparameter_info)
    with open("saved/hyperparameter_info.json", "w") as file:
        json.dump(hyperparameter_info, file)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "train":
            # Perform training
            nn = crossvalidate()
        elif mode == "test":
            # Perform testing
            nn = DualModeNeuralNetwork(
                num_questions=input_size,
                num_hidden1=64,
                num_hidden2=64,
                num_targets=output_size,
                learning_rate=0.1,
                activation_func="relu",
            )
            nn.load("saved/zoo.json")
            test(nn)
        elif mode == "hyperparameter":
            # Perform hyperparameter tuning
            hyperparameter_tuning()
        else:
            print("Invalid mode. Please specify 'train' or 'test'.")
    else:
        print("No mode specified. Please specify 'train' or 'test'.")
