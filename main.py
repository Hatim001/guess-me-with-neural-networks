import sys
import random
import numpy as np
from zoo import Zoo
from game import Q20Game
from neural import DualModeNeuralNetwork
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

SRC = "data/zoo.csv"

# Initialize Zoo Object
Z = Zoo(SRC)
qs = Z.questions
ts = Z.targets
num_qs = len(qs)
num_ts = len(ts)
input_size = len(qs)
output_size = len(ts)


# Training Function
def train(nn, training_size, question_limit):
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
    all_predictions = []
    all_targets = []

    for _ in range(num_games):
        t = random.randint(0, len(ts) - 1)
        target_vector = [0] * len(ts)
        target_vector[t] = 1
        win, input_vector = game.autoplay_smartq(t)
        testing_sets.append([input_vector, target_vector])

        if win == 1:
            wins += 1

        # Collect all targets and predictions
        all_targets.append(target_vector)
        output = nn.feed_forward_questions_to_targets(np.array(input_vector))
        # Binarize the predictions with a threshold of 0.5
        binary_output = (output >= 0.5).astype(int)
        all_predictions.append(binary_output)

    accuracy = float(wins) / float(num_games)
    error = nn.calculate_total_error(testing_sets)

    # Convert lists to numpy arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)

    # Calculate additional evaluation metrics
    precision = precision_score(all_targets, all_predictions, average="micro")
    recall = recall_score(all_targets, all_predictions, average="micro")
    f1 = f1_score(all_targets, all_predictions, average="micro")
    auc = roc_auc_score(all_targets, all_predictions, average="micro")

    print('=======================================')
    print(f'{accuracy * 100}% accuracy, error={error}')
    print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc}')
    print('=======================================\n')

    return error, accuracy


# Cross Validation Function
def crossvalidate(learning_rate, hidden_size, activation_func, question_limit):
    num_weights = int(hidden_size * input_size + output_size * hidden_size)

    nn = DualModeNeuralNetwork(
        num_questions=input_size,
        num_hidden1=hidden_size,
        num_hidden2=hidden_size,
        num_targets=output_size,
        learning_rate=learning_rate,
        activation_func=activation_func,
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
    print(f"error breakpoint <= {validate_epoch / 13.3}")  # changed to 13.3

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

    model_filename = f"saved/zoo_{question_limit}.json"
    nn.save(model_filename)
    return nn


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "train":
            if len(sys.argv) != 6:
                print(
                    "Usage: python main.py train <learning_rate> <hidden_size> <activation_func> <question_limit>"
                )
                sys.exit(1)

            learning_rate = float(sys.argv[2])
            hidden_size = int(sys.argv[3])
            activation_func = sys.argv[4]
            question_limit = int(sys.argv[5])

            # Perform training
            nn = crossvalidate(
                learning_rate, hidden_size, activation_func, question_limit
            )

        elif mode == "test":
            if len(sys.argv) != 3:
                print("Usage: python main.py test <question_limit>")
                sys.exit(1)

            question_limit = int(sys.argv[2])

            nn = DualModeNeuralNetwork(
                num_questions=input_size,
                num_hidden1=128,
                num_hidden2=128,
                num_targets=output_size,
                learning_rate=0.1,
                activation_func="relu",
            )
            nn.load(f"saved/zoo_{question_limit}.json")

            # Testing Function
            def test(nn):
                game = Q20Game(nn, qs, ts, question_limit)
                while True:
                    game.play()
                    nn.save(f"saved/zoo_{question_limit}.json")

            test(nn)

        else:
            print("Invalid mode. Please specify 'train' or 'test'.")
    else:
        print("No mode specified. Please specify 'train' or 'test'.")
