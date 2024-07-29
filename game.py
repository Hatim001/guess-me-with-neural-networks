import random
import numpy as np
from copy import copy

from zoo import Zoo

SRC = "data/zoo.csv"
Z = Zoo(SRC)


class Q20Game:
    """
    Represents a 20 Questions game.

    Attributes:
    - nn (NeuralNetwork): The neural network used for the game.
    - qs (list): The list of questions.
    - ts (list): The list of possible targets.
    - q_limit (int): The maximum number of questions to ask.

    Methods:
    - __init__(self, nn, qs, ts, q_limit): Initializes a new instance of the Q20Game class.
    - autoplay(self, target): Plays the game automatically by asking questions and updating the neural network.
    - autoplay_smartq(self, target): Plays the game automatically using a smart strategy for selecting questions.
    - smart_q(self, output, input_vector, qs_asked): Selects the best question to ask based on the current output and input vector.
    - play(self): Plays the game interactively with the user.
    """

    def __init__(self, nn, qs, ts, q_limit):
        """
        Initializes a new instance of the Q20Game class.

        Parameters:
        - nn (NeuralNetwork): The neural network used for the game.
        - qs (list): The list of questions.
        - ts (list): The list of possible targets.
        - q_limit (int): The maximum number of questions to ask.
        """
        self.qs = qs
        self.ts = ts
        self.q_limit = q_limit

        self.num_qs = len(qs)
        self.num_ts = len(ts)

        self.nn = nn

    def autoplay(self, target):
        """
        Plays the game automatically by asking questions and updating the neural network.

        Parameters:
        - target (int): The index of the target in the list of possible targets.

        Returns:
        - input_vector (list): The input vector used for the last question asked.
        """
        input_vector = [0] * self.num_qs
        qs_asked = []

        target_vector = [0] * self.num_ts
        target_vector[target] = 1

        for _ in range(self.q_limit):
            qi = random.randint(0, self.num_qs - 1)
            while qi in qs_asked:
                qi = random.randint(0, self.num_qs - 1)

            input_vector[qi] = Z.get_answer_i(qi, target)
            qs_asked.append(qi)

        self.nn.backpropagate(input_vector, target_vector)
        return input_vector

    def autoplay_smartq(self, target):
        """
        Plays the game automatically using a smart strategy for selecting questions.

        Parameters:
        - target (int): The index of the target in the list of possible targets.

        Returns:
        - result (tuple): A tuple containing the result of the game (0 for loss, 1 for win) and the input vector used for the last question asked.
        """
        output = [0] * self.num_ts
        input_vector = [0] * self.num_qs

        target_vector = [0] * self.num_ts
        target_vector[target] = 1

        qi_order = []
        o_order = []

        qs_asked = []

        best_guess = None

        for i in range(self.q_limit):
            best_guess = np.argmax(output)

            best_qi = self.smart_q(output, input_vector, qs_asked)
            if best_qi == -1:
                break

            input_vector[best_qi] = Z.get_answer_i(best_qi, target)
            if best_qi not in qs_asked:
                qs_asked.append(best_qi)
            qi_order.append(best_qi)

            output = self.nn.feed_forward_questions_to_targets(input_vector)
            o_order.append([self.ts[best_guess], 100 * round(max(output), 2)])

        qs_str = []
        for i in range(len(qi_order)):
            ans = Z.get_answer_i(qi_order[i], target)
            ansstr = " y " if ans == 1 else " n "
            q_str = f"{i + 1}. {self.qs[qi_order[i]]}{ansstr}({o_order[i][0]} {o_order[i][1]}%)"
            qs_str.append(q_str)

        if best_guess == target:
            print(f'win: {self.ts[best_guess]}... {len(qs_str)} qs: {" ".join(qs_str)}')
            return 1, input_vector
        else:
            print(
                f'LOSS: {self.ts[best_guess]} ({self.ts[target]})... {len(qs_str)} qs: {" ".join(qs_str)}'
            )
            return 0, input_vector

    def smart_q(self, output, input_vector, qs_asked):
        """
        Selects the best question to ask based on the current output and input vector.

        Parameters:
        - output (list): The current output of the neural network.
        - input_vector (list): The current input vector.
        - qs_asked (list): The list of indices of questions already asked.

        Returns:
        - best_qi (int): The index of the best question to ask.
        """
        best_qi = -1
        best_diff = -1

        for j in range(self.num_qs):
            if j in qs_asked:
                continue

            test_vector_no = copy(input_vector)
            test_vector_no[j] = -1
            test_vector_yes = copy(input_vector)
            test_vector_yes[j] = 1

            test_output_yes = self.nn.feed_forward_questions_to_targets(test_vector_yes)
            test_output_no = self.nn.feed_forward_questions_to_targets(test_vector_no)

            test_diff_yes = sum(abs(x - y) for x, y in zip(test_output_yes, output))
            test_diff_no = sum(abs(x - y) for x, y in zip(test_output_no, output))

            test_diff = min(test_diff_yes, test_diff_no)

            if best_qi == -1 or test_diff >= best_diff:
                best_diff = test_diff
                best_qi = j

        return best_qi

    def play(self):
        """
        Plays the game interactively with the user.
        """
        print("\n================ YOUR TURN: SELECT ANIMAL ================")
        print(f'select an animal from the list:\n[{", ".join(self.ts)}]')

        input("press enter to start")

        print(f"====================== {self.q_limit} QUESTIONS ======================")

        output = [0] * self.num_ts
        input_vector = [0] * self.num_qs
        qs_asked = []

        for i in range(self.q_limit - 1):
            best_guess = np.argmax(output)

            best_qi = self.smart_q(output, input_vector, qs_asked)
            if best_qi == -1:
                break
            print(f"{i + 1}. {self.qs[best_qi]}")
            if best_qi not in qs_asked:
                qs_asked.append(best_qi)

            ans_str = input("Answer? (y/n) ").strip().lower()
            while ans_str not in ["y", "n"]:
                ans_str = input("Invalid. Answer? (y/n) ").strip().lower()

            ans = 1 if ans_str == "y" else -1

            input_vector[best_qi] = ans
            output = self.nn.feed_forward_questions_to_targets(input_vector)

            print(f"({self.ts[best_guess]} {100 * round(max(output), 2)}%)")

        best_guess = np.argmax(output)

        target_vector = [0] * self.num_ts

        print(f"{self.q_limit}. are you thinking of a {self.ts[best_guess]}?")
        ans_str = input("Answer? (y/n) ").strip().lower()
        while ans_str not in ["y", "n"]:
            ans_str = input("Invalid. Answer? (y/n) ").strip().lower()
        if ans_str == "y":
            print("\nBOT WINS.")
            target_vector[best_guess] = 1
        else:
            print("\nYOU WIN.\n")
            t_str = input("what animal were you thinking of?\nAnimal: ").strip().lower()
            while t_str not in self.ts:
                t_str = input("Not in list, select again. Animal: ").strip().lower()
            target_vector[self.ts.index(t_str)] = 1

        print("training network...")
        self.nn.backpropagate(input_vector, target_vector)
