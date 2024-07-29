import streamlit as st
import numpy as np
from game import Q20Game
from zoo import Zoo
from neural import DualModeNeuralNetwork

model_mapper = {
    8: {
        "learning_rate": 0.1,
        "hidden_size": 128,
        "activation_func": "relu",
        "question_limit": 8,
        "model_filename": "saved/zoo_8.json",
    },
    10: {
        "learning_rate": 0.1,
        "hidden_size": 128,
        "activation_func": "leaky_relu",
        "question_limit": 10,
        "model_filename": "saved/zoo_10.json",
    },
    15: {
        "learning_rate": 0.1,
        "hidden_size": 128,
        "activation_func": "leaky_relu",
        "question_limit": 15,
        "model_filename": "saved/zoo_15.json",
    },
    20: {
        "learning_rate": 0.1,
        "hidden_size": 128,
        "activation_func": "leaky_relu",
        "question_limit": 20,
        "model_filename": "saved/zoo_20.json",
    },
}


# Function to load the data and initialize objects
def initialize_game(
    learning_rate, hidden_size, activation_func, question_limit, model_filename
):
    SRC = "data/zoo.csv"
    zoo_data = Zoo(SRC)
    questions = zoo_data.questions
    targets = zoo_data.targets
    input_size = len(questions)
    output_size = len(targets)

    neural_network = DualModeNeuralNetwork(
        num_questions=input_size,
        num_hidden1=hidden_size,
        num_hidden2=hidden_size,
        num_targets=output_size,
        learning_rate=learning_rate,
        activation_func=activation_func,
    )

    # Load pre-trained neural network model if available
    try:
        neural_network.load(model_filename)
    except FileNotFoundError:
        pass

    game_instance = Q20Game(neural_network, questions, targets, question_limit)
    return game_instance, neural_network, questions, targets, question_limit


# Streamlit App
def main():
    st.title("20 Questions Game")

    if "game_started" not in st.session_state:
        st.session_state.game_started = False

    if not st.session_state.game_started:
        show_intro()
    else:
        play_game()


def show_intro():
    st.write(
        """
        Welcome to the 20 Questions Game!
        Think of an animal from the list below, and the game will try to guess
        it by asking you yes/no questions based on the question limit you 
        have selected.
        """
    )

    if "targets" not in st.session_state:
        _, _, _, targets, _ = initialize_game(**model_mapper[10])
        st.session_state.targets = targets

    st.write("### Animals to choose from:")
    st.write(", ".join(st.session_state.targets))

    question_limit = st.selectbox(
        "Select the number of questions:",
        options=[
            "8 (Accuracy: 80.5%)",
            "10 (Accuracy: 94.5%)",
            "15 (Accuracy: 100%)",
            "20 (Accuracy: 100%)",
        ],
        index=1,  # Default to 10 questions
    )

    question_limit = int(question_limit.split()[0])

    print("Question Limit", question_limit, type(question_limit))

    if st.button("Start Game"):
        (
            st.session_state.game_instance,
            st.session_state.neural_network,
            st.session_state.questions,
            st.session_state.targets,
            st.session_state.question_limit,
        ) = initialize_game(**model_mapper[question_limit])
        st.session_state.question_index = 0
        st.session_state.asked_questions = []
        st.session_state.input_vector = [0] * len(st.session_state.questions)
        st.session_state.output = [0] * len(st.session_state.targets)
        st.session_state.current_question = None
        st.session_state.game_completed = False
        st.session_state.correct_animal_selected = False
        st.session_state.game_started = True
        st.rerun()


def play_game():
    """
    Play the 20 Questions Game.

    This function allows the user to play the 20 Questions Game.
    It uses the session state variables to keep track of the game
    progress and user inputs.

    Returns:
        None
    """
    game_instance = st.session_state.game_instance
    questions = st.session_state.questions
    targets = st.session_state.targets

    if not st.session_state.game_completed:
        if st.session_state.question_index < st.session_state.question_limit - 1:
            if st.session_state.current_question is None:
                best_question_index = game_instance.smart_q(
                    st.session_state.output,
                    st.session_state.input_vector,
                    st.session_state.asked_questions,
                )
                if (
                    best_question_index != -1
                    and best_question_index not in st.session_state.asked_questions
                ):
                    st.session_state.asked_questions.append(best_question_index)
                    st.session_state.current_question = (
                        best_question_index,
                        questions[best_question_index],
                    )

            if st.session_state.current_question:
                question_index, question = st.session_state.current_question
                st.write(f"Question {st.session_state.question_index + 1}: {question}")

                answer = st.radio("Answer:", ("Yes", "No"))

                if st.button("Submit Answer"):
                    st.session_state.input_vector[question_index] = (
                        1 if answer == "Yes" else -1
                    )
                    st.session_state.output = st.session_state.neural_network.feed_forward_questions_to_targets(
                        st.session_state.input_vector
                    )
                    st.session_state.question_index += 1
                    st.session_state.current_question = None
                    st.rerun()

                if (
                    st.session_state.question_index >= 4
                    and st.session_state.question_index
                    < st.session_state.question_limit - 1
                ):
                    best_guess_index = np.argmax(st.session_state.output)
                    st.write(
                        f"Most probable answer: {targets[best_guess_index]} (Probability: {int(round(st.session_state.output[best_guess_index], 2) * 100)}%)"
                    )

        else:
            best_guess_index = np.argmax(st.session_state.output)
            st.write(f"Is your animal a {targets[best_guess_index]}?")

            final_answer = st.radio("Final Answer:", ("Yes", "No"))

            if st.button("Finish Game"):
                if final_answer == "Yes":
                    st.write("The game wins!")
                    st.session_state.game_completed = True
                else:
                    st.write("You win!")
                    st.session_state.correct_animal_selected = True

    if st.session_state.correct_animal_selected:
        correct_animal = st.selectbox("What was your animal?", targets)
        if st.button("Submit Animal"):
            target_vector = [0] * len(targets)
            target_vector[targets.index(correct_animal)] = 1
            st.session_state.neural_network.backpropagate(
                st.session_state.input_vector, target_vector
            )
            st.session_state.neural_network.save(
                model_mapper[st.session_state.question_limit].get("model_filename")
            )
            st.session_state.correct_animal_selected = False
            st.session_state.game_completed = True
            st.rerun()

    if st.session_state.game_completed and not st.session_state.correct_animal_selected:
        st.write("Thank you for playing the 20 Questions Game!")
        if st.button("Play Again"):
            st.session_state.game_started = False
            st.rerun()


if __name__ == "__main__":
    main()
