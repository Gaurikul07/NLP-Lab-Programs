!pip install -q numpy
import numpy as np
states = ["START", "GREETING", "QUESTION", "GOODBYE"]
transitions = {
    "START": ["GREETING"],
    "GREETING": ["QUESTION", "GOODBYE"],
    "QUESTION": ["GREETING", "GOODBYE"],
    "GOODBYE": []
}
transition_probabilities = {
    "START": [1.0],
    "GREETING": [0.7, 0.3],
    "QUESTION": [0.6, 0.4],
    "GOODBYE": []
}
def predict_next_state(current_state):

    possible_states = transitions[current_state]

    if not possible_states:
        return None

    probabilities = transition_probabilities[current_state]

    next_state = np.random.choice(
        possible_states,
        p=probabilities
    )

    return next_state
current_state = "START"

print("Generated Sequence:")

while current_state is not None:

    print(current_state)

    current_state = predict_next_state(current_state)

print("Sequence completed!")
