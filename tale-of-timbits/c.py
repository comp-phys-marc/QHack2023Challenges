import json
import pennylane as qml
import pennylane.numpy as np
import scipy


def abs_dist(rho, sigma):
    """A function to compute the absolute value |rho - sigma|."""
    polar = scipy.linalg.polar(rho - sigma)
    return polar[1]


def word_dist(word):
    """A function which counts the non-identity operators in a Pauli word"""
    return sum(word[i] != "I" for i in range(len(word)))


# Produce the Pauli density for a given Pauli word and apply noise


def noisy_Pauli_density(word, lmbda):
    """
       A subcircuit which prepares a density matrix (I + P)/2**n for a given Pauli
       word P, and applies depolarizing noise to each qubit. Nothing is returned.

    Args:
            word (str): A Pauli word represented as a string with characters I, X, Y and Z.
            lmbda (float): The probability of replacing a qubit with something random.
    """

    # Put your code here #
    # Put your code here #

    for i, gate in enumerate(word):
        if gate == "X":
            qml.PauliX(wires=i)
        elif gate == "Y":
            qml.PauliY(wires=i)
        elif gate == "Z":
            qml.PauliZ(wires=i)
        else:
            qml.Identity(wires=i)

    for i in range(len(word)):
        qml.DepolarizingChannel(p=lmbda, wires=i)


# Compute the trace distance from a noisy Pauli density to the maximally mixed density


def maxmix_trace_dist(word, lmbda):
    """
       A function compute the trace distance between a noisy density matrix, specified
       by a Pauli word, and the maximally mixed matrix.

    Args:
            word (str): A Pauli word represented as a string with characters I, X, Y and Z.
            lmbda (float): The probability of replacing a qubit with something random.

    Returns:
            float: The trace distance between two matrices encoding Pauli words.
    """

    # Put your code here #
    N = len(word)
    word_matrix = noisy_Pauli_density(word, lmbda)
    identity_matrix = (1 / 2**N) * np.eye(2**N)

    return 0.5 * np.trace(abs_dist(word_matrix, identity_matrix))


def bound_verifier(word, lmbda):
    """
       A simple check function which verifies the trace distance from a noisy Pauli density
       to the maximally mixed matrix is bounded by (1 - lambda)^|P|.

    Args:
            word (str): A Pauli word represented as a string with characters I, X, Y and Z.
            lmbda (float): The probability of replacing a qubit with something random.

    Returns:
            float: The difference between (1 - lambda)^|P| and T(rho_P(lambda), rho_0).
    """

    # Put your code here #
    P = word_dist(word)

    lmbda_term = (1 - lmbda) ** P
    trace_term = maxmix_trace_dist(word, lmbda)

    return lmbda_term - trace_term


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    word, lmbda = json.loads(test_case_input)
    output = np.real(bound_verifier(word, lmbda))

    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-4
    ), "Your trace distance isn't quite right!"


test_cases = [
    ['["XXI", 0.7]', "0.0877777777777777"],
    ['["XXIZ", 0.1]', "0.4035185185185055"],
    ['["YIZ", 0.3]', "0.30999999999999284"],
    ['["ZZZZZZZXXX", 0.1]', "0.22914458207245006"],
]

for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    # try:
    output = run(input_)
    print(output)
    # except Exception as exc:
    #     print(f"Runtime Error. {exc}")

    # else:
    #     if message := check(output, expected_output):
    #         print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

    #     else:
    #         print("Correct!")
