import json
import pennylane as qml
import pennylane.numpy as np

def hamiltonian(num_wires):
    """A function for creating the Hamiltonian in question for a general
    number of qubits.

    Args:
        num_wires (int): The number of qubits.

    Returns:
        (qml.Hamiltonian): A PennyLane Hamiltonian.
    """
    wires = []
    for w in range(num_wires):
        wires.append(w)

    coeffs = []
    observables = []

    for j in wires:
        for i in wires:
            if i < j:
                coeffs.append(1 / 3)
                observables.append(qml.prod(qml.PauliX(wires=[i]), qml.PauliX(wires=[j])))
            else:
                break

    for k in wires:
        coeffs.append(-1)
        observables.append(qml.PauliZ(wires=[k]))

    # hamiltonian = first_sum - second_sum

    return qml.Hamiltonian(coeffs, observables)

def expectation_value(num_wires):
    """Simulates the circuit in question and returns the expectation value of the 
    Hamiltonian in question.

    Args:
        num_wires (int): The number of qubits.

    Returns:
        (float): The expectation value of the Hamiltonian.
    """
    # Define a device using qml.device
    dev = qml.device("default.qubit", wires=num_wires)

    @qml.qnode(dev)
    def circuit(num_wires):
        """A quantum circuit with Hadamard gates on every qubit and that measures
        the expectation value of the Hamiltonian in question. 
        """

        for w in range(num_wires):
            qml.Hadamard(wires=[w])

        return qml.expval(hamiltonian(num_wires))

    return circuit(num_wires)

# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    num_wires = json.loads(test_case_input)
    output = expectation_value(num_wires)

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(solution_output, expected_output, rtol=1e-4)

test_cases = [['8', '9.33333']]

for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")
