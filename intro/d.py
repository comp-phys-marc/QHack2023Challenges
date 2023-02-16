import json
import pennylane as qml
import pennylane.numpy as np

def is_product(state, subsystem, wires):
    """Determines if a pure quantum state can be written as a product state between 
    a subsystem of wires and their compliment.

    Args:
        state (numpy.array): The quantum state of interest.
        subsystem (list(int)): The subsystem used to determine if the state is a product state.
        wires (list(int)): The wire/qubit labels for the state. Use these for creating a QNode if you wish!

    Returns:
        (str): "yes" if the state is a product state or "no" if it isn't.
    """

    print("STATE:")
    print(state)
    print("SUBSYSTEMS:")
    print(subsystem)
    print("WIRES:")
    print(wires)
    
    def is_diagonal(state):
        diag_total = 0
        for i in range(len(state[0])):
            diag_total += abs(state[i][i]) ** 2
            
        n = 1 / np.sqrt(diag_total)
        
        diag_total = 0
        for i in range(len(state[0])):
            diag_total += state[i][i] * n
    
        if not diag_total == 1:
            return False
        
        for i in range(len(state[0])):
            for j in range(len(state[0])):
                if i != j:
                    if not state[i][j] == 0:
                        return False
                    if not state[j][i] == 0:
                        return False
        return True
    
    # make sure the state is pure
    dagger = np.transpose(np.atleast_2d(state)).conj()
    print("COMPLEX CONJUGAT TRANSPOSE:")
    print(dagger)
    
    density_matrix = np.matmul(dagger, np.atleast_2d(state))
    print("DENSITY MATRIX:")
    print(density_matrix)
    
    if not is_diagonal(density_matrix):
        return "no"
    
    # attempt Schmidt decomposition
    try:
        U, coeffs, V_dagger = np.linalg.svd(density_matrix)
    except Exception as e:
        print(f"svd impossible: {e}")
        return "no"
    
    print("DECOMPOSITION:")
    print(U)
    print(coeffs)
    print(V_dagger)
    
    return "yes"

# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    state, subsystem, wires = ins
    state = np.array(state)
    output = is_product(state, subsystem, wires)
    return output

def check(solution_output: str, expected_output: str) -> None:
    assert solution_output == expected_output

test_cases = [['[[0.707107, 0, 0, 0.707107], [0], [0, 1]]', 'no'], ['[[1, 0, 0, 0], [0], [0, 1]]', 'yes']]

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