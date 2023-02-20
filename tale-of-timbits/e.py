import json
import pennylane as qml
import pennylane.numpy as np
import scipy

U_NP = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]


def get_partial_trace(matrix, subsystem):
    device = qml.device("default.qubit", wires=2)

    # get the partial trace
    @qml.qnode(device)
    def get_timbit_density(matrix):
        qml.QubitUnitary(matrix, wires=[0, 1])
        return qml.density_matrix(wires=[subsystem])

    return get_timbit_density(matrix)


def calculate_timbit(U, rho_0, rho, n_iters):
    """
    This function will return a timbit associated to the operator U and a state passed as an attribute.

    Args:
        U (numpy.tensor): A 2-qubit gate in matrix form.
        rho_0 (numpy.tensor): The matrix of the input density matrix.
        rho (numpy.tensor): A guess at the fixed point C[rho] = rho.
        n_iters (int): The number of iterations of C.

    Returns:
        (numpy.tensor): The fixed point density matrices.
    """

    def get_matrix_timbit(U, rho_0, rho):
        U_dagger = np.transpose(np.conjugate(U))
        density_tensor = np.kron(rho_0, rho)
        return np.matmul(U_dagger, np.matmul(density_tensor, U))

    timbit = rho
    print(timbit)
    for i in range(n_iters):
        matrix = get_matrix_timbit(U, rho_0, timbit)
        timbit = get_partial_trace(matrix, 0)

    return timbit


def apply_timbit_gate(U, rho_0, timbit):
    """
    Function that returns the output density matrix after applying a timbit gate to a state.
    The density matrix is the one associated with the first qubit.

    Args:
        U (numpy.tensor): A 2-qubit gate in matrix form.
        rho_0 (numpy.tensor): The matrix of the input density matrix.
        timbit (numpy.tensor): The timbit associated with the operator and the state.

    Returns:
        (numpy.tensor): The output density matrices.
    """

    def get_matrix_timbit(U, rho_0, rho):
        U_dagger = np.transpose(np.conjugate(U))
        density_tensor = np.kron(rho_0, rho)
        return np.matmul(U_dagger, np.matmul(density_tensor, U))

    matrix = get_matrix_timbit(U, rho_0, timbit)

    return get_partial_trace(matrix, 1)
    # Put your code here #


def SAT(U_f, q, rho, n_bits):
    """A timbit-based algorithm used to guess if a Boolean function ever outputs 1.

    Args:
        U_f (numpy.tensor): A multi-qubit gate in matrix form.
        q (int): Number of times we apply the Timbit gate.
        rho (numpy.tensor): An initial guess at the fixed point C[rho] = rho.
        n_bits (int): The number of bits the Boolean function is defined on.

    Returns:
        numpy.tensor: The measurement probabilities on the last wire.
    """
    device = qml.device("default.qubit", wires=n_bits + 1)

    rho_0 = np.array([[1, 0], [0, 0]], dtype=np.complex64)

    timbit = calculate_timbit(U_NP, rho_0, rho, 10)
    timbit_gate = apply_timbit_gate(U_NP, rho_0, timbit)

    @qml.qnode(device)
    def get_measurement():
        for i in range(n_bits):
            qml.Hadamard(wires=i)

        qml.QubitUnitary(U_f, wires=list(range(n_bits)))

        for i in range(q):
            qml.QubitUnitary(timbit_gate, wires=n_bits)

        return qml.probs(wires=[n_bits])

    print(qml.draw(get_measurement)())
    # Put your code here #
    measurements = get_measurement().data

    return [measurements[0], measurements[1]]


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    I = np.eye(2)
    X = qml.matrix(qml.PauliX(0))

    U_f = scipy.linalg.block_diag(I, X, I, I, I, I, I, I)
    rho = [[0.6 + 0.0j, 0.1 - 0.1j], [0.1 + 0.1j, 0.4 + 0.0j]]

    q = json.loads(test_case_input)
    output = list(SAT(U_f, q, rho, 4))

    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)

    rho = [[0.6 + 0.0j, 0.1 - 0.1j], [0.1 + 0.1j, 0.4 + 0.0j]]
    rho_0 = [[0.6 + 0.0j, 0.1 - 0.1j], [0.1 + 0.1j, 0.4 + 0.0j]]

    assert np.allclose(
        solution_output, expected_output, atol=0.01
    ), "Your NP-solving timbit computer isn't quite right yet!"


test_cases = [["1", "[0.78125, 0.21875]"], ["2", "[0.65820312, 0.34179687]"]]

for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        print("My output : ", output)
        print("Correct : ", expected_output)
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")
