import pennylane as qml
import pennylane.numpy as np

def encode(i, j, k):
    """
    Quantum encoding function. It must act only on the first two qubits.
    This function does not return anything, it simply applies gates.

    Args:
        i, j, k (int): The three encoding bits. They will take the values 1 or 0.

    """

    # Put your code here #
    # U1
    U1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    # U2
    U2 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
    ])
    # U3
    U3 = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    # U4
    U4 = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
    ])
    # U5
    U5 = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])
    # U6
    U6 = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, -1],
        [0, 0, 1, 0],
    ])
    # U7
    U7 = np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0]
    ])
    # U8
    U8 = np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [-1, 0, 0, 0],
    ])

    U = [U1, U7, U5, U3, U2, U8, U6, U4]
    qml.QubitUnitary(U[int(str(i)+str(j)+str(k), 2)], wires=[0, 1])


def decode():
    """
    Quantum decoding function. It can act on the three qubits.
    This function does not return anything, it simply applies gates.
    """

    # Put your code here #

    qml.CNOT([0, 2])
    qml.CNOT([0, 1])
    qml.Hadamard(0)



dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def circuit(i, j, k):
    """
    Circuit that generates the complete communication protocol.

    Args:
        i, j, k (int): The three encoding bits. They will take the value 1 or 0.
    """

    # We prepare the state 1/sqrt(2)(|000> + |111>)
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])

    # Zenda encodes the bits
    encode(i, j, k)

    # Reece decode the information
    decode()

    return qml.probs(wires=range(3))

# These functions are responsible for testing the solution.



def run(test_case_input: str) -> str:

    return None

def check(solution_output: str, expected_output: str) -> None:

    for i in range(2):
        for j in range(2):
            for k in range(2):
                assert np.isclose(circuit(i, j , k)[4 * i + 2 * j + k],1)

                dev = qml.device("default.qubit", wires=3)

                @qml.qnode(dev)
                def circuit2(i, j, k):
                    encode(i, j, k)
                    return qml.probs(wires=range(3))

                circuit2(i, j, k)
                ops = circuit2.tape.operations

                for op in ops:
                    assert not (2 in op.wires), "Invalid connection between qubits."
