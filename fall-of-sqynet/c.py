import json
import pennylane as qml
import pennylane.numpy as np

def half_life(gamma, p):
    """Calculates the relaxation half-life of a quantum system that exchanges energy with its environment.
    This process is modeled via Generalized Amplitude Damping.

    Args:
        gamma (float): 
            The probability per unit time of the system losing a quantum of energy
            to the environment.
        p (float): The de-excitation probability due to environmental effect

    Returns:
        (float): The relaxation haf-life of the system, as explained in the problem statement.
    """

    num_wires = 1

    dev = qml.device("default.mixed", wires=num_wires)

    DEL_T = 1

    @qml.qnode(dev)
    def noise(
        gamma,
        total_time
    ):
        """Implement the sequence of Generalized Amplitude Damping channels in this QNode
        You may pass instead of return if you solved this problem analytically, it's possible!

        Args:
            gamma (float): The probability per unit time of the system losing a quantum of energy
            to the environment.
            total_time (float): The allowed relaxation time.
        
        Returns:
            (float): The relaxation half-life.
        """

        qml.Hadamard(wires=0)

        time = 0
        while time < total_time:
            time += DEL_T
            qml.GeneralizedAmplitudeDamping(gamma * DEL_T, p, wires=0)

        return qml.probs(wires=0)

    t = 0

    res = [0.5, 0.5]

    while res[0] < 0.75:
        t += DEL_T
        res = noise(gamma, t)

    return t

# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:

    ins = json.loads(test_case_input)
    output = half_life(*ins)

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, atol=2e-1
    ), "The relaxation half-life is not quite right."

test_cases = [['[0.1,0.92]', '9.05'], ['[0.2,0.83]', '7.09']]

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