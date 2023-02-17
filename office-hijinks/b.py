import json
import pennylane as qml
import pennylane.numpy as np

dev = qml.device("default.qubit", wires=3)

def normalize_probs(probs):
    res = []
    totals = []

    for alph in range(8):
        total = 0
        for val in range(8):
            total += probs[alph][val]
        totals.append(total)

    for alph in range(8):
        res.append([])
        for val in range(8):
            res[alph].append(probs[alph][val] / totals[alph])

    return res

@qml.qnode(dev)
def model(alpha):
    """In this qnode you will define your model in such a way that there is a single 
    parameter alpha which returns each of the basic states.

    Args:
        alpha (float): The only parameter of the model.

    Returns:
        (numpy.tensor): The probability vector of the resulting quantum state.
    """
    def ReLu(angle, cutoff=0):
        if angle < cutoff:
            return cutoff
        else:
            return angle
    
    def envelope(x):
        return min((1 / (1 + np.exp(-x)) - 0.5) * 4.4, 1)
    
    # def bit_0_envelope(x):
    #     return 1 / (1 + np.exp(-x + 2))

    if isinstance(alpha, np.tensor):
        alpha = ReLu(alpha, round(alpha.item()))
    else:
        alpha = ReLu(alpha, round(alpha))

    qml.RX(envelope(ReLu(alpha) % 2) * np.pi, wires=[2])

    # odd numbers
    if alpha in [5, 7]:
        
        qml.RX(envelope(ReLu(alpha - 1) % 4) * np.pi, wires=[1])
        qml.RX(envelope(ReLu(alpha - 2) % 4) * np.pi, wires=[0]) # qml.RX(bit_0_envelope(alpha) * np.pi, wires=[0])
    
    # center point
    elif alpha == 3:
        qml.RX(envelope(ReLu(alpha - 1) % 3) * np.pi, wires=[1])
        qml.RX(0, wires=[0]) # qml.RX(bit_0_envelope(alpha) * np.pi, wires=[0])
    
    #even numbers
    else:
        qml.RX(envelope(ReLu(alpha - 1) % 3) * np.pi, wires=[1])
        qml.RX(envelope(ReLu(alpha - 2) % 3) * np.pi, wires=[0]) # qml.RX(bit_0_envelope(alpha) * np.pi, wires=[0])
        
    return qml.probs(wires=range(3))

def generate_coefficients():
    """This function must return a list of 8 different values of the parameter that
    generate the states 000, 001, 010, ..., 111, respectively, with your ansatz.

    Returns:
        (list(int)): A list of eight real numbers.
    """
    return [0, 1, 2, 3, 4, 5, 6, 7]


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    return None

def check(solution_output, expected_output: str) -> None:
    coefs = generate_coefficients()
    output = np.array([model(c) for c in coefs])
    epsilon = 0.001

    for i in range(len(coefs)):
        print("OUTPUT:")
        print(output)
        print("COEFF:")
        print(coefs[i])
        print("PROBABILITY CORRECT:")
        print(output[i][i])
        assert np.isclose(output[i][i], 1)

    def is_continuous(function, point):
        limit = calculate_limit(function, point)

        if limit is not None and sum(abs(limit - function(point))) < epsilon:
            return True
        else:
            print(f"NOT CONTINUOUS @ {point} : {sum(abs(limit - function(point)))} < {epsilon}")
            print(limit)
            print(function(point))
            return False

    def is_continuous_in_interval(function, interval):
        for point in interval:
            if not is_continuous(function, point):
                return False
        return True

    def calculate_limit(function, point):
        x_values = [point - epsilon, point, point + epsilon]
        y_values = [function(x) for x in x_values]
        average = sum(y_values) / len(y_values)

        return average

    assert is_continuous_in_interval(model, np.arange(0,10,0.001))

    for coef in coefs:
        assert coef >= 0 and coef <= 10

test_cases = [['No input', 'No output']]

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
