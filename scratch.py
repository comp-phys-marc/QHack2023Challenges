#! /usr/bin/python3

import sys
import math
import random
import pennylane as qml
from pennylane import numpy as np


dev = qml.device("default.qubit", wires=3)


def prepare_state(alpha, beta):
    """Construct a circuit that prepares the (not necessarily maximally) entangled state in terms of alpha and beta
    Do not forget to normalize.

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    """

    # normalize

    # normalization_constant = 1 / math.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    # alpha = alpha * normalization_constant
    # beta = beta * normalization_constant

    # rotate

    # theta = 2 * math.acos(alpha)
    # qml.RY(theta, wires=[0])

@qml.qnode(dev)
def circuit(alpha, z):

    def ReLu(angle):
        if angle < 0:
            return 0
        else:
            return angle
    
    def bit_1_2_envelope(x):
        return min((1 / (1 + np.exp(-x)) - 0.5) * 4.4, 1)
    
    def bit_0_envelope(x):
        return 1 / (1 + np.exp(-x + 2)) + (0.2 / 4) * x - z[int(alpha)]
    
    # prepare_state(alpha, beta)

    qml.RX(bit_1_2_envelope(ReLu(alpha) % 2) * np.pi, wires=[2])

    # odd numbers
    if alpha in [5, 7]:
        qml.RX(bit_1_2_envelope(ReLu(alpha - 1) % 4) * np.pi, wires=[1])
        qml.RX(bit_0_envelope(ReLu(alpha)) * np.pi, wires=[0])
    
    # center point
    elif alpha == 3:
        qml.RX(bit_1_2_envelope(ReLu(alpha - 1) % 3) * np.pi, wires=[1])
        qml.RX(bit_0_envelope(ReLu(alpha)) * np.pi, wires=[0])
    
    #even numbers
    else:
        qml.RX(bit_1_2_envelope(ReLu(alpha - 1) % 3) * np.pi, wires=[1])
        qml.RX(bit_0_envelope(ReLu(alpha)) * np.pi, wires=[0])
        
    return qml.probs(wires=range(3))
    

def winning_prob(alpha, z):
    """Define a function that returns the probability of winning the game.

    Returns:
        - (float): Probability of winning the game
    """

    # find a and b based on circuit and return prob(x * y == a + b mod 2)

    output = circuit(alpha, z)
    prob_wins = np.isclose(output[int(alpha)], 1)

    return prob_wins

def optimize(alpha):
    """Define a function that optimizes z to maximize the probability of winning the game

    Returns:
        - (float): Probability of winning
    """

    def cost(z):
        """Define a cost function that only depends on params, given alpha and beta fixed"""

        return 1 - winning_prob(alpha, z)

    #Initialize parameters, choose an optimization method and number of steps
    init_params = np.array([0.119, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)
    steps = 100

    # set the initial parameter values
    z = init_params

    for i in range(steps):
        # update the circuit parameters

        z = opt.step(cost, z)

    print(z)
    print(winning_prob(z))


if __name__ == '__main__':
    output = optimize(float(0))
    print(f"{output}")
