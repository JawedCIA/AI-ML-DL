# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:13:37 2024

@author: jawed
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import networkx as nx

def draw_simple_neural_network(layers):
    """
    Draws a simplified and clear diagram of a neural network showing neuron connections.
    Parameters:
    layers (list of int): Number of neurons in each layer.
    """
    G = nx.DiGraph()
    pos = {}
    count = 0

    # Create nodes and position them layer by layer
    for i, num_neurons in enumerate(layers):
        for j in range(num_neurons):
            G.add_node(count)
            pos[count] = (i, -j)  # Arrange nodes vertically per layer
            count += 1

    # Create representative edges between layers
    previous_layer_neurons = 0
    for i in range(len(layers) - 1):
        for j in range(min(10, layers[i])):  # Show up to 10 neurons per layer
            for k in range(min(10, layers[i + 1])):
                G.add_edge(previous_layer_neurons + j, previous_layer_neurons + layers[i] + k)
        previous_layer_neurons += layers[i]

    # Plot the graph
    plt.figure(figsize=(10, 6))
    nx.draw(
        G, pos, with_labels=False, node_size=300, 
        node_color="lightblue", edge_color="gray", alpha=0.8, connectionstyle="arc3,rad=0.1"
    )
    
    # Annotate layers
    for i, num_neurons in enumerate(layers):
        plt.text(i, 1, f"Layer {i + 1}\n({num_neurons} neurons)", fontsize=10, ha="center", color="black")

    plt.title("Simplified Neural Network Architecture", fontsize=16)
    plt.axis("off")
    plt.show()


# Define the neural network layers
layer_sizes = [11, 128, 64, 32, 1]
draw_simple_neural_network(layer_sizes)

