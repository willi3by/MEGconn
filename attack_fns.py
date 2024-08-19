import cupy as cp
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, eye
from scipy.sparse.csgraph import connected_components

def get_components(adj):
    if adj.shape[0] != adj.shape[1]:
        raise ValueError('This adjacency matrix is not square')

    # Ensure the adjacency matrix is symmetric
    adj = np.logical_or(adj, adj.T).astype(int)

    # Add self-loops to ensure all nodes are connected
    if np.sum(np.diag(adj)) != adj.shape[0]:
        adj = np.logical_or(adj, np.eye(adj.shape[0])).astype(int)

    # Convert the adjacency matrix to a sparse format
    adj_sparse = csr_matrix(adj)

    # Find connected components
    n_components, labels = connected_components(csgraph=adj_sparse, directed=False, return_labels=True)

    # Calculate the size of each component
    comp_sizes = np.bincount(labels)

    return labels, comp_sizes


def calc_node_participation(second_comp_arr, attack_arr):
    num_attacks, num_nodes = second_comp_arr.shape
    node_part_arr = np.zeros(num_nodes)
    for i in range(num_attacks):
        max_value = np.max(second_comp_arr[i,:])
        max_indices = np.where(second_comp_arr[i,:] == max_value)[0]
        last_max_index = max_indices[-1]
        nodes_involved = attack_arr[i,:last_max_index]
        node_part_arr[nodes_involved] += 1
    return node_part_arr/num_attacks

def simulate_random_attacks(adj, number_attacks, z_thresh):
    num_nodes = adj.shape[0]

    # Initialize attack sequence array
    attack_arr = np.array([np.random.permutation(np.arange(num_nodes)) for _ in range(number_attacks)])    

    adj[np.isnan(adj)] = 0
    adj_thresholded = (adj > z_thresh).astype(int) * adj

    # Initialize arrays to store the sizes of the largest and second largest components
    first_comp = np.zeros((number_attacks, num_nodes))
    second_comp = np.zeros((number_attacks, num_nodes))

    for i in range(number_attacks):
        for j in range(num_nodes):
            adj_for_attack = np.copy(adj_thresholded)
            nodes_to_remove = attack_arr[i, :j]
            
            # Remove rows and columns corresponding to the nodes in the attack sequence
            adj_for_attack = np.delete(adj_for_attack, nodes_to_remove, axis=0)
            adj_for_attack = np.delete(adj_for_attack, nodes_to_remove, axis=1)
            
            _, comp_sizes = get_components(adj_for_attack)
            
            first_comp[i, j] = max(comp_sizes) if len(comp_sizes) > 0 else 0
            sorted_sizes = np.sort(comp_sizes)
            if len(sorted_sizes) > 1:
                second_comp[i, j] = sorted_sizes[-2]
            else:
                second_comp[i, j] = 0
    
    node_participation = calc_node_participation(second_comp, attack_arr)
    return first_comp, second_comp, node_participation

    