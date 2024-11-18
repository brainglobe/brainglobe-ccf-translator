import json
import networkx as nx
import numpy as np
from networkx.algorithms.approximation import steiner_tree
from networkx.algorithms import approximation as approx


def calculate_route(source, target, G):
    path = nx.shortest_path(G, source, target)
    return path


def create_G(metadata):
    G = nx.Graph()
    for i, m in metadata.iterrows():
        s = m["source_space"] + "_P" + str(m["source_age_pnd"])
        t = m["target_space"] + "_P" + str(m["target_age_pnd"])
        G.add_edge(s, t)
    return G

def find_path_through_nodes(G, nodes):
    # Create a subgraph containing only the specified nodes
    subgraph = steiner_tree(G, nodes)
    # Find the shortest route through the nodes using TSP solver
    tsp_path = approx.traveling_salesman_problem(subgraph, cycle=False)
    # Convert the TSP path to the full path in the original graph
    full_path = []
    for i in range(len(tsp_path) - 1):
        segment = nx.shortest_path(G, source=tsp_path[i], target=tsp_path[i+1], weight='weight')
        if i > 0:
            segment = segment[1:]  # Avoid duplicating nodes
        full_path.extend(segment)
    
    return full_path

    
def find_hamiltonian_path(G):
    def backtrack(path):
        if len(path) == len(G):
            return path
        for neighbor in G.neighbors(path[-1]):
            if neighbor not in path:
                path.append(neighbor)
                result = backtrack(path)
                if result:
                    return result
                path.pop()
        return None
    try:
        path = nx.algorithms.tournament.hamiltonian_path(G)
        return path
    except nx.NetworkXNoPath:
        return None