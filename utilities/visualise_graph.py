'import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pygraphviz as pgv

base_path = r"/home/harryc/github/brainglobe_ccf_translator/brainglobe_ccf_translator"

def calculate_route(source, target, metadata):
    G = nx.DiGraph()
    for i, m in metadata.iterrows():
        s = m['source_space'] + '_P' + str(m['source_age_pnd'])
        t = m['target_space'] + '_P' + str(m['target_age_pnd'])
        G.add_edge(s, t)
    
    # Visualize the graph
    visualize_graph(G)

def visualize_graph(G):
    plt.figure(figsize=(14, 10))  # Increase figure size
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')  # Use the dot layout for the entire graph
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", font_size=12, font_weight="bold", arrows=True)
    plt.savefig("graph.png", format="png", transparent=False)  # Save as PNG with transparent background
    plt.show()

key_ages = [56, 28, 21, 14, 7, 4]

# Example usage
metadata_path = os.path.join(base_path, "metadata", "translation_metadata.csv")
metadata = pd.read_csv(metadata_path)
#get rid of interpolated nodes
metadata = metadata.loc[metadata[metadata['key_age']][['source_space', 'target_space', 'source_key_age', 'target_key_age']].drop_duplicates().index]
demba = metadata[metadata['source_space'].str.contains('demba')]
other = metadata[~metadata['source_space'].str.contains('demba')]
demba = demba[demba['source_age_pnd'].isin(key_ages)]
demba = demba[demba['target_age_pnd'].isin(key_ages)]
metadata = pd.concat([demba, other])
source = "source_node"
target = "target_node"
calculate_route(source, target, metadata)'