#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 09:52:33 2021

@author: erikhormann
"""

import igraph
import pickle
import numpy as np
import networkx as nx
import walker
import re
import warnings
import random
import matplotlib.pyplot as plt
from itertools import chain
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components

def simulate_random_walks(graph, ns, nw, num_walks):
    # Initialize a matrix to store counts of walks from sources to wells
    counts_matrix = np.zeros((len(ns), len(flatten(nw))))
    wellsNodes = flatten(nw)
    N = graph.number_of_nodes()

    # Generate random walks
    walks = walker.random_walks(graph, n_walks=num_walks, walk_len=10*N)
    j=0
    k=len(ns)
    # Perform random walks
    for source_index, source_nodes in enumerate(ns):
        j=j+1
        print(str(j) + " of " + str(k))
        for walk in walks:
            source_node = walk[0]
            if source_node in source_nodes:
                current_node = source_node
                for node in walk[1:]:
                    # Check if the current node is a well
                    if any(current_node in sublist for sublist in nw):
                        well_index = wellsNodes.index(current_node)
                        counts_matrix[source_index][well_index] += 1
                        break
                    current_node = node

    # Normalize counts to get probabilities
    probabilities_matrix = normalize_matrix_by_row(counts_matrix)
    return probabilities_matrix


def normalize_matrix_by_row(matrix):
    # Calculate the sum of each row
    row_sums = np.sum(matrix, axis=1)

    # Create a mask to identify rows with non-zero sums
    non_zero_mask = row_sums != 0

    # Initialize the normalized matrix with zeros
    normalized_matrix = np.zeros_like(matrix, dtype=float)

    # Normalize rows with non-zero sums
    normalized_matrix[non_zero_mask] = matrix[non_zero_mask] / row_sums[non_zero_mask][:, np.newaxis]

    return normalized_matrix


def FindIO(G):
    # Keep the matrix sparse! Do NOT use .toarray()
    A = nx.adjacency_matrix(G)

    wells = []
    sources = []
    gcc = []
    unconnected = []

    # SCCs are fine, they are memory efficient
    clusters = list(nx.strongly_connected_components(G))

    for cl in clusters:
        clist = list(cl)

        # Instead of slicing a massive array, we look at the degrees
        # of the nodes within the SCC relative to the whole graph.

        # Out-degree: Number of edges leaving these nodes
        # In-degree: Number of edges entering these nodes
        out_degree_total = sum(dict(G.out_degree(clist)).values())
        in_degree_total = sum(dict(G.in_degree(clist)).values())

        # Edges internal to the SCC
        # (An SCC with n nodes has at least n edges if n > 1)
        # We need to know if there are edges going OUTSIDE the cluster

        has_out = any(any(neighbor not in cl for neighbor in G.neighbors(n)) for n in clist)
        has_in = any(any(predecessor not in cl for predecessor in G.predecessors(n)) for n in clist)

        if has_out and not has_in:
            sources.append(clist)
        elif not has_out and has_in:
            wells.append(clist)
        elif has_out and has_in:
            gcc.append(clist)
        else:  # not has_out and not has_in
            unconnected.append(clist)

    gcc = [el for subl in gcc for el in subl]
    return sources, wells, gcc, unconnected


def flatten(mixed_list):
    # If the input isn't even a list, just return it as is
    if not isinstance(mixed_list, list):
        return mixed_list

    flat_list = []
    for item in mixed_list:
        # Only extend if the item is a list; otherwise append
        if isinstance(item, list):
            flat_list.extend(item)
        else:
            flat_list.append(item)
    return flat_list

# Legacy implementation retained for validation only.
# Use FindIO() in all production code.
def FindIO_lenght(G):
    warnings.warn(
        "FindIO_lenght() is a legacy implementation retained for "
        "validation purposes only. Use FindIO() for all production "
        "analysis.",
        DeprecationWarning,
        stacklevel=2,
    )
    A = nx.adjacency_matrix(G)
    A = A.toarray()

    wells = []
    sources = []
    gcc = []
    unconnected = []

    clusters = nx.strongly_connected_components(G)
    clusters = list(clusters)

    all_indices = list(range(G.number_of_nodes()))

    for cl in clusters:
        clist = list(cl)
        clist = list(map(int, clist))
        # test for output
        others = [item for item in all_indices if item not in clist]
        outgoing_links = A[np.ix_(clist, others)]
        incoming_links = A[np.ix_(others, clist)]
        if (outgoing_links != 0).any() and (incoming_links == 0).all():
            sources.append(clist)
        elif (outgoing_links == 0).all() and (incoming_links != 0).any():
            wells.append(clist)
        elif (outgoing_links != 0).any() and (incoming_links != 0).any():
            gcc.append(clist)
        elif (outgoing_links == 0).all() and (incoming_links == 0).all():
            unconnected.append(clist)
        else:
            raise ValueError("Error with the splitting of the graph in Strongly Connected Components")


    sources = [el for subl in sources for el in subl]
    wells = [el for subl in wells for el in subl]
    gcc = [el for subl in gcc for el in subl]
    unconnected = [el for subl in unconnected for el in subl]

    
    return len(sources),len(wells), len(gcc), len(unconnected)

def import_graph(filename):

    if filename.endswith('.xml'):
        G = igraph.Graph.Read_GraphML(filename)
        return G

    elif filename.endswith('.gml'):
        G = nx.DiGraph()
        G = nx.read_gml(filename)
        #G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
    
    elif filename.endswith('.csv'):
        file = open(filename, "r")
        Graphtype = nx.DiGraph()

        G = nx.parse_edgelist(file, comments='t', delimiter=',', create_using=nx.DiGraph,)
        G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
        return G


    elif filename.endswith('.el') or filename.endswith('.edges'):
        G = nx.DiGraph()

        with open(filename, 'r') as f:
            for line in f:
                # 1. Clean line
                line = line.strip()
                if not line or line.startswith(('#', '%')):
                    continue

                # 2. Split using regex for robustness
                # This handles spaces, tabs, commas, and semicolons
                parts = re.split(r'[,\s;]+', line)

                # 3. Add edge directly to G (skipping the 'edges' list)
                if len(parts) >= 2:
                    G.add_edge(parts[0], parts[1])

        # 4. Convert in-place if possible (Relabeling is safer than 'convert_node_labels')
        mapping = {node: i for i, node in enumerate(G.nodes())}
        nx.relabel_nodes(G, mapping, copy=False)

        return G
    
    elif filename.endswith(".mat"):
        G = nx.DiGraph()
        A = np.loadtxt(filename)
        G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
        G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
        return G
    
    elif filename.endswith(".adj"):
        G = nx.DiGraph()
        G = nx.read_adjlist(filename, create_using=nx.DiGraph())
        G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
        return G

    else:
        print("error: filename " + filename + " has invalid extension")

def rewire_core(G, core, wells, sources, uc):
    core_nodes = flatten(core)
    wells_nodes = flatten(wells)
    sources_nodes = flatten(sources)
    uc_nodes = flatten(uc)

    # Extract edges within each part and between parts
    core_edges = [(u, v) for u, v in G.edges() if u in core_nodes and v in core_nodes]
    other_edges = [(u, v) for u, v in G.edges() if not (u in core_nodes and v in core_nodes)]


    # Generate a configuration model graph for the core
    core_subgraph = G.subgraph(core_nodes).copy()  # Use copy to ensure it's an induced subgraph
    in_degrees = [core_subgraph.in_degree(n) for n in core_nodes]
    out_degrees = [core_subgraph.out_degree(n) for n in core_nodes]
    rewired_core_graph = nx.directed_configuration_model(in_degrees, out_degrees)
    rewired_core_graph = nx.DiGraph(rewired_core_graph) # Convert to simple graph (remove parallel edges and self-loops)

    # Relabel nodes to preserve original ordering
    mapping = {i: node for i, node in enumerate(core_nodes)}
    rewired_core_graph = nx.relabel_nodes(rewired_core_graph, mapping)

    # Create the final graph
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes)
    H.add_edges_from(other_edges)
    H.add_edges_from(rewired_core_graph.edges())

    # Verify that non-core nodes retain their original connections
    for u, v in other_edges:
        if not H.has_edge(u, v):
            print(f"Missing edge: ({u}, {v})")

    return H

