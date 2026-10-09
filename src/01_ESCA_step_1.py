#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESCA Step 1: Dataset Loading and Analysis
=============================
# Pipeline position:
#
# Script 1: Detect ergodic sets and transient core      <-- this file
# Script 2: Construct source-sink transport matrix B
# Script 3: Compute SVD compression of B
# Script 4: Generate figures and statistics


This script performs the first stage of the ESCA analysis on a collection
of directed networks.

For each network:

    1. Load the graph.
    2. Identify:
       - source ergodic sets (backward ergodic sets),
       - sink ergodic sets (forward ergodic sets),
       - the transient core,
       - disconnected/unclassified nodes.
    3. Compute the first-stage compression factor C1, corresponding
       to the reduction obtained by collapsing ergodic sets into
       single super-nodes.
    4. Store the decomposition and associated metadata for later
       analysis.

The output of this script is not the transport matrix B nor the final
ESCA compression. Instead, it generates the structural decomposition
required for subsequent stages of the pipeline.

Saved quantities include:

    graphs        : original directed networks
    sources       : source ergodic sets
    wells         : sink ergodic sets
    core          : transient core nodes
    unconnected   : disconnected nodes
    C1            : first-stage compression factor
    applications  : network application domain

These objects are reused by later scripts which compute source-sink
transport matrices, low-rank approximations, and the final ESCA
compression coefficient C2.
"""

import mygraph as myg
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
import pickle
import networkx as nx
import warnings
from random import randrange

with_plotting=True


#%%
models_folder = '../real_world_networks/Validated/'
allfiles = [f for f in os.listdir(models_folder) if os.path.isfile(os.path.join(models_folder, f))]
fileslist = list(set(allfiles) - set(['.DS_Store']))

#%% Creating applications dictionary
app_dict = {
    'W_': 'information',
    'T_': 'transportation',
    'E_': 'economic',
    'B_': 'biologic',
    }


# checking if sims exist
sims_path = "../results/"

filenames = [
    "N.dat", "Nsources.dat", "Nwells.dat", "applications_data.dat",
    "C1.dat", "G.dat", "wells.dat", "sources.dat", "core.dat", "unconnected.dat"
]

# Check if all files exist in the path
files_exist = all(os.path.exists(os.path.join(sims_path, f)) for f in filenames)

if files_exist:
    print("Files found! Loading data...")

#%% Reading network

if not files_exist:
    print("Files missing. Running simulations...")
    N = []
    Nwells = []
    Nsources = []
    applications = []
    C1 = []
    sources = []
    wells = []
    core = []
    unconnected = []
    graphs = []

    for file in fileslist:
        filepath = models_folder+file
        filename = os.path.basename(filepath)
        application_key = filename[:2]
        applications.append(app_dict[application_key])
        G = myg.import_graph(filepath)
        # ESCA decomposition:
        # ns  -> source ergodic sets (backward ergodic sets)
        # nw  -> sink ergodic sets (forward ergodic sets)
        # c   -> transient core
        # un  -> disconnected / unclassified nodes
        [ns, nw, c, un] = myg.FindIO(G)
        accounted = (
                len(myg.flatten(ns))
                + len(myg.flatten(nw))
                + len(c)
                + len(myg.flatten(un))
        )

        if accounted != G.number_of_nodes():
            warnings.warn(

                f"Node count mismatch: accounted = {accounted}, total = {G.number_of_nodes()}",

                UserWarning

            )
        nn = G.number_of_nodes()
        e = G.number_of_edges()
        N.append(nn)
        Nsources.append(len(myg.flatten(ns)))
        Nwells.append(len(myg.flatten(nw)))
        wells.append(nw)
        sources.append(ns)
        core.append(c)
        unconnected.append(un)
        # First-stage compression factor.
        #
        # Each ergodic set will eventually be represented by a single node.
        # The quantity  below measures the fraction of nodes removed through
        # this aggregation step alone, prior to any transport-based compression
        # of the transient core.
        C1.append((len(myg.flatten(ns))-len(ns)+len(myg.flatten(nw))-len(nw))/nn)
        graphs.append(G)

        print("Graph " + str(file) + " processed. Moving on")


if files_exist:
    N = pickle.load(open("../results/N.dat", "rb"))
    applications = pickle.load(open("../results/applications_data.dat", "rb"))
    C1 = pickle.load(open("../results/C1.dat", "rb"))
    graphs = pickle.load(open("../results/G.dat", "rb"))
    wells = pickle.load(open("../results/wells.dat", "rb"))
    sources = pickle.load(open("../results/sources.dat", "rb"))
    core = pickle.load(open("../results/core.dat", "rb"))
    unconnected = pickle.load(open("../results/unconnected.dat", "rb"))

#%% Plotting

if with_plotting:
    plt.bar(range(len(C1)), C1)
    plt.xlabel("Network index")
    plt.ylabel("First-stage compression factor C1")
    plt.show()

#%% Saving
if not files_exist:
    # Now that the variables are defined, we map them
    data_to_save = {
        "N.dat": N,
        "applications_data.dat": applications,
        "C1.dat": C1,
        "G.dat": graphs,
        "wells.dat": wells,
        "sources.dat": sources,
        "core.dat": core,
        "unconnected.dat": unconnected
    }

    for filename, obj in data_to_save.items():
        save_path = os.path.join("../results/", filename)
        with open(save_path, "wb") as f:
            pickle.dump(obj, f)