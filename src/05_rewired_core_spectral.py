
import mygraph as myg
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
import pickle
import networkx as nx
import warnings
import walker
from scipy.sparse import csr_matrix
import scipy
from scipy.sparse.csgraph import connected_components
import csv

Nrewires = 10
NwalksR = 10
Nwalks = NwalksR

Plotting = False
Printing = False
stateNames = ["alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "nebraska", "nevada", "newhampshire", "newjersey", "newmexico", "newyork", "northcarolina", "northdakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhodeisland", "southcarolina", "southdakota", "tennessee", "texas", "utah", "vermont", "virginia", "washington", "westvirginia", "winsconsin", "wyoming"]

my_dict = {}
fields = ['State', 'Spectral zeroes [original]', 'Spectral zeroes [rewired]']

for stateName in stateNames:
    G = myg.import_graph("../real_world_networks/Validated/W_" +stateName+".el")
    Ncc = nx.number_weakly_connected_components(G)
    [ns, nw, n, un] = myg.FindIO(G)

    # Simulate random walks on the original graph

    if os.path.exists("../data/"+stateName+".bb") and os.path.exists("../data/"+stateName+".g"):
        B = pickle.load(open("../data/"+stateName+".bb", "rb"))
        G = pickle.load(open("../data/"+stateName+".g", "rb"))
        # run the detection before rewiring and then use the same after rewiring
    else:
        B = myg.simulate_random_walks(G, ns, nw, Nwalks)
        pickle.dump(G, open("../data/" + stateName + ".g", "wb"))
        pickle.dump(B, open("../data/"+stateName+".bb", "wb"))



    # Simulate random walks on the rewired graph

    if os.path.exists("../data/"+stateName+"_R.bb") and os.path.exists("../data/"+stateName+"_R.g"):
        GR = pickle.load(open("../data/"+stateName+"_R.g", "rb"))
        BR = pickle.load(open("../data/"+stateName+"_R.bb", "rb"))
    else:
        BR = np.zeros(np.shape(B))
        for i in range(Nrewires):
            GR = myg.rewire_core(G, n, ns, nw, un)
            BR_temp = myg.simulate_random_walks(GR, ns, nw, Nwalks)
            if np.shape(BR_temp) == np.shape(BR):
                BR = np.add(BR, np.asarray(BR_temp))
            else:
                print("Error in shape, skipping")
        pickle.dump(GR, open("../data/" + stateName + "_R.g", "wb"))
        pickle.dump(BR, open("../data/"+stateName+"_R.bb", "wb"))

    NccR = nx.number_weakly_connected_components(GR)

    B = myg.normalize_matrix_by_row(B)

    M1 = B.T @ B

    L1 = scipy.sparse.csgraph.laplacian(M1)

    sigma1 = np.linalg.eigvals(L1)
    sigma1.sort()
    if Printing:
        print(sigma1[0:Ncc+100])
    #print(sigma1)

    BR = myg.normalize_matrix_by_row(BR)

    M2 = BR.T @ BR

    L2 = scipy.sparse.csgraph.laplacian(M2)

    sigma2 = (np.linalg.eigvals(L2))
    sigma2.sort()
    if Printing:
        print(sigma2[0:NccR+100])
    #print(sigma2)

    num_sources = len(ns)
    num_wells = len(nw)

    if Plotting:

        plt.imshow(np.log(B), cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title('Adjacency Matrix Original Network [Log]')
        plt.xlabel('Sinks')
        plt.ylabel('Sources')
        plt.savefig("../figures/BlackBox_"+stateName+"Log.eps")
        plt.savefig("../figures/BlackBox_"+stateName+"Log.svg")
        plt.savefig("../figures/BlackBox_"+stateName+"Log.png")
        plt.show()

        plt.imshow(np.log(BR), cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title('Adjacency Matrix Rewired Network [Log]')
        plt.xlabel('Sinks')
        plt.ylabel('Sources')
        plt.savefig("../figures/BlackBox_"+stateName+"Log_R.eps")
        plt.savefig("../figures/BlackBox_"+stateName+"Log_R.svg")
        plt.savefig("../figures/BlackBox_"+stateName+"Log_R.png")
        plt.show()


        plt.imshow(np.log(M1), cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title("$B^T B$ Matrix Original Network [Log]")
        plt.xlabel('Sinks')
        plt.ylabel('Sources')
        plt.savefig("../figures/BlackBox_ATA_"+stateName+"Log.eps")
        plt.savefig("../figures/BlackBox_ATA_"+stateName+"Log.svg")
        plt.savefig("../figures/BlackBox_ATA_"+stateName+"Log.png")
        plt.show()

        plt.imshow(np.log(M2), cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(r"$B^T B$ Matrix Rewired Network [Log]")
        plt.xlabel('Sinks')
        plt.ylabel('Sources')
        plt.savefig("../figures/BlackBox_ATA_"+stateName+"Log_R.eps")
        plt.savefig("../figures/BlackBox_ATA_"+stateName+"Log_R.svg")
        plt.savefig("../figures/BlackBox_ATA_"+stateName+"Log_R.png")
        plt.show()

        plt.plot(np.log(sigma1), 'o', markersize=1, label="Original")
        plt.plot(np.log(sigma2), 'o', markersize=1, label="Rewired")
        plt.legend()
        plt.title(r"$\sigma(B^T B)$ Spectrum of Symmetrised Adjacency [Log]")
        plt.ylabel("Eigenvalues values")
        plt.show()

    # Builiding the spectral data

    my_dict[stateName] = [next(i for i, v in enumerate(sigma1) if v > 1e-5), next(i for i, v in enumerate(sigma2) if v > 1e-5)]

    if Printing:
        print(next(i for i,v in enumerate(sigma1) if v > 1e-5))
        print(next(i for i,v in enumerate(sigma2) if v > 1e-5))
    else:
        print("Done")

# Prepare the data for CSV
csv_data = []
for state, data in my_dict.items():
    csv_data.append({
        'State': state,
        'Spectral zeroes [original]': data[0],
        'Spectral zeroes [rewired]': data[1]
    })

with open("spectralData.csv", 'w', newline='') as csvfile:
    # creating a csv dict writer object
    writer = csv.DictWriter(csvfile, fieldnames=fields)

    # writing headers (field names)
    writer.writeheader()

    # writing data rows
    writer.writerows(csv_data)
