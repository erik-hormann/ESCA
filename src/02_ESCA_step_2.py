import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy import sparse
import mygraph as myg
import pickle
import os
import warnings
import seaborn as sns
import pandas as pd
import time


"""
ESCA Step 2: Construction of the source-sink transport matrix B
===============================================================

Pipeline position:

    Script 1: Detect ergodic sets and transient core
    Script 2: Construct transport matrix B      <-- this file
    Script 3: Perform latent-state factorisation (SVD)
    Script 4: Generate figures and statistics


Overview
--------

For each network:

    1. Load the ESCA decomposition obtained in Script 1.
    2. Check whether B has already been calculated and saved.
    3. If B exists and has the expected dimensions:
           - load the existing B,
           - do not recompute it.
    4. Otherwise:
           - construct a Markov transition matrix,
           - convert each sink ergodic set into an absorbing state,
           - initialise a uniform distribution in each source,
           - propagate it through the network,
           - construct the source-sink transport matrix B.
    5. Compute the SVD of B.
    6. Determine:
           - full SVD rank,
           - numerical matrix rank,
           - ESCA tolerance-based rank.
    7. Compute C2_FULL and C2.
    8. Store the complete dataset for the network.


Output
------

Two files:

    ../results/ESCA_step2_results.dat
        Complete results including B and singular values.

    ../results/ESCA_network_summary.csv
        Network-level summary table suitable for analysis and
        inclusion in the supplementary material / appendix.
"""


# ==============================================================
# Settings
# ==============================================================

sims_path = "../results/"

# Number of diffusion steps used to approximate
# asymptotic absorption probabilities.
NS = 200

# Relative threshold used to define significant singular values.
tol = 1e-3


# ==============================================================
# Input files from ESCA Step 1
# ==============================================================

filenames = [
    "N.dat",
    "applications_data.dat",
    "C1.dat",
    "G.dat",
    "wells.dat",
    "sources.dat",
    "core.dat",
    "unconnected.dat"
]


# ==============================================================
# Verify that Step 1 has been completed
# ==============================================================

files_exist = all(
    os.path.exists(
        os.path.join(
            sims_path,
            f
        )
    )
    for f in filenames
)

if not files_exist:

    raise FileNotFoundError(
        "Some files are missing. "
        "Run Script 1 (ESCA decomposition) first."
    )


# ==============================================================
# Load data from Step 1
# ==============================================================

data = {}

for filename in filenames:

    key = filename.replace(
        ".dat",
        ""
    )

    file_full_path = os.path.join(
        sims_path,
        filename
    )

    with open(
        file_full_path,
        "rb"
    ) as file:

        data[key] = pickle.load(
            file
        )


# ==============================================================
# Load existing Step 2 results if available
# ==============================================================

save_path = os.path.join(
    sims_path,
    "ESCA_step2_results.dat"
)


if os.path.exists(save_path):

    print()
    print(
        "Existing ESCA Step 2 results found."
    )

    print(
        f"Loading:\n{save_path}"
    )

    with open(
        save_path,
        "rb"
    ) as file:

        existing_results = pickle.load(
            file
        )

    print(
        f"Loaded {len(existing_results)} existing networks."
    )

else:

    print()
    print(
        "No existing ESCA Step 2 results found."
    )

    print(
        "B matrices will be computed from scratch."
    )

    existing_results = []


# ==============================================================
# Storage for complete Step 2 results
# ==============================================================

results = []


# ==============================================================
# Main ESCA Step 2 loop
# ==============================================================

for network_index, (
    G,
    wells,
    sources,
    core,
    unc,
    N
) in enumerate(
    zip(
        data["G"],
        data["wells"],
        data["sources"],
        data["core"],
        data["unconnected"],
        data["N"]
    )
):
    E = G.number_of_edges()
    # Start total runtime clock for this network
    network_start_time = time.perf_counter()

    print()
    print("=" * 70)
    print(
        f"Network {network_index}"
    )
    print("=" * 70)


    # ----------------------------------------------------------
    # Sanity check
    # ----------------------------------------------------------

    decomposition_size = (
        len(sources)
        + len(myg.flatten(wells))
        + len(myg.flatten(core))
        + len(myg.flatten(unc))
    )

    if N - decomposition_size != 0:

        warnings.warn(
            "Node count mismatch. "
            "The ESCA decomposition may not cover all nodes."
        )


    # ----------------------------------------------------------
    # Ergodic-set statistics
    # ----------------------------------------------------------

    source_sizes = [
        len(S)
        for S in sources
    ]

    well_sizes = [
        len(W)
        for W in wells
    ]

    total_ES = (
        len(sources)
        + len(wells)
    )

    singleton_ES = (
        sum(
            size == 1
            for size in source_sizes
        )
        +
        sum(
            size == 1
            for size in well_sizes
        )
    )

    nontrivial_ES = (
        total_ES
        - singleton_ES
    )

    singleton_ES_percentage = (
        100 * singleton_ES / total_ES
        if total_ES > 0
        else 0
    )

    max_ES_size = max(
        source_sizes + well_sizes,
        default=0
    )


    # ----------------------------------------------------------
    # Expected B dimensions
    # ----------------------------------------------------------

    nS = len(sources)
    nW = len(wells)

    expected_shape = (
        nS,
        nW
    )


    # ==========================================================
    # Check whether B already exists
    # ==========================================================

    existing_result = None

    if network_index < len(existing_results):

        candidate = existing_results[
            network_index
        ]

        if (
            isinstance(candidate, dict)
            and candidate.get(
                "network_index"
            ) == network_index
        ):

            existing_result = candidate


    # ----------------------------------------------------------
    # Check stored B
    # ----------------------------------------------------------

    B_exists = False

    if existing_result is not None:

        if "B" in existing_result:

            B_candidate = np.asarray(
                existing_result["B"]
            )

            if B_candidate.shape == expected_shape:

                B_exists = True

                B = B_candidate


    # ==========================================================
    # Time B construction / loading
    # ==========================================================

    B_start_time = time.perf_counter()


    # ==========================================================
    # Use existing B
    # ==========================================================

    if B_exists:

        print(
            f"Existing B found: "
            f"shape={B.shape}"
        )

        print(
            "Skipping B construction."
        )


    # ==========================================================
    # Compute B
    # ==========================================================

    else:

        print(
            f"Computing B: "
            f"shape={expected_shape}"
        )


        # ------------------------------------------------------
        # Convert sink nodes into absorbing states
        # ------------------------------------------------------

        G.add_edges_from(
            (node, node)
            for node in myg.flatten(
                wells
            )
        )


        # ------------------------------------------------------
        # Construct sparse adjacency matrix
        # ------------------------------------------------------

        adj = nx.to_scipy_sparse_array(
            G,
            format="csr"
        )


        # ------------------------------------------------------
        # Construct row-stochastic transition matrix P
        # ------------------------------------------------------

        # P[i,j] is the probability of moving from
        # node i to node j.

        row_sums = np.array(
            adj.sum(
                axis=1
            )
        ).flatten()


        # Protect against isolated nodes.
        row_sums[
            row_sums == 0
        ] = 1.0


        inv_row_sums = (
            1.0 / row_sums
        )


        D_inv = sparse.diags(
            inv_row_sums
        )


        P = (
            D_inv @ adj
        )


        # ------------------------------------------------------
        # Construct source-sink transport matrix B
        # ------------------------------------------------------

        B = np.zeros(
            expected_shape
        )


        # ------------------------------------------------------
        # Construct B row by row
        # ------------------------------------------------------

        for i, S in enumerate(
            sources
        ):

            # --------------------------------------------------
            # Uniform initial distribution over source set
            # --------------------------------------------------

            v = np.zeros(
                P.shape[0]
            )


            v[S] = 1.0


            v = (
                v / v.sum()
            )


            # --------------------------------------------------
            # Propagate distribution
            # --------------------------------------------------

            for _ in range(NS):

                v = v @ P


            # --------------------------------------------------
            # Final distribution
            # --------------------------------------------------

            v_final = np.asarray(
                v
            ).flatten()


            # --------------------------------------------------
            # Aggregate probability inside each sink
            # --------------------------------------------------

            for j, well_nodes in enumerate(
                wells
            ):

                well_probability = (
                    v_final[
                        well_nodes
                    ].sum()
                )


                B[i, j] = (
                    well_probability
                )


        print(
            "B construction complete."
        )


    B_runtime = (
        time.perf_counter()
        - B_start_time
    )


    # ==========================================================
    # SVD of B
    # ==========================================================

    SVD_start_time = time.perf_counter()


    U, s_full, Vh = np.linalg.svd(
        B,
        full_matrices=False
    )


    # ----------------------------------------------------------
    # Full SVD rank
    # ----------------------------------------------------------

    rank_full = len(
        s_full
    )


    # ----------------------------------------------------------
    # Numerical matrix rank
    # ----------------------------------------------------------

    rank = np.linalg.matrix_rank(
        B
    )


    # ==========================================================
    # ESCA significant singular values
    # ==========================================================

    if (
        len(s_full) > 0
        and s_full[0] > 0
    ):

        significant = (
            s_full
            >= tol * s_full[0]
        )


        rank_ESCA = int(
            np.sum(
                significant
            )
        )


        # Keep only significant singular values.
        s = s_full[
            significant
        ]


    else:

        significant = np.array(
            [],
            dtype=bool
        )


        rank_ESCA = 0


        s = np.array(
            []
        )


    # ==========================================================
    # Full-rank representation
    # ==========================================================

    N2_FULL = (
        len(sources)
        + len(wells)
        + rank_full
    )


    C2_FULL = (
        1
        - N2_FULL / N
    )


    # ==========================================================
    # Tolerance-based representation
    # ==========================================================

    N2 = (
        len(sources)
        + len(wells)
        + rank_ESCA
    )


    C2 = (
        1
        - N2 / N
    )


    # ==========================================================
    # Runtime
    # ==============================================================

    SVD_runtime = (
        time.perf_counter()
        - SVD_start_time
    )

    total_runtime = (
        time.perf_counter()
        - network_start_time
    )


    # ==========================================================
    # Store complete network dataset
    # ==============================================================

    results.append({

        # ------------------------------------------------------
        # Network identification / size
        # ------------------------------------------------------

        "network_index": network_index,
        "N": N,
        "E": E,

        # ------------------------------------------------------
        # ESCA decomposition
        # ------------------------------------------------------

        "n_sources": len(sources),

        "n_wells": len(wells),

        "source_nodes": len(
            myg.flatten(
                sources
            )
        ),

        "well_nodes": len(
            myg.flatten(
                wells
            )
        ),

        "core_size": len(
            myg.flatten(
                core
            )
        ),

        "unconnected_size": len(
            myg.flatten(
                unc
            )
        ),


        # ------------------------------------------------------
        # Ergodic-set statistics
        # ------------------------------------------------------

        "total_ES": total_ES,

        "singleton_ES": singleton_ES,

        "nontrivial_ES": nontrivial_ES,

        "singleton_ES_percentage": (
            singleton_ES_percentage
        ),

        "max_ES_size": max_ES_size,


        # ------------------------------------------------------
        # Transport matrix
        # ------------------------------------------------------

        "B": B,


        # ------------------------------------------------------
        # SVD
        # ------------------------------------------------------

        "s_full": s_full,

        "s": s,


        # ------------------------------------------------------
        # Ranks
        # ------------------------------------------------------

        "rank_full": rank_full,

        "rank": rank,

        "rank_ESCA": rank_ESCA,


        # ------------------------------------------------------
        # Compressed network sizes
        # ------------------------------------------------------

        "N2_FULL": N2_FULL,

        "N2": N2,


        # ------------------------------------------------------
        # Compression coefficients
        # ------------------------------------------------------

        "C2_FULL": C2_FULL,

        "C2": C2,


        # ------------------------------------------------------
        # Runtime
        # ------------------------------------------------------

        "B_runtime_seconds": B_runtime,

        "SVD_runtime_seconds": SVD_runtime,

        "total_runtime_seconds": total_runtime,


        # ------------------------------------------------------
        # Parameters
        # ------------------------------------------------------

        "tol": tol,

        "NS": NS
    })


    # ----------------------------------------------------------
    # Print summary
    # ----------------------------------------------------------

    if B_exists:

        print(
            "B loaded from existing results."
        )

    else:

        print(
            "B newly computed."
        )


    print(
        f"B={B.shape}, "
        f"rank={rank}, "
        f"rank_ESCA={rank_ESCA}, "
        f"rank_full={rank_full}, "
        f"rank/rank_full="
        f"{rank / rank_full:.3f}, "
        f"ESCA/rank_full="
        f"{rank_ESCA / rank_full:.3f}"
    )

    print(
        f"ES: total={total_ES}, "
        f"singletons={singleton_ES} "
        f"({singleton_ES_percentage:.2f}%), "
        f"non-trivial={nontrivial_ES}, "
        f"max size={max_ES_size}"
    )

    print(
        f"Runtime: B={B_runtime:.3f}s, "
        f"SVD={SVD_runtime:.3f}s, "
        f"total={total_runtime:.3f}s"
    )


# ==============================================================
# Save complete Step 2 dataset
# ==============================================================

with open(
    save_path,
    "wb"
) as file:

    pickle.dump(
        results,
        file
    )


print()
print("=" * 70)
print("ESCA Step 2 complete")
print("=" * 70)

print(
    f"Saved {len(results)} networks to:"
)

print(
    save_path
)


# ==============================================================
# Export network-level summary table
# ==============================================================

summary_rows = []


for r, application in zip(
    results,
    data["applications_data"]
):

    summary_rows.append({

        # ------------------------------------------------------
        # Network identification
        # ------------------------------------------------------

        "network_index": r["network_index"],

        "application": application,


        # ------------------------------------------------------
        # Original network
        # ------------------------------------------------------

        "N": r["N"],

        "E": r["E"],

        # ------------------------------------------------------
        # ESCA decomposition
        # ------------------------------------------------------

        "N_sources": r["n_sources"],

        "N_wells": r["n_wells"],

        "source_nodes": r["source_nodes"],

        "well_nodes": r["well_nodes"],

        "N_core": r["core_size"],

        "N_unconnected": r["unconnected_size"],


        # ------------------------------------------------------
        # Ergodic-set statistics
        # ------------------------------------------------------

        "total_ES": r["total_ES"],

        "singleton_ES": r["singleton_ES"],

        "nontrivial_ES": r["nontrivial_ES"],

        "singleton_ES_percentage": (
            r["singleton_ES_percentage"]
        ),

        "max_ES_size": r["max_ES_size"],


        # ------------------------------------------------------
        # Compression
        # ------------------------------------------------------

        "C1": data["C1"][
            r["network_index"]
        ],

        "C2_full": r["C2_FULL"],

        "C2": r["C2"],


        # ------------------------------------------------------
        # SVD
        # ------------------------------------------------------

        "rank_full": r["rank_full"],

        "rank": r["rank"],

        "rank_ESCA": r["rank_ESCA"],


        # ------------------------------------------------------
        # Compressed network sizes
        # ------------------------------------------------------

        "N2_full": r["N2_FULL"],

        "N2": r["N2"],


        # ------------------------------------------------------
        # Runtime
        # ------------------------------------------------------

        "B_runtime_seconds": (
            r["B_runtime_seconds"]
        ),

        "SVD_runtime_seconds": (
            r["SVD_runtime_seconds"]
        ),

        "total_runtime_seconds": (
            r["total_runtime_seconds"]
        ),


        # ------------------------------------------------------
        # Parameters
        # ------------------------------------------------------

        "tolerance": r["tol"],

        "diffusion_steps": r["NS"],
    })


summary_df = pd.DataFrame(
    summary_rows
)


csv_path = os.path.join(
    sims_path,
    "ESCA_network_summary.csv"
)


summary_df.to_csv(
    csv_path,
    index=False
)


print()
print(
    "Network summary table saved to:"
)

print(
    csv_path
)


# ==============================================================
# Convenience arrays for plotting
# ==============================================================

N_array = np.array([
    r["N"]
    for r in results
])


C2_array = np.array([
    r["C2"]
    for r in results
])


C2_FULL_array = np.array([
    r["C2_FULL"]
    for r in results
])


core_size_array = np.array([
    r["core_size"]
    for r in results
])


# ==============================================================
# Plotting data
# ==============================================================

xdata = np.divide(
    core_size_array,
    N_array
)


x = np.linspace(
    0,
    1,
    100
)


# ==============================================================
# Plot C2
# ==============================================================

sns.set_context("talk")
sns.set_style("whitegrid")


# Match the colours used in the main compression figure
cmrmap = sns.color_palette(
    "CMRmap",
    3
)


c2_colour = cmrmap[1]

c2_full_colour = cmrmap[2]


plt.figure(
    figsize=(12, 8),
    dpi=300
)


# --------------------------------------------------------------
# C2
# --------------------------------------------------------------

plt.scatter(
    xdata,
    C2_array,
    s=70,
    alpha=0.8,
    color=c2_colour,
    label=r"$C_2$"
)


# --------------------------------------------------------------
# C2 full
# --------------------------------------------------------------

#plt.scatter(
#    xdata,
#    C2_FULL_array,
#    s=70,
#    alpha=0.8,
#    color=c2_full_colour,
#    label=r"$C_2^{\mathrm{full}}$"
#)


# --------------------------------------------------------------
# Reference line
# --------------------------------------------------------------

plt.plot(
    x,
    x,
    color="black",
    linestyle="--",
    linewidth=1.5,
    alpha=0.6,
    label=r"$\frac{N_C}{N} = C_2$"
)


# --------------------------------------------------------------
# Labels
# --------------------------------------------------------------

plt.xlabel(
    r"$N_C/N$",
    fontsize=20,
    labelpad=15
)


plt.ylabel(
    "Compression Ratio",
    fontsize=20,
    labelpad=15
)


# --------------------------------------------------------------
# Ticks
# --------------------------------------------------------------

plt.xticks(
    fontsize=20
)


plt.yticks(
    fontsize=20
)


# --------------------------------------------------------------
# Limits
# --------------------------------------------------------------

plt.xlim(
    0,
    1
)


plt.ylim(
    #min(
        C2_array.min() - 0.05,
        #C2_FULL_array.min()
    #) - 0.05,
    1
)


# --------------------------------------------------------------
# Legend
# --------------------------------------------------------------

plt.legend(
    fontsize=16
)


plt.tight_layout()


# ==============================================================
# Save
# ==============================================================

plt.savefig(
    "../figures/C2_compression.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()