import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np


# ==============================================================
# 1. Load Data
# ==============================================================

try:

    # Step 1 data
    C1 = pickle.load(
        open("../results/C1.dat", "rb")
    )

    application = pickle.load(
        open("../results/applications_data.dat", "rb")
    )

    # Complete Step 2 dataset
    with open(
        "../results/ESCA_step2_results.dat",
        "rb"
    ) as file:

        step2 = pickle.load(file)


except FileNotFoundError as e:

    print(
        f"File not found: {e}. "
        "Please check your file paths."
    )
    raise


# ==============================================================
# 2. Extract C2 quantities from new format
# ==============================================================

C2 = np.array([
    result["C2"]
    for result in step2
])

C2_FULL = np.array([
    result["C2_FULL"]
    for result in step2
])


# ==============================================================
# 3. Data Preparation
# ==============================================================

df_wide = pd.DataFrame({

    "Application": application,

    "$C_1$": C1,

    "$C_2$": C2,

    "$C_2^{full}$": C2_FULL
})


# Melt to long format for Seaborn
df_long = df_wide.melt(

    id_vars=["Application"],

    value_vars=[
        "$C_1$",
        "$C_2$",
        "$C_2^{full}$"
    ],

    var_name="Compression_Method",

    value_name="Ratio"
)


# ==============================================================
# 4. Plotting Configuration
# ==============================================================

sns.set_context("talk")
sns.set_style("whitegrid")

plt.figure(
    figsize=(12, 8),
    dpi=300
)


# --------------------------------------------------------------
# A. Violin distributions
# --------------------------------------------------------------

sns.violinplot(

    data=df_long,

    x="Ratio",

    y="Application",

    hue="Compression_Method",

    dodge=False,

    density_norm="count",

    inner=None,

    alpha=0.3,

    palette="CMRmap"
)


# --------------------------------------------------------------
# B. Individual points
# --------------------------------------------------------------

sns.stripplot(
    data=df_long,
    x="Ratio",
    y="Application",
    hue="Compression_Method",
    dodge=False,

    alpha=0.8,

    palette="CMRmap",

    jitter=0.15,

    size=7,

    legend=False
)


# ==============================================================
# 5. Scientific Formatting
# ==============================================================

plt.xlim(
    -.2,
    1
)

plt.ylabel("")

plt.xlabel(
    "Compression Ratio",
    fontsize=20,
    labelpad=15
)

plt.title(
    "Compression Ratio Distribution by Application",
    fontsize=22,
    fontweight="bold",
    pad=20
)


# Tick labels
plt.xticks(
    fontsize=20
)

plt.yticks(
    fontsize=20
)


# Legend
plt.legend(
    title="Method",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=20,
    title_fontsize=20
)


plt.tight_layout()


# ==============================================================
# 6. Save Figure
# ==============================================================

plt.savefig(

    "../figures/compression_comparison.png",

    dpi=300,

    bbox_inches="tight"
)

plt.show()


# ==============================================================
# 7. Summary statistics
# ==============================================================

for col in [
    "$C_1$",
    "$C_2$",
    "$C_2^{full}$"
]:

    print(
        f"\n--- Statistics for {col} by Application ---"
    )

    summary = (

        df_wide

        .groupby("Application")[col]

        .agg([
            "mean",
            "median",
            "std",
            "min",
            "max"
        ])
    )

    print(summary)