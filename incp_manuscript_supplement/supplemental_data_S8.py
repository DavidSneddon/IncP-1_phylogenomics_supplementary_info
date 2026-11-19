# ============================================================
# Resistance-category carriage analysis WITH 99,999 permutations
#
# Overall tests: Pearson chi-squared statistic with permutation p-values
# Pairwise tests: exact two-sided Fisher tests
# Multiple testing: Holm correction
# ============================================================

from pathlib import Path
from itertools import combinations
from textwrap import fill


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.transforms import Bbox

from scipy.stats import fisher_exact


# ---------- Settings ----------
# This path should point to the data from the "Resistance Categ Binary Counts" workbook in the supplemental_data_S4 Excel file. Cut and pase this data to a file called resistance_category_count.csv in the path you provide below.
folder = Path(
    r"/our/path/here/"
)

input_file = folder / "resistance_category_count.csv"

alpha = 0.05
seed = 1
n_permutations = 99999

categories = {
    "Drugs": "drug_genes",
    "Metals": "metal_genes",
    "Multi-compound": "multicompound_genes",
}

category_names = list(categories)
category_columns = list(categories.values())


# ---------- Holm correction ----------

def holm_adjust(p_values):
    """Adjust one complete family of p-values using Holm's method."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="stable")

    adjusted_sorted = np.minimum(
        1.0,
        np.maximum.accumulate(
            p[order] * np.arange(len(p), 0, -1)
        ),
    )

    adjusted = np.empty(len(p))
    adjusted[order] = adjusted_sorted

    return adjusted


# ---------- Step 1: Read and validate ----------

resistance_category_count = pd.read_csv(input_file)

required = ["plasmid_id", "subgroup", *category_columns]

missing_columns = set(required) - set(resistance_category_count.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {sorted(missing_columns)}")

dat = resistance_category_count[required].copy()

for column in ["plasmid_id", "subgroup"]:
    dat[column] = dat[column].astype("string").str.strip()

if dat.isna().any().any():
    raise ValueError("Missing values found.")

if dat[["plasmid_id", "subgroup"]].eq("").any().any():
    raise ValueError("Blank plasmid identifiers or subgroup assignments.")

if len(dat) != 187:
    raise ValueError(f"Expected 187 plasmids; found {len(dat)}.")

if dat["plasmid_id"].duplicated().any():
    raise ValueError("Duplicate plasmid identifiers found.")

for column in category_columns:
    dat[column] = pd.to_numeric(dat[column], errors="raise")

    if not dat[column].isin([0, 1]).all():
        raise ValueError(f"{column} must contain only 0 and 1.")

    dat[column] = dat[column].astype(int)

if (
    not isinstance(n_permutations, (int, np.integer))
    or n_permutations < 1
):
    raise ValueError("n_permutations must be a positive integer.")

groups = sorted(dat["subgroup"].unique())
sizes = dat.groupby("subgroup").size().reindex(groups)

n_groups = len(groups)
n_plasmids = len(dat)

if n_groups < 2:
    raise ValueError("At least two subgroups are required.")

print("\nInput validation passed.")
print(f"Plasmids: {n_plasmids}")
print(f"Subgroups: {n_groups}")
print(sizes.to_string())


# ---------- Steps 2–3: Frequencies and relative frequencies ----------

frequency_rows = []

for category, column in categories.items():

    positives = (
        dat.groupby("subgroup")[column]
        .sum()
        .reindex(groups)
    )

    overall_frequency = dat[column].mean()

    for group in groups:

        n = int(sizes[group])
        present = int(positives[group])
        frequency = present / n

        frequency_rows.append({
            "subgroup": group,
            "category": category,
            "n_total": n,
            "n_present": present,
            "n_absent": n - present,
            "frequency": frequency,
            "percent_present": 100 * frequency,
            "overall_frequency": overall_frequency,
            "relative_frequency": (
                frequency / overall_frequency
                if overall_frequency > 0 else np.nan
            ),
        })

subgroup_frequencies = pd.DataFrame(frequency_rows)

relative_frequency_table = (
    subgroup_frequencies.pivot(
        index="subgroup",
        columns="category",
        values="relative_frequency",
    )
    .reindex(index=groups, columns=category_names)
)

print("\nRelative carriage frequencies:")
print(relative_frequency_table.round(3).to_string())


# ---------- Step 4: Construct observed contingency tables ----------

# Numeric subgroup codes identify the fixed subgroup positions.
group_codes = pd.Categorical(
    dat["subgroup"],
    categories=groups,
).codes

# One row per plasmid, one column per resistance category.
outcomes = dat[category_columns].to_numpy(dtype=int)

def count_positives(values):
    """Count positives per subgroup for each category separately."""
    return np.column_stack([
        np.bincount(
            group_codes,
            weights=values[:, j],
            minlength=n_groups,
        )
        for j in range(len(category_names))
    ])

observed_positive = count_positives(outcomes)

group_sizes = sizes.to_numpy(dtype=float)[:, None]
overall_frequencies = outcomes.mean(axis=0)

expected_positive = group_sizes * overall_frequencies
expected_absent = group_sizes * (1 - overall_frequencies)

# For binary outcomes, the usual Pearson statistic can be written:
#
# sum over subgroups:
# (observed_positive - expected_positive)^2 /
# (subgroup_size * overall_frequency * (1 - overall_frequency))
#
# This equals the statistic calculated from the full
# subgroup x presence/absence contingency table.

denominators = (
    group_sizes
    * overall_frequencies
    * (1 - overall_frequencies)
)

def pearson_statistics(positive_counts):
    """Return one Pearson statistic for each resistance category."""
    contributions = np.divide(
        (positive_counts - expected_positive) ** 2,
        denominators,
        out=np.zeros_like(expected_positive, dtype=float),
        where=denominators > 0,
    )
    return contributions.sum(axis=0)

observed_statistics = pearson_statistics(observed_positive)

# Preserve observed and expected contingency-table counts.
contingency_rows = []

for j, category in enumerate(category_names):
    for i, group in enumerate(groups):

        contingency_rows.append({
            "category": category,
            "subgroup": group,
            "observed_present": int(observed_positive[i, j]),
            "observed_absent": int(
                sizes[group] - observed_positive[i, j]
            ),
            "expected_present": expected_positive[i, j],
            "expected_absent": expected_absent[i, j],
        })

contingency_tables = pd.DataFrame(contingency_rows)


# ---------- Step 5: Overall permutation tests ----------

# Each permutation:
# 1. Keeps subgroup sizes fixed.
# 2. Shuffles complete three-category plasmid profiles.
# 3. Preserves each category's total positive count.
# 4. Calculates a separate statistic for each category.

rng = np.random.default_rng(seed)

extreme_counts = np.zeros(len(category_names), dtype=int)

# Small numerical tolerance includes theoretical ties.
tolerance = (
    100 * np.finfo(float).eps
    * np.maximum(1.0, np.abs(observed_statistics))
)

print(f"\nRunning {n_permutations:,} permutations...")

for iteration in range(n_permutations):

    shuffled_outcomes = outcomes[rng.permutation(n_plasmids)]

    shuffled_positive = count_positives(shuffled_outcomes)

    shuffled_statistics = pearson_statistics(shuffled_positive)

    extreme_counts += (
        shuffled_statistics >= observed_statistics - tolerance
    )

    if (iteration + 1) % 10000 == 0:
        print(f"Completed {iteration + 1:,} permutations.")

# Added ones ensure that permutation p-values are never zero.
permutation_p_values = (
    (extreme_counts + 1) / (n_permutations + 1)
)

overall_rows = []

for j, category in enumerate(category_names):

    overall_rows.append({
        "category": category,
        "n_plasmids": n_plasmids,
        "n_subgroups": n_groups,
        "n_present": int(outcomes[:, j].sum()),
        "n_absent": int((outcomes[:, j] == 0).sum()),
        "chi_squared": observed_statistics[j],
        "p_value": permutation_p_values[j],
        "extreme_permutations": extreme_counts[j],
        "n_permutations": n_permutations,
        "seed": seed,
        "constant_outcome": bool(
            overall_frequencies[j] in (0, 1)
        ),
        "method": "Pearson statistic with permutation p-value",
    })

overall_results = pd.DataFrame(overall_rows)


# ---------- Step 6: Holm correction across three overall tests ----------

overall_results["p_holm"] = holm_adjust(
    overall_results["p_value"]
)

overall_results["significant"] = (
    overall_results["p_holm"] < alpha
)

print("\nOverall permutation tests:")
print(
    overall_results[
        [
            "category",
            "chi_squared",
            "p_value",
            "p_holm",
            "significant",
        ]
    ].to_string(index=False)
)


# ---------- Step 7: Exact pairwise Fisher tests ----------

# These tests do not use simulations.
pairwise_rows = []

for j, category in enumerate(category_names):

    for i_1, i_2 in combinations(range(n_groups), 2):

        group_1 = groups[i_1]
        group_2 = groups[i_2]

        n_1 = int(sizes[group_1])
        n_2 = int(sizes[group_2])

        present_1 = int(observed_positive[i_1, j])
        present_2 = int(observed_positive[i_2, j])

        absent_1 = n_1 - present_1
        absent_2 = n_2 - present_2

        pair_table = np.array([
            [present_1, absent_1],
            [present_2, absent_2],
        ])

        # Both groups entirely negative or both entirely positive.
        if np.any(pair_table.sum(axis=0) == 0):
            pair_p = 1.0
        else:
            _, pair_p = fisher_exact(
                pair_table,
                alternative="two-sided",
            )

        pairwise_rows.append({
            "category": category,
            "subgroup_1": group_1,
            "subgroup_2": group_2,
            "n_1": n_1,
            "present_1": present_1,
            "absent_1": absent_1,
            "percent_present_1": 100 * present_1 / n_1,
            "n_2": n_2,
            "present_2": present_2,
            "absent_2": absent_2,
            "percent_present_2": 100 * present_2 / n_2,
            "difference_percentage_points": (
                100 * (present_1 / n_1 - present_2 / n_2)
            ),
            "p_value": pair_p,
        })

pairwise_results = pd.DataFrame(pairwise_rows)


# ---------- Step 8: Holm correction across ALL pairwise tests ----------

# With 14 subgroups:
# 91 subgroup pairs x 3 categories = 273 comparisons.
# Apply Holm once across the complete family.

pairwise_results["p_holm"] = holm_adjust(
    pairwise_results["p_value"]
)

pairwise_results["pairwise_significant"] = (
    pairwise_results["p_holm"] < alpha
)

overall_significant = overall_results.set_index(
    "category"
)["significant"]

pairwise_results["overall_category_significant"] = (
    pairwise_results["category"].map(overall_significant)
)

pairwise_results["supported"] = (
    pairwise_results["pairwise_significant"]
    & pairwise_results["overall_category_significant"]
)

pairwise_results = pairwise_results.sort_values(
    ["p_holm", "p_value"]
).reset_index(drop=True)

supported_pairs = pairwise_results.loc[
    pairwise_results["supported"]
].copy()

print(f"\nNumber of pairwise tests: {len(pairwise_results)}")
print(f"Supported pairwise differences: {len(supported_pairs)}")

print(
    supported_pairs[
        [
            "category",
            "subgroup_1",
            "subgroup_2",
            "percent_present_1",
            "percent_present_2",
            "difference_percentage_points",
            "p_holm",
        ]
    ].to_string(index=False)
)


# ---------- Step 9: Save numerical results ----------

tables_to_save = {
    "subgroup_category_frequencies": subgroup_frequencies,
    "contingency_tables": contingency_tables,
    "overall_category_tests": overall_results,
    "pairwise_category_tests": pairwise_results,
    "supported_pairwise_differences": supported_pairs,
}

for filename, table in tables_to_save.items():

    table.to_csv(
        folder / f"{filename}_with_permutations.csv",
        index=False,
    )

relative_frequency_table.to_csv(
    folder / "relative_category_frequencies_with_permutations.csv"
)


# ============================================================
# Combined figure:
# A. Relative carriage frequencies
# B. Significant pairwise comparisons and adjusted p-values
# ============================================================

from matplotlib.colors import (
    TwoSlopeNorm,
    LinearSegmentedColormap,
)

# ---------- Prepare the plotting data ----------

category_names = list(categories)

preferred_order = [
    "alpha", "beta_1", "beta_2", "delta",
    "epsilon_1", "epsilon_2", "gamma",
    "iota", "kappa", "rho", "zeta",
    "eta", "theta", "lambda",
]

available_groups = set(dat["subgroup"])

plot_groups = [
    group for group in preferred_order
    if group in available_groups
]

plot_groups += sorted(
    available_groups - set(plot_groups)
)

greek_labels = {
    "alpha": "α",
    "beta_1": "β₁",
    "beta_2": "β₂",
    "gamma": "γ",
    "delta": "δ",
    "epsilon_1": "ε₁",
    "epsilon_2": "ε₂",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "rho": "ρ",
}

def subgroup_label(group):
    return greek_labels.get(group, group)


plot_sizes = dat.groupby("subgroup").size()

relative = (
    subgroup_frequencies.pivot(
        index="subgroup",
        columns="category",
        values="relative_frequency",
    )
    .reindex(
        index=plot_groups,
        columns=category_names,
    )
)

relative_lookup = subgroup_frequencies.set_index(
    ["subgroup", "category"]
)["relative_frequency"]

# Require significance in both the Holm-adjusted pairwise test
# and the corresponding Holm-adjusted overall test.
significant_pairs = pairwise_results.loc[
    pairwise_results["supported"]
].copy()

lower_color = "#287B9A"
higher_color = "#BA512D"
text_color = "#233240"
muted_color = "#596773"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 22,
    "text.color": text_color,
    "axes.labelcolor": text_color,
    "xtick.color": text_color,
    "ytick.color": text_color,
    "pdf.fonttype": 42,
})


# ---------- Figure layout ----------

fig = plt.figure(
    figsize=(30, 15),
    facecolor="white",
)

layout = fig.add_gridspec(
    1,
    5,
    width_ratios=[4.2, 0.75, 1.7, 3.9, 1.45],
    left=0.085,
    right=0.975,
    top=0.81,
    bottom=0.21,
    wspace=0.04,
)

heat_ax = fig.add_subplot(layout[0, 0])
pair_label_ax = fig.add_subplot(layout[0, 2])
dot_ax = fig.add_subplot(layout[0, 3])
pvalue_ax = fig.add_subplot(layout[0, 4])


# ---------- Panel A: Relative-frequency heatmap ----------

values = relative.to_numpy(dtype=float)
finite_values = values[np.isfinite(values)]

maximum = (
    max(2.0, float(finite_values.max()))
    if finite_values.size else 2.0
)

heat_cmap = LinearSegmentedColormap.from_list(
    "carriage",
    ["#235E83", "#F8FAFB", "#B64A2F"],
)

heat_cmap.set_bad("lightgrey")

heat_image = heat_ax.imshow(
    values,
    aspect="auto",
    cmap=heat_cmap,
    norm=TwoSlopeNorm(
        vmin=0,
        vcenter=1,
        vmax=maximum,
    ),
)

column_labels = [
    "Multi-compound"
    if category in ("Multicompound", "Multi-compound")
    else category
    for category in category_names
]

heat_ax.set_xticks(range(len(category_names)))
heat_ax.set_xticklabels(
    column_labels,
    fontsize=22,
    fontweight="bold",
)

heat_ax.set_yticks(range(len(plot_groups)))
heat_ax.set_yticklabels([
    f"{subgroup_label(group)}  ({plot_sizes[group]})"
    for group in plot_groups
], fontsize=22)

heat_ax.set_ylabel(
    "Subgroup (total plasmids)",
    fontsize=22,
    labelpad=12,
)

heat_ax.tick_params(
    axis="both",
    length=0,
    pad=9,
)

heat_ax.set_xticks(
    np.arange(-0.5, len(category_names), 1),
    minor=True,
)

heat_ax.set_yticks(
    np.arange(-0.5, len(plot_groups), 1),
    minor=True,
)

heat_ax.grid(
    which="minor",
    color="white",
    linewidth=1.5,
)

heat_ax.tick_params(which="minor", length=0)

for spine in heat_ax.spines.values():
    spine.set_visible(False)

for i, group in enumerate(plot_groups):
    for j, category in enumerate(category_names):

        value = relative.loc[group, category]

        if pd.isna(value):
            label = "NA"
            label_color = text_color

        else:
            label = f"{value:.2f}"

            red, green, blue, _ = heat_cmap(
                heat_image.norm(value)
            )

            brightness = (
                0.2126 * red
                + 0.7152 * green
                + 0.0722 * blue
            )

            label_color = (
                "white"
                if brightness < 0.48
                else text_color
            )

        heat_ax.text(
            j,
            i,
            label,
            ha="center",
            va="center",
            fontsize=22,
            color=label_color,
        )

# Center the horizontal colorbar beneath Panel A.
heat_position = heat_ax.get_position()

colorbar_ax = fig.add_axes([
    heat_position.x0 + heat_position.width * 0.10,
    0.135,
    heat_position.width * 0.80,
    0.016,
])

colorbar = fig.colorbar(
    heat_image,
    cax=colorbar_ax,
    orientation="horizontal",
)

colorbar.outline.set_visible(False)
colorbar.ax.tick_params(
    length=0,
    labelsize=18,
    pad=3,
)

colorbar.set_label(
    "Relative frequency of resistance category",
    fontsize=22,
    labelpad=10,
)

fig.text(
    heat_ax.get_position().x0
    + heat_ax.get_position().width / 2,
    (0.853 + heat_ax.get_position().y1) / 2,
    r"Pearson chi-square p-value = $3 \times 10^{-5}$ for each resistance category",
    ha="center",
    va="center",
    fontstyle="italic",
    fontsize=20,
    color=muted_color,
)
# ---------- Panel B: Significant-pair dumbbell plot ----------

plot_rows = []
category_headers = []
empty_categories = []

y = 0.0

for category in category_names:

    category_headers.append((y, category))
    y += 1.05

    category_pairs = significant_pairs.loc[
        significant_pairs["category"] == category
    ].sort_values("p_holm")

    if category_pairs.empty:
        empty_categories.append(y)
        y += 1.0

    else:
        for row in category_pairs.itertuples(index=False):

            lower_group = row.subgroup_1
            higher_group = row.subgroup_2

            lower_frequency = relative_lookup[
                lower_group, category
            ]

            higher_frequency = relative_lookup[
                higher_group, category
            ]

            if lower_frequency > higher_frequency:

                lower_group, higher_group = (
                    higher_group,
                    lower_group,
                )

                lower_frequency, higher_frequency = (
                    higher_frequency,
                    lower_frequency,
                )

            plot_rows.append({
                "y": y,
                "lower_group": lower_group,
                "higher_group": higher_group,
                "lower_frequency": lower_frequency,
                "higher_frequency": higher_frequency,
                "p_holm": row.p_holm,
            })

            y += 1.0

    y += 0.65

y_maximum = y - 0.55

for axis in [pair_label_ax, dot_ax, pvalue_ax]:

    axis.set_ylim(y_maximum, -0.65)
    axis.set_yticks([])

    for spine in axis.spines.values():
        spine.set_visible(False)

pair_label_ax.set_xlim(0, 1)
pair_label_ax.set_xticks([])

pvalue_ax.set_xlim(0, 1)
pvalue_ax.set_xticks([])

dot_maximum = max(3.4, maximum + 0.2)

dot_ax.set_xlim(-0.12, dot_maximum)

dot_ax.set_xticks(
    np.arange(0, np.floor(dot_maximum) + 1)
)

dot_ax.set_xticklabels(
    [f"{tick:g}" for tick in dot_ax.get_xticks()],
    fontsize=20,
    fontweight="bold",
)

dot_ax.set_xlabel(
    "Relative frequency of resistance category",
    fontsize=22,
    labelpad=24,
)

dot_ax.tick_params(
    axis="x",
    length=0,
    labelsize=20,
)

for x in dot_ax.get_xticks():
    if x != 1:
        dot_ax.axvline(
            x,
            color="#E5EAED",
            linewidth=0.8,
            zorder=0,
        )

dot_ax.axvline(
    1,
    color="#8997A2",
    linewidth=1,
    linestyle=(0, (3, 3)),
    zorder=0,
)

for header_y, category in category_headers:

    display_category = (
        "Multi-compound"
        if category == "Multicompound"
        else category
    )

    pair_label_ax.text(
        0.02,
        header_y,
        display_category,
        fontweight="bold",
        fontsize=22,
        va="center",
    )

    for axis in [pair_label_ax, dot_ax, pvalue_ax]:
        axis.axhline(
            header_y + 0.48,
            color="#DDE4E9",
            linewidth=0.8,
            zorder=0,
        )

for empty_y in empty_categories:
    pair_label_ax.text(
        0.02,
        empty_y,
        "No significant pairs",
        fontsize=18,
        color=muted_color,
        va="center",
    )

for row in plot_rows:

    row_y = row["y"]

    pair_label_ax.text(
        0.08,
        row_y,
        subgroup_label(row["lower_group"]),
        color=lower_color,
        va="center",
        fontsize=24,
        fontweight="bold",
    )

    pair_label_ax.text(
        0.45,
        row_y,
        "vs",
        color=muted_color,
        ha="center",
        va="center",
        fontsize=18,
    )

    pair_label_ax.text(
        0.73,
        row_y,
        subgroup_label(row["higher_group"]),
        color=higher_color,
        va="center",
        fontsize=24,
        fontweight="bold",
    )

    dot_ax.plot(
        [
            row["lower_frequency"],
            row["higher_frequency"],
        ],
        [row_y, row_y],
        color="#A8B3BC",
        linewidth=2.2,
        zorder=2,
    )
    
    if row["lower_frequency"] > 0:
        fold_ratio = (
            row["higher_frequency"] / row["lower_frequency"]
        )
        fold_label = f"{fold_ratio:.2f}×"
    else:
        absolute_difference = abs(
            row["higher_frequency"] - row["lower_frequency"]
        )
        fold_label = f"Δ = {absolute_difference:.2f}"

    dot_ax.text(
        (
            row["lower_frequency"]
            + row["higher_frequency"]
        ) / 2,
        row_y,
        fold_label,
        ha="center",
        va="center",
        fontsize=16,
        color=text_color,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "pad": 2,
        },
        zorder=4,
    )
    
    dot_ax.scatter(
        row["lower_frequency"],
        row_y,
        s=126,
        color=lower_color,
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
    )

    dot_ax.scatter(
        row["higher_frequency"],
        row_y,
        s=126,
        color=higher_color,
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
    )

    p_value = row["p_holm"]

    if p_value == 0:
        p_label = "Below numerical precision"

    elif p_value < 0.001:

        exponent = int(np.floor(np.log10(p_value)))
        coefficient = p_value / (10.0 ** exponent)

        p_label = (
            rf"${coefficient:.2f}"
            rf"\times10^{{{exponent}}}$"
        )

    else:
        p_label = f"{p_value:.4f}"

    pvalue_ax.text(
        0.12,
        row_y,
        p_label,
        va="center",
        fontsize=20,
    )


# ---------- Titles, legends, and caption ----------

fig.text(
    0.085,
    0.948,
    "Carriage of resistance genes by category among IncP-1 plasmid subgroups",
    fontsize=44,
    fontweight="bold",
)

fig.text(
    0.085,
    0.909,
    f"\n{len(dat)} plasmids across {len(plot_groups)} subgroups; "
    "presence/absence counted within plasmids of each subgroup for each resistance category (Drugs, Metals, or Multi-compound)",
    fontsize=24,
    color=muted_color,
)

fig.text(
    0.085, 0.853, "A",
    fontsize=32,
    fontweight="bold",
)

fig.text(
    0.109, 0.853, "Relative frequency of categories",
    fontsize=28,
    fontweight="bold",
)

panel_b_x = pair_label_ax.get_position().x0

fig.text(
    panel_b_x, 0.853, "B",
    fontsize=32,
    fontweight="bold",
)

b_heading = fig.text(
    panel_b_x + 0.026,
    0.853,
    "Subgroup pairs with differences in frequency of categories",
    fontsize=28,
    fontweight="bold",
)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()

if pvalue_ax.texts:
    value_boxes = [
        text.get_window_extent(renderer)
        for text in pvalue_ax.texts
    ]

    heading_x = fig.transFigure.inverted().transform((
        (
            min(box.x0 for box in value_boxes)
            + max(box.x1 for box in value_boxes)
        ) / 2,
        0,
    ))[0]

    heading_y = fig.transFigure.inverted().transform((
        0,
        (
            b_heading.get_window_extent(renderer).y0
            + max(box.y1 for box in value_boxes)
        ) / 2,
    ))[1]

else:
    heading_x = (
        pvalue_ax.get_position().x0
        + pvalue_ax.get_position().width / 2
    )
    heading_y = (
        0.853 + pvalue_ax.get_position().y1
    ) / 2

fig.text(
    heading_x,
    heading_y,
    "Two-sided Fisher's test p-value",
    fontstyle="italic",
    ha="center",
    va="center",
    fontsize=20,
    color=muted_color,
)

fig.text(
    panel_b_x,
    0.105,
    "● Lower carriage",
    color=lower_color,
    fontsize=22,
)

fig.text(
    panel_b_x + 0.13,
    0.105,
    "● Higher carriage",
    color=higher_color,
    fontsize=22,
)


# ---------- Save and display the combined figure ----------

output_file = (
    folder / "combined_carriage_and_significants_pairs.png"
)

fig.savefig(
    output_file,
    dpi=600,
    facecolor="white",
    bbox_inches=Bbox.from_bounds(
        0,
        0.5,
        fig.get_figwidth(),
        fig.get_figheight() - 0.5,
    ),
)
plt.show()
plt.close(fig)

print(f"\nCombined figure saved to:\n{output_file}")