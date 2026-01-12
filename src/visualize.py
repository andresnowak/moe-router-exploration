import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from typing import List


def plot_radar(df: pd.DataFrame, title="Radar Chart"):
    # example df
    # df.index = ['E_0','E_1',...,'E_29'], df.columns = ['arxiv','book',...]
    # df.loc['E_0','arxiv'] = some value

    labels = df.index.tolist()
    n = len(labels)

    # compute equally‐spaced angles in [0, 2pi)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for col in df.columns:
        values = df[col].tolist()
        values += values[:1]
        ax.plot(angles, values, label=col)
        ax.fill(angles, values, alpha=0.2)

    # fix the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # optionally tweak the radial limits
    ax.set_rlabel_position(180 / n)
    ax.set_ylim(df.min().min(), df.max().max())

    plt.title(title)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.show()


def plot_radar_plotly(df: pd.DataFrame):
    # melt the DataFrame to long form
    df_long = df.reset_index().melt(
        id_vars="index", var_name="source", value_name="value"
    )
    df_long.rename(columns={"index": "expert"}, inplace=True)

    fig = px.line_polar(
        df_long,
        r="value",
        theta="expert",
        color="source",
        line_close=True,
        template="plotly_dark",
        render_mode="svg"
    )
    fig.update_traces(fill="toself", opacity=0.4)
    fig.update_layout(title="Expert Usage by Corpus")
    fig.show()


def plot_bar(df: pd.DataFrame, title: str = "plot bar"):
    """
    Given a DataFrame whose index are experts and whose columns are sources,
    draws a grouped bar chart of the values.
    """
    experts = df.index.astype(str)
    sources = df.columns.astype(str)
    values = df.values

    # Number of experts and sources
    n_experts = len(experts)
    n_sources = len(sources)

    # X locations for the groups
    x = np.arange(n_experts)

    # Total width (you can tweak)
    total_width = 0.8
    # Width of each bar
    bar_width = total_width / n_sources

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot one bar per source
    for i, source in enumerate(sources):
        ax.bar(
            x + i * bar_width,
            values[:, i],
            width=bar_width,
            label=source,
            alpha=0.8
        )

    # Labels, ticks, legend
    ax.set_title(title)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Value")
    # Center the x-tick labels under the group
    ax.set_xticks(x + total_width/2 - bar_width/2)
    ax.set_xticklabels(experts, rotation=45, ha="right")
    ax.legend(title="Source")
    plt.tight_layout()
    plt.show()


def plot_box(df, title: str = "plot box"):
    """
    Given a DataFrame whose index are experts and columns are subjects,
    draws one boxplot per expert showing their distribution across subjects.
    """
    experts = df.index.astype(str)
    # For each expert (row), grab the array of subject‐scores
    data = [df.loc[e].values for e in df.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        data,
        labels=experts,
        showmeans=True,      # optionally show the mean point
        meanline=True,        # draw the mean as a line
        meanprops=dict(       # style for that mean‐line
            linestyle=':',
            color='firebrick',
        )
    )

    ax.set_title(title)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_router_distribution(global_probs: torch.Tensor, bin_edges: List[float] = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    """
    Plot histogram of router probability distribution with custom bins.

    Args:
        global_probs: Tensor of router probabilities
    """

    # Count values in each bin
    counts, _ = np.histogram(global_probs.numpy(), bins=bin_edges)
    fraction = counts / counts.sum()  # Normalize to get probabilities

    # Create labels for each bin
    labels = [f"{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}" for i in range(len(bin_edges)-1)]

    # Create bar chart with equal width bars
    plt.figure(figsize=(14, 6))
    x_pos = np.arange(len(labels))
    plt.bar(x_pos, fraction, alpha=0.7, edgecolor='black')
    plt.title("Router score Distribution across all Layers, Experts, and Subjects (When expert is chosen in Top-K)")
    plt.xlabel("Routing score Range")
    plt.ylabel("Fraction")
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def plot_per_expert_router_distribution(per_expert_probs: List[List[torch.Tensor]], layer: int, ncols: int = 6, bin_edges: List[float] = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    """
    Plot grid of histograms showing router probability distribution per expert.

    Args:
        per_expert_probs: List of tensors, one per expert
        ncols: Number of columns in subplot grid
    """
    # Create labels for each bin
    labels = [f"{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}" for i in range(len(bin_edges)-1)]

    num_experts = len(per_expert_probs[layer])
    nrows = (num_experts + ncols - 1) // ncols  # Ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten() if num_experts > 1 else [axes]

    for idx, probs in enumerate(per_expert_probs[layer]):
        # Count values in each bin
        counts, _ = np.histogram(probs.numpy(), bins=bin_edges)
        fraction = counts / counts.sum()  # Normalize to get probabilities

        # Create bar chart with equal width bars
        x_pos = np.arange(len(labels))

        axes[idx].bar(x_pos, fraction, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f"Expert {idx}")
        axes[idx].set_xlabel("Routing score Range")
        axes[idx].set_ylabel("Fraction")
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(labels, rotation=45, ha='right')
        axes[idx].grid(True, alpha=0.3, axis='y')

    # Hide unused subplots
    for idx in range(num_experts, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f"Router score Distribution per Expert (when expert is in the top-k) - Layer {layer}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space at top for suptitle
    plt.show()


def plot_per_layer_router_distribution(per_layer_probs: List[torch.Tensor], ncols: int = 4, bin_edges: List[float] = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    """
    Plot grid of histograms showing router probability distribution per layer.

    Args:
        per_layer_probs: List of tensors, one per layer
        ncols: Number of columns in subplot grid
    """
    # Create labels for each bin
    labels = [f"{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}" for i in range(len(bin_edges)-1)]

    num_layers = len(per_layer_probs)
    nrows = (num_layers + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten() if num_layers > 1 else [axes]

    for layer, probs in enumerate(per_layer_probs):
        # Count values in each bin
        counts, _ = np.histogram(probs.numpy(), bins=bin_edges)
        total = counts.sum()
        fraction = counts / total if total > 0 else counts

        # Create bar chart with equal width bars
        x_pos = np.arange(len(labels))

        axes[layer].bar(x_pos, fraction, alpha=0.7, edgecolor='black')
        axes[layer].set_title(f"Layer {layer}")
        axes[layer].set_xlabel("Routing score Range")
        axes[layer].set_ylabel("Fraction")
        axes[layer].set_xticks(x_pos)
        axes[layer].set_xticklabels(labels, rotation=45, ha='right')
        axes[layer].grid(True, alpha=0.3, axis='y')

    # Hide unused subplots
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')

    fig.suptitle("Router score Distribution per Layer (aggregated across all experts and subjects)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()