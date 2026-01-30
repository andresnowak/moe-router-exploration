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
        bin_edges: Probability bin edges
    """

    # Count values in each bin
    counts, _ = np.histogram(global_probs.numpy(), bins=bin_edges)
    fraction = counts / counts.sum()  # Normalize to get probabilities

    # Create labels for each bin
    labels = [f"{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}" for i in range(len(bin_edges)-1)]

    # Use a colormap for the bins
    num_bins = len(bin_edges) - 1
    colors = plt.cm.tab20(np.linspace(0, 1, num_bins))

    # Create bar chart with equal width bars and different colors
    plt.figure(figsize=(14, 6))
    x_pos = np.arange(len(labels))
    plt.bar(x_pos, fraction, color=colors, edgecolor='black')
    plt.title("Router score Distribution across all Layers, Experts, and Subjects (When expert is chosen in Top-K)")
    plt.xlabel("Routing score Range")
    plt.ylabel("Fraction")
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def plot_per_expert_router_distribution(per_expert_probs: List[List[torch.Tensor]], layer: int, bin_edges: List[float] = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    """
    Plot stacked bar chart showing router probability distribution per expert.
    Each expert has one bar with colored segments representing different probability bins.

    Args:
        per_expert_probs: List of lists of tensors, one per layer per expert
        layer: Layer index to plot
        bin_edges: Probability bin edges used for binning
    """
    # Create labels for each bin
    labels = [f"{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}" for i in range(len(bin_edges)-1)]

    num_experts = len(per_expert_probs[layer])
    num_bins = len(bin_edges) - 1

    # Compute bin fractions for each expert
    data = np.zeros((num_experts, num_bins))

    for idx, probs in enumerate(per_expert_probs[layer]):
        # Count values in each bin
        counts, _ = np.histogram(probs.numpy(), bins=bin_edges)
        total = counts.sum()
        fraction = counts / total if total > 0 else counts
        data[idx] = fraction

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(16, 8))

    x_pos = np.arange(num_experts)
    bottom = np.zeros(num_experts)

    # Use a colormap for the bins
    colors = plt.cm.tab20(np.linspace(0, 1, num_bins))

    # Stack each bin on top of the previous one
    for bin_idx in range(num_bins):
        ax.bar(
            x_pos,
            data[:, bin_idx],
            bottom=bottom,
            label=labels[bin_idx],
            color=colors[bin_idx],
            edgecolor='white',
            linewidth=0.5,
            width=0.8
        )
        bottom += data[:, bin_idx]

    ax.set_xlabel("Expert", fontsize=13)
    ax.set_ylabel("Fraction of Router Scores", fontsize=13)
    ax.set_title(f"Router Score Distribution per Expert (when expert is in the top-k) - Layer {layer}", fontsize=15)

    # Handle tick labels based on number of experts
    if num_experts > 32:
        # Show every 4th tick label for many experts
        tick_step = 4
        ax.set_xticks(x_pos[::tick_step])
        ax.set_xticklabels([f"E{i}" for i in range(0, num_experts, tick_step)], fontsize=9)
    else:
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"E{i}" for i in range(num_experts)], fontsize=10)

    ax.legend(title="Routing Score Range", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def plot_per_layer_router_distribution(per_layer_probs: List[torch.Tensor], bin_edges: List[float] = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    """
    Plot stacked bar chart showing router probability distribution per layer.
    Each layer has one bar with colored segments representing different probability bins.

    Args:
        per_layer_probs: List of tensors, one per layer
        bin_edges: Probability bin edges used for binning
    """
    # Create labels for each bin
    labels = [f"{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}" for i in range(len(bin_edges)-1)]

    num_layers = len(per_layer_probs)
    num_bins = len(bin_edges) - 1

    # Compute bin fractions for each layer
    data = np.zeros((num_layers, num_bins))

    for layer, probs in enumerate(per_layer_probs):
        # Count values in each bin
        counts, _ = np.histogram(probs.numpy(), bins=bin_edges)
        total = counts.sum()
        fraction = counts / total if total > 0 else counts
        data[layer] = fraction

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(16, 8))

    x_pos = np.arange(num_layers)
    bottom = np.zeros(num_layers)

    # Use a colormap for the bins
    colors = plt.cm.tab20(np.linspace(0, 1, num_bins))

    # Stack each bin on top of the previous one
    for bin_idx in range(num_bins):
        ax.bar(
            x_pos,
            data[:, bin_idx],
            bottom=bottom,
            label=labels[bin_idx],
            color=colors[bin_idx],
            edgecolor='white',
            linewidth=0.5,
            width=0.8
        )
        bottom += data[:, bin_idx]

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Fraction of Router Scores", fontsize=13)
    ax.set_title("Router Score Distribution per Layer (aggregated across all experts and subjects)", fontsize=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"L{i}" for i in range(num_layers)], fontsize=10)
    ax.legend(title="Routing Score Range", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def plot_deactivated_experts_by_threshold(
    subject_list: List[dict],
    thresholds: List[float]
):
    """
    Plot line chart showing mean number of experts (out of top_k) that would be deactivated
    per token per layer, for multiple probability thresholds, with std as shaded area.

    Args:
        subject_list: List of dicts containing router distribution data with 'distributions', 'num_layers', 'num_experts', 'top_k'
        thresholds: List of probability thresholds below which experts are deactivated (can also pass a single float)
    """
    if not subject_list:
        print("No data provided")
        return

    # Handle single threshold
    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds]

    num_layers = subject_list[0]['num_layers']
    top_k = subject_list[0]['top_k']

    # Build token occurrences once (independent of threshold)
    layer_token_occurrences = []

    for layer in range(num_layers):
        # For each token occurrence, we need all K experts with their probs
        token_occurrences = {}  # (token_id, occurrence_idx) -> {rank: prob}

        # Iterate over all subjects
        for subject in subject_list:
            distributions = subject['distributions']
            num_experts = subject['num_experts']

            for expert_idx in range(num_experts):
                expert_data = distributions[layer][expert_idx]
                if expert_data is None or not isinstance(expert_data, dict):
                    continue

                token_ids = expert_data.get('token_ids', torch.tensor([]))
                probs = expert_data.get('probs', torch.tensor([]))
                ranks = expert_data.get('ranks', torch.tensor([]))

                if len(token_ids) == 0:
                    continue

                # Reset occurrence counter for this expert's pass
                local_token_occurrence = {}

                # Each entry corresponds to one instance where this expert was selected
                for tok_id, prob, rank in zip(token_ids.tolist(), probs.tolist(), ranks.tolist()):
                    # Track how many times we've seen this token in THIS expert's data
                    if tok_id not in local_token_occurrence:
                        local_token_occurrence[tok_id] = 0
                    else:
                        local_token_occurrence[tok_id] += 1

                    occ_idx = local_token_occurrence[tok_id]
                    key = (tok_id, occ_idx)

                    if key not in token_occurrences:
                        token_occurrences[key] = {}

                    # Store this rank's probability
                    token_occurrences[key][int(rank)] = prob

        layer_token_occurrences.append(token_occurrences)

    # Compute stats for each threshold
    all_threshold_stats = {}

    for threshold in thresholds:
        layer_stats = []

        for layer in range(num_layers):
            token_occurrences = layer_token_occurrences[layer]

            # Count deactivations for each token occurrence in this layer
            all_deactivation_counts = []

            for (tok_id, occ_idx), rank_probs in token_occurrences.items():
                # Count how many ranks have prob below threshold
                deactivated = sum(1 for rank, prob in rank_probs.items() if prob < threshold)
                all_deactivation_counts.append(deactivated)

            if len(all_deactivation_counts) > 0:
                deactivation_counts = np.array(all_deactivation_counts)
                layer_stats.append({
                    'mean': deactivation_counts.mean(),
                    'std': deactivation_counts.std(),
                    'min': deactivation_counts.min(),
                    'max': deactivation_counts.max(),
                    'num_tokens': len(deactivation_counts)
                })
            else:
                layer_stats.append({'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'num_tokens': 0})

        all_threshold_stats[threshold] = layer_stats

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))

    x_pos = np.arange(num_layers)
    colors = plt.cm.tab10(np.linspace(0, 1, len(thresholds)))

    for idx, threshold in enumerate(thresholds):
        layer_stats = all_threshold_stats[threshold]
        means = np.array([stats['mean'] for stats in layer_stats])
        stds = np.array([stats['std'] for stats in layer_stats])

        # Plot line with shaded std
        ax.plot(x_pos, means, label=f'Threshold {threshold:.3f}',
                linewidth=2, marker='o', markersize=4, color=colors[idx])
        ax.fill_between(x_pos, means - stds, means + stds, alpha=0.2, color=colors[idx])

        # Add lines for std boundaries
        ax.plot(x_pos, means - stds, color=colors[idx], linewidth=0.8, linestyle='--', alpha=0.6)
        ax.plot(x_pos, means + stds, color=colors[idx], linewidth=0.8, linestyle='--', alpha=0.6)

    ax.set_xlabel('Layer', fontsize=13)
    ax.set_ylabel('Mean Number of Experts Deactivated (out of K)', fontsize=13)
    ax.set_title('Mean Deactivated TopK Experts per Token by Layer', fontsize=15)

    if num_layers > 40:
        tick_step = 4
        ax.set_xticks(x_pos[::tick_step])
        ax.set_xticklabels([f'L{i}' for i in range(0, num_layers, tick_step)], fontsize=9)
    else:
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'L{i}' for i in range(num_layers)], fontsize=10)

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, top_k)

    plt.tight_layout()
    plt.show()

    return all_threshold_stats


# For now I think this one is not good, as it takes to much time to compute and plot
def plot_average_experts_per_range_by_layer(
    layer_results: List[dict],
    bin_edges: List[float] = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
):
    """
    Plot stacked bar chart showing average number of experts per token in each probability range, per layer.
    Each layer has one stacked bar with different colored segments for each probability bin.

    Args:
        layer_results: List of dicts from get_average_experts_per_token_by_layer
        bin_edges: Probability bin edges used for binning
    """
    num_layers = len(layer_results)
    num_bins = len(bin_edges) - 1

    # Create labels for each bin
    labels = [f"{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}" for i in range(num_bins)]

    # Process each layer to compute average experts per token in each bin
    data = np.zeros((num_layers, num_bins))

    for layer_idx, layer_result in enumerate(layer_results):
        token_probs = layer_result['token_probs']

        if not token_probs:
            continue

        # For each token, count how many experts fall into each bin
        token_bin_counts = []

        for tok_id, probs in token_probs.items():
            bin_counts = np.zeros(num_bins)

            # Count probabilities in each bin for this token
            for prob in probs:
                bin_idx = np.searchsorted(bin_edges[1:], prob, side='right')
                bin_idx = min(bin_idx, num_bins - 1)
                bin_counts[bin_idx] += 1

            token_bin_counts.append(bin_counts)

        # Average across all tokens in this layer
        if token_bin_counts:
            data[layer_idx] = np.mean(token_bin_counts, axis=0)

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(16, 8))

    x_pos = np.arange(num_layers)
    bottom = np.zeros(num_layers)

    # Use a colormap for the bins
    colors = plt.cm.tab20(np.linspace(0, 1, num_bins))

    # Stack each bin on top of the previous one
    for bin_idx in range(num_bins):
        ax.bar(
            x_pos,
            data[:, bin_idx],
            bottom=bottom,
            label=labels[bin_idx],
            color=colors[bin_idx],
            edgecolor='white',
            linewidth=0.5
        )
        bottom += data[:, bin_idx]

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Average Number of Experts per Token", fontsize=13)
    ax.set_title("Average Number of Experts per Token in Each Probability Range by Layer", fontsize=15)

    if num_layers > 40:
        # Show every 4th tick label for many layers
        tick_step = 4
        ax.set_xticks(x_pos[::tick_step])
        ax.set_xticklabels([f"L{i}" for i in range(0, num_layers, tick_step)], fontsize=9)
    else:
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"L{i}" for i in range(num_layers)], fontsize=10)

    ax.legend(title="Routing Score Range", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()