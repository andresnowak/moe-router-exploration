import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px


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
    )
    fig.update_traces(fill="toself", opacity=0.4)
    fig.update_layout(title="Expert Usage by Corpus")
    fig.show()


def plot_bar(df: pd.DataFrame):
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
    ax.set_title("Expert Usage by Corpus")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Value")
    # Center the x-tick labels under the group
    ax.set_xticks(x + total_width/2 - bar_width/2)
    ax.set_xticklabels(experts, rotation=45, ha="right")
    ax.legend(title="Source")
    plt.tight_layout()
    plt.show()


def plot_box(df):
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

    ax.set_title("Expert Score Distributions Across Subjects")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()