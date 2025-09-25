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
