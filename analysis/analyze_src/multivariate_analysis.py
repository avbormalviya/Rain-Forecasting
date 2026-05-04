from abc import ABC, abstractmethod

from typing import Callable, Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Dictionary to store registered plots
PLOTS: Dict[str, Callable] = {}


def register_plot(name: str) -> Callable:
    """
    Decorator to register a plot function.
    """
    def decorator(func) -> Callable:
        PLOTS[name] = func
        return func
    return decorator


@register_plot("correlation_matrix")
def plot_correlation_matrix(df: pd.DataFrame, features: list[str] = None) -> None:
    """
    Plots a correlation matrix for the given features.
    """
    features = features or df.select_dtypes(include='number').columns.tolist()

    sns.heatmap(df[features].corr(), annot=True, cmap="Blues")
    plt.title("Correlation Matrix")
    plt.show()


@register_plot("pivot_heatmap")
def plot_pivot_heatmap(df: pd.DataFrame, index: str, columns: str, values: str, bins: int = 10) -> None:
    """
    Bin the columns variable before pivoting to keep the heatmap readable.
    """
    df = df.copy()
    df[f"{columns}_binned"] = pd.cut(df[columns], bins=bins)
    pivot = df.groupby([index, f"{columns}_binned"])[values].mean().unstack()

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, cmap="Blues", ax=ax)
    ax.set_title(f"Pivot Heatmap of {values} by {index} and {columns}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


@register_plot("boxplot")
def plot_boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str = None,
    top_n: int = None,
) -> None:
    if top_n is not None and top_n > 0:
        top_values = (
            df.groupby(x)[y].mean()
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )
        df = df[df[x].isin(top_values)]

    n_categories = df[x].nunique()
    fig_width = max(10, n_categories * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(f"Boxplot of {y} by {x}")

    # Move legend outside if it's going to be large
    if hue is not None and df[hue].nunique() > 10:
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=7,
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


@register_plot("state_month_heatmap")
def plot_state_month_heatmap(df: pd.DataFrame, states: list[str] = None, top_n: int = 15) -> None:
    """
    Better alternative to boxplot-with-continuous-hue for state x month analysis.
    """
    if states:
        df = df[df["state_name"].isin(states)]
    else:
        top_states = (
            df.groupby("state_name")["actual"].mean()
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )
        df = df[df["state_name"].isin(top_states)]

    pivot = df.groupby(["state_name", "month"])["actual"].mean().unstack()

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, cmap="Blues", annot=True, fmt=".0f", ax=ax)
    ax.set_title("Mean Actual Rainfall by State and Month")
    plt.tight_layout()
    plt.show()



# Run a registered plot
def multivariate_analysis(name, **kwargs):
    if name not in PLOTS:
        raise ValueError(f"Invalid plot name: {name}")
    PLOTS[name](**kwargs)