from dashboard.utils.config import *
import numpy as np
import pandas as pd
import scipy
import warnings
from functools import wraps
import itertools
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.preprocessing import label_binarize
from datetime import datetime, timedelta
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.pyplot as plt
from matplotlib import animation
import plotly.graph_objects as go
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shpreader
import seaborn as sns
import networkx as nx
import pydot
import streamlit as st
from PIL import Image
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['axes.titlesize'] = 20
sns.set_context("notebook", rc={"axes.labelsize":16,"axes.titlesize":20, "legend.fontsize":16})
plt.style.use('seaborn')

def no_warning(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('ignore')  # turn off filter
        return func(*args, **kwargs)

    return new_func

@no_warning
def plot_density(start_date, end_date, x, y, z=None, regimes=True, **kwargs):
    '''
    Method to plot the density of two variables
    Args:
        start_date: str indicating the start date to consider
        end_date: str indicating the end date to consider
        x: pd.Series containing the elements to plot for the variable x
        y: pd.Series containing the elements to plot for the variable y
        regimes: boolean indicating whether the reference of the regimes must be inserted
    '''

    if 'ax' not in kwargs:
        fig, ax = plt.subplots(figsize=(7, 7))  # subplot_kw={'projection': 'polar'})

    else:
        ax = kwargs['ax']

    if z is None:
        x, y = np.squeeze(x.loc[start_date:end_date]), np.squeeze(y.loc[start_date:end_date])
        sns.kdeplot(x, y, shade=True, ax=ax, cmap='RdBu_r', levels=10, alpha=.7,
                    label=kwargs['label'] if 'label' in kwargs else None)
    else:
        x, y = x[z.index.intersection(x.index)], y[z.index.intersection(y.index)]
        sns.scatterplot(x, y, ax=ax, hue=z, palette='RdBu_r')

    if regimes:
        ax.vlines(0, -3, 3, linewidth=.1, colors='k', zorder=100)
        ax.hlines(0, -3, 3, linewidth=.1, colors='k', zorder=100)
        ax.plot([-3, 3], [-3, 3], linewidth=.1, c='k', zorder=100)
        ax.plot([-3, 3], [3, -3], linewidth=.1, c='k', zorder=100)
        circle = plt.Circle((0, 0), 1, linewidth=.1, edgecolor='k', facecolor=None, fill=False)
        ax.add_patch(circle)
        ax.text(-2.95, 0., "NAO-", size='large', weight='bold')
        ax.text(0., -2.95, "BL-", size='large', weight='bold')
        ax.text(0., 2.85, "BL+", size='large', weight='bold')
        ax.text(2.4, 0., "NAO+", size='large', weight='bold')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

    if 'label' in kwargs:
        ax.legend()

    if 'savefig' in kwargs:
        plt.savefig(kwargs['savefig'])

    if 'ax' not in kwargs:
        plt.show()

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_lines(df, cols, title = None, **kwargs):
    '''
    Method to dynamically reporduce a line plot
    Args:
        df: a pandas DataFrame containing data
        cols: a list of str or list of list corresponding to the lines to be plot. For each sub-list the plot color the areas included between the two lines
        title (optional): the title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    fig, axs = plt.subplots(1, len(cols), figsize=(7.5 * len(cols), 15))
    for i, (ax, col) in enumerate(zip(axs.flat, cols)):
        df[col].plot(kind='line', ax=ax, linewidth=2.1, linestyle='-.')

        if not isinstance(col, str):
            ax.fill_between(df.index.values, df[col[0]], df[col[1]], where=df[col[0]] > df[col[1]],
                            interpolate=True, alpha=.25)
            ax.fill_between(df.index.values, df[col[0]], df[col[1]], where=df[col[0]] < df[col[1]],
                            interpolate=True, alpha=.25)
        ax.legend()

    fig.tight_layout()
    if title:
        plt.savefig(title)
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_bands(df, by, cols, xlabel, ylabel, title = None, **kwargs):
    '''
    Method to plot the confidence intervals
    Args:
        df: a pandas DataFrame containing data
        by: the column name used to group by
        cols: a list of str or a list of list of columns that will be plotted
        xlabel: name of the x variable
        ylabel: name of the y variable
        title (optional): the title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    fig, axs = plt.subplots(1, len(cols), figsize=(7.5 * len(cols), 7),
                            sharey=True if 'sharey' in kwargs else False)
    grouped_df = df.groupby(by).agg({k: ["mean", "std"] for k in df.columns})

    if not hasattr(axs,"flat"):
        axs = np.array([axs])

    for ax, col_ in zip(axs.flat, cols):
        grouped_df.xs('mean', level=1, axis=1)[col_]. \
            plot(kind='line', ax=ax, xlabel=xlabel, ylabel=ylabel, linewidth=2.1, linestyle='-.')
        if isinstance(col_, str):
            col = col_
            ax.fill_between(grouped_df.index.values,
                            grouped_df.xs('mean', level=1, axis=1)[col] - 2 * grouped_df.xs('std', level=1, axis=1)[
                                col],
                            grouped_df.xs('mean', level=1, axis=1)[col] + 2 * grouped_df.xs('std', level=1, axis=1)[
                                col],
                            alpha=.25, label="95% C.I. " + col)
        else:
            for col in col_:
                ax.fill_between(grouped_df.index.values,
                                grouped_df.xs('mean', level=1, axis=1)[col] - 2 *
                                grouped_df.xs('std', level=1, axis=1)[col],
                                grouped_df.xs('mean', level=1, axis=1)[col] + 2 *
                                grouped_df.xs('std', level=1, axis=1)[col],
                                alpha=.25, label="95% C.I. " + col)
        ax.legend()

    fig.tight_layout()
    if title:
        plt.savefig(title)
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_3Dpcs(df, pcs, cols, title = None, **kwargs):
    '''
    Method to plot the densities of variables against the two regimes' Principal Components (PCs)
    Args:
        df: a pandas DataFrame containing data
        pcs: a pandas DataFrame containing the Principal Components
        cols: a list of str containing the names of the columns to be plotted
        title (optional): the title to save the figure
        **kwargs: optionla additional parameters

    Returns:
        A matplotlib figure

    '''
    fig, axs = plt.subplots(1, len(cols), figsize=(len(cols) * 7.5, 7), sharex=True, sharey=True)
    if not hasattr(axs,"flat"):
        axs = np.array([axs])
    for ax, col in zip(axs.flat, cols):
        plot_density(start_date=df.index.values[0], end_date=df.index.values[-1],
                     x=pcs['PC1'], y=pcs['PC2'], z=df[col], ax=ax)

    fig.tight_layout()
    plt.savefig(title)
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_densities_vs_variable(df, x_col, y_cols, title=None, **kwargs):
    '''
    Method to plot the densities of two variables against
    Args:
        df: a pandas DataFrame containing data
        x_col: a str containing the name of the column acting as the x variable
        y_cols: a list of str containing the names of the columns acting as the y variables
        title (optional): the title to save the figure

    Returns:
        A matplotlib figure

    '''
    fig, axs = plt.subplots(len(y_cols), 4, figsize=(30, len(y_cols) * 7.5), sharey='row')

    for i, y_col in enumerate(y_cols):
        for j, (ax, regime) in enumerate(zip(axs.flat[i * 4:(i + 1) * 4], sorted(df['Regime'].unique()))):
            plot_density(df[df['Regime'] == regime].index.values[0],
                         df[df['Regime'] == regime].index.values[-1],
                         df[df['Regime'] == regime][x_col],
                         df[df['Regime'] == regime][y_col], label=regime,
                         ax=ax, regimes=False)

    if title:
        plt.savefig(title)
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def compare_timeseries(df1, df2, cols, labels, title = None, **kwargs):
    '''
    Method to compare timseries coming from two different data sources
    Args:
        df1: a pandas DataFrame containing the first source of data
        df2: a pandas DataFrame containing the second source of data
        cols: list of str containing the columns names to extract from both the data sources
        labels: labels to put in legend
        title (optional): title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    fig, axs = plt.subplots(1, len(cols), figsize=(len(cols) * 7.5, 7))
    if not hasattr(axs,"flat"):
        axs = np.array([axs])
    for i, (ax, col) in enumerate(zip(axs.flat, cols)):
        df1[col].plot(kind='line', label=col + "-" + labels[0], ax=ax, linewidth=2.1, linestyle='-.')
        df2[col].plot(kind='line', label=col + "-" + labels[1], ax=ax, linewidth=2.1, linestyle='-.')
        if not isinstance(col, str):
            ax.fill_between(df1.index, df1[col], df2[col], where=df1[col] > df2[col],
                            interpolate=True, alpha=.25)
            ax.fill_between(df1.index, df1[col], df2[col], where=df1[col] < df2[col],
                            interpolate=True, alpha=.25)
        ax.legend()
    fig.tight_layout()
    if title:
        plt.savefig(title)
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def compare_distributions(df1, df2, cols, labels, title = None, **kwargs):
    '''
    Method to compare distributions coming from two different data sources
    Args:
        df1: a pandas DataFrame containing the first source of data
        df2: a pandas DataFrame containing the second source of data
        cols: list of str containing the columns names to extract from both the data sources
        labels: labels to put in legend
        title (optional): title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    fig, axs = plt.subplots(1, len(cols), figsize=(len(cols) * 7.5, 7))
    if not hasattr(axs,"flat"):
        axs = np.array([axs])
    for i, (ax, col) in enumerate(zip(axs.flat, cols)):
        if isinstance(col, str):
            sns.kdeplot(x=df1[col], label=col + "-" + labels[0], ax=ax,
                        bw_adjust=kwargs['bw_adjust'] if 'bw_adjust' in kwargs else 1)
            sns.kdeplot(x=df2[col], label=col + "-" + labels[1], ax=ax,
                        bw_adjust=kwargs['bw_adjust'] if 'bw_adjust' in kwargs else 1)
        else:
            for col_ in col:
                sns.kdeplot(x=df1[col_], label=col_ + "-" + labels[0], ax=ax,
                            bw_adjust=kwargs['bw_adjust'] if 'bw_adjust' in kwargs else 1)
                sns.kdeplot(x=df2[col_], label=col_ + "-" + labels[1], ax=ax,
                            bw_adjust=kwargs['bw_adjust'] if 'bw_adjust' in kwargs else 1)
        ax.legend()
    fig.tight_layout()
    if title:
        plt.savefig(title)
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def compare_distributions_by_country(df1, df2, cols, labels, title = None, **kwargs):
    '''
    Method to compare distributions coming from two different data sources, discriminating them by country
    Args:
        df1: a pandas DataFrame containing the first source of data
        df2: a pandas DataFrame containing the second source of data
        cols: list of str containing the columns names to extract from both the data sources
        labels: labels to put in legend
        title (optional): title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    countries = df1.reset_index(level=0)['Country'].unique()
    fig, axs = plt.subplots(len(countries), len(cols),
                            figsize=(len(cols) * 7.5, len(countries) * 7.5), sharex='col')
    if not hasattr(axs,"flat"):
        axs = np.array([axs])
    for i, country in enumerate(countries):
        for j, (ax, col) in enumerate(zip(axs.flat[i * len(cols):(i + 1) * len(cols)], cols)):
            if isinstance(col, str):
                sns.kdeplot(x=df1.xs(country, level=0)[col], label=col + "-" + labels[0] + "-" + country,
                            ax=ax, bw_adjust=kwargs['bw_adjust'] if 'bw_adjust' in kwargs else 1)
                sns.kdeplot(x=df2.xs(country, level=0)[col], label=col + "-" + labels[1] + "-" + country,
                            ax=ax, bw_adjust=kwargs['bw_adjust'] if 'bw_adjust' in kwargs else 1)
            else:
                for col_ in col:
                    sns.kdeplot(x=df1.xs(country, level=0)[col_], label=col_ + "-" + labels[0],
                                ax=ax, bw_adjust=kwargs['bw_adjust'] if 'bw_adjust' in kwargs else 1)
                    sns.kdeplot(x=df2.xs(country, level=0)[col_], label=col_ + "-" + labels[1],
                                ax=ax, bw_adjust=kwargs['bw_adjust'] if 'bw_adjust' in kwargs else 1)
            ax.legend()
    fig.tight_layout()
    if title:
        plt.savefig(title)
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_distribution(df, cols, title = None, **kwargs):
    '''
    Method to plot a distribution of some variables
    Args:
        df: a pandas DataFrame containing data
        cols: list of str containing columns names to be plotted
        title (optional): title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    fig, axs = plt.subplots(1, len(cols), figsize=(len(cols) * 7.5, 7))
    if not hasattr(axs,"flat"):
        axs = np.array([axs])
    for i, (ax, col) in enumerate(zip(axs.flat, cols)):
        if isinstance(col, str):
            sns.kdeplot(x=df[col], label=col, ax=ax, bw_adjust=kwargs['bw_adjust'] if 'bw_adjust' in kwargs else 1)
        else:
            for col_ in col:
                sns.kdeplot(x=df[col_], label=col_, ax=ax,
                            bw_adjust=kwargs['bw_adjust'] if 'bw_adjust' in kwargs else 1)
        ax.legend()
    fig.tight_layout()
    plt.savefig(title)
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_distribution_by_regime(df, cols, predictions, model, **kwargs):
    '''
    Metbod to retrieve the statistis of the distribution of some variables weighting them by predicted regimes
    Args:
        df: a pandad DataFrame containing data
        cols: list of str containing the columns names of data to be considered
        predictions: a pandas DataFrame containign historical daily weather regimes predictions of several models
        model: the name of the model whose predictions are considered
        **kwargs: optional additional parameters

    Returns:
        A pandas DataFrame containing the mean and std for each distribution

    '''
    stats = []
    for i, col in enumerate(cols):
        countries = dict()
        for country in df['Country'].unique():
            if col == 'Load':
                variable_df = df.loc[np.logical_and(df['Country'] == country, df.index.map(pd.tseries.offsets.BDay().onOffset)), col]
            else:
                variable_df = df.loc[df['Country'] == country, col]
            variable_df.dropna(inplace = True)
            distributions_regimes = {
                regime: scipy.stats.gaussian_kde(variable_df.loc[predictions.index.intersection(variable_df.index)],
                                                 weights=predictions.loc[predictions.index.intersection(variable_df.index), (model, regime)].values,
                                                 bw_method=.1)
                for regime in sorted(REGIMES)}

            points = np.linspace(variable_df.min(), variable_df.max(), 1000)
            normal = variable_df.mean()
            print(col,country,normal)

            means = {k: sum(points * v.pdf(points)) / sum(v.pdf(points))
                     for k, v in distributions_regimes.items()}

            #anomalies = {k: v - normal for k, v in means.items()}

            stds = {k: np.sqrt(np.sum([(p - v)**2 for p in points])/len(points))
                     for k, v in means.items()}

            statistics = {k: dict(mean = means[k], std = stds[k])
                     for k in distributions_regimes.keys()}

            countries[country] = statistics

        countries = {country: {(outerKey, innerKey): values for outerKey, innerDict in countries[country].items() for innerKey, values in innerDict.items()} for country in countries.keys()}
        stats.append(pd.DataFrame.from_dict(countries, orient = 'index'))

    return stats

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_distribution_by_regime_days(df, cols, predictions, model, title = None, **kwargs):
    '''
    Method to plot the distributions of some variables, divided by business days and days off
    Args:
        df: a pandas DataFrame containing the data
        cols: list of str containing the columns names to be plotted
        predictions: a pandas DataFrame containing the historical daily weather regimes predictions
        model: the name of the model whose predictions are considered
        title (optional): the title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    fig, axs = plt.subplots(len(cols), 2, figsize=(15, len(cols) * 7.5), sharey = 'row', sharex = 'row')
    if len(axs.shape) < 2:
        axs = np.array([axs])
    for i, col in enumerate(cols):
        for j, regime in enumerate(REGIMES):
            bdays = df.loc[df.index.map(pd.tseries.offsets.BDay().onOffset)]
            sns.kdeplot(x=bdays[col], weights=predictions.loc[bdays.index, (model, regime)].values,
                        label=regime, ax=axs[i,0], bw_adjust=kwargs['bw_adjust'] if 'bw_adjust' in kwargs else 1)

            days_off = df.loc[df.index.difference(bdays.index)]
            sns.kdeplot(x=days_off[col], weights=predictions.loc[days_off.index, (model, regime)].values,
                        label=regime, ax=axs[i, 1], bw_adjust=kwargs['bw_adjust'] if 'bw_adjust' in kwargs else 1)
        '''
        for j, regime in enumerate(sorted(df['Regime'].unique())):
            preds = predictions.xs(model, level = 0, axis = 1)
            idx_hard = preds[preds['Prediction'] == regime].index

            sns.kdeplot(x=df.loc[idx_hard.intersection(df.index), col], label = regime + " hard", ax = ax, linestyle='--')
        '''
        for j in range(2):
            axs[i,j].set_xlabel(col, fontsize = 16)
            axs[i,j].set_ylabel("Density", fontsize = 16)
            axs[i,j].legend()
    fig.tight_layout()
    if title:
        plt.savefig(title)
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_distribution_by_regime(df, cols, predictions, model, title = None, **kwargs):
    '''
    Method to plot the distributions of some variables
    Args:
        df: a pandas DataFrame containing the data
        cols: list of str containing the columns names to be plotted
        predictions: a pandas DataFrame containing the historical daily weather regimes predictions
        model: the name of the model whose predictions are considered
        title (optional): the title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    fig, axs = plt.subplots(1, len(cols), figsize=(len(cols) * 7.5, 7))
    if not hasattr(axs,"flat"):
        axs = np.array([axs])
    for i, (col, ax) in enumerate(zip(cols, axs.flat)):
        for j, regime in enumerate(REGIMES):
            sns.kdeplot(x=df[col], weights=predictions.loc[df.index, (model, regime)].values,
                        label=regime, ax=ax, bw_adjust=kwargs['bw_adjust'] if 'bw_adjust' in kwargs else 1)
        '''
        for j, regime in enumerate(sorted(df['Regime'].unique())):
            preds = predictions.xs(model, level = 0, axis = 1)
            idx_hard = preds[preds['Prediction'] == regime].index

            sns.kdeplot(x=df.loc[idx_hard.intersection(df.index), col], label = regime + " hard", ax = ax, linestyle='--')
        '''

        ax.set_xlabel(col, fontsize = 16)
        ax.set_ylabel("Density", fontsize = 16)
        ax.axvline(df[col].mean(), linestyle='--', color='k', label = 'Normal')
        ax.legend()

    fig.tight_layout()
    if title:
        plt.savefig(title)
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_distribution_by_regime_monthly(df, cols, predictions, model, title = None, **kwargs):
    '''
    Method to plot the distributions of some variables grouped by the month
    Args:
        df: a pandas DataFrame containing the data
        cols: list of str containing the columns names to be plotted
        predictions: a pandas DataFrame containing the historical daily weather regimes predictions
        model: the name of the model whose predictions are considered
        title (optional): the title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    monthly_stats = df.copy()
    monthly_stats['Month'] = monthly_stats.index.month
    unique_months = monthly_stats['Month'].unique()
    fig, axs = plt.subplots(len(cols), len(unique_months),
                            figsize=(len(unique_months) * 7.5, len(cols) * 7.5), sharey='row')
    if not hasattr(axs,"flat"):
        axs = np.array([axs])
    for i, col in enumerate(cols):
        for j, (ax, month) in enumerate(
                zip(axs.flat[i * len(unique_months):(i + 1) * len(unique_months)], unique_months)):
            for k, regime in enumerate(sorted(monthly_stats['Regime'].unique())):
                sns.kdeplot(x=monthly_stats[monthly_stats['Month'] == month][col],
                            weights=predictions.loc[df.index[df.index.month == month], (model, regime)].values, label=regime, ax=ax,
                            bw_adjust=kwargs['bw_adjust'] if 'bw_adjust' in kwargs else 1)
            ax.set_xlabel(col, fontsize=16)
            ax.set_ylabel("Density", fontsize=16)
            ax.set_title(datetime.strptime(str(month), "%m").strftime("%B"), fontsize = 20)
            ax.legend()
    fig.tight_layout()
    if title:
        plt.savefig(title)
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_extreme_events(df, cols, predictions, model, title = None,  **kwargs):
    '''
    Method to plot a pie chart of the extreme events of some variables, divided by weather regimes
    Args:
        df: a pandas DataFrame containing the data
        cols: list of str containing the columns names to be plotted
        predictions: a pandas DataFrame containing the historical daily weather regimes predictions
        model: the name of the model whose predictions are considered
        title (optional): the title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    fig, axs = plt.subplots(1, len(cols), figsize=(len(cols) * 7.5, 7))
    if not hasattr(axs, "flat"):
        axs = np.array([axs])
    for i, (col, ax) in enumerate(zip(cols, axs.flat)):
        extreme_events = df[col].dropna()[df[col].dropna() >= np.percentile(df[col].dropna(), 98)]
        weights = predictions.xs(model, level=0, axis=1).loc[extreme_events.index.intersection(predictions.index)].mean()
        weights = weights / weights.sum()
        weights.sort_index(inplace = True)
        ax.pie(weights * 100, labels = weights.index, autopct = '%1.1f%%', startangle = 0, textprops={'fontsize': 16})
        ax.axis('equal')
        ax.set_title(col, fontsize = 20)
    fig.tight_layout()
    if title:
        plt.savefig(title)
    return fig

@st.cache
def get_extreme_events(df, cols, predictions, model):
    '''
    Method to get the extreme events of some variables, divided by weather regimes
    Args:
        df: a pandas DataFrame containing the data
        cols: list of str containing the columns names to be plotted
        predictions: a pandas DataFrame containing the historical daily weather regimes predictions
        model: the name of the model whose predictions are considered

    Returns:
        A list of pandas DataFrames containing the extreme events of each of the variables specified in cols

        '''
    tables = []
    for col in cols:
        tmp_df = dict()
        for country in df['Country'].unique():
            extreme_events = df[df['Country'] == country][col].dropna()
            extreme_events = extreme_events[extreme_events >= np.percentile(extreme_events, 98)]
            weights = predictions.xs(model, level=0, axis=1).loc[extreme_events.index.intersection(predictions.index)].mean()
            weights = weights / weights.sum()
            weights.sort_index(inplace = True)
            tmp_df[country] = weights
        tables.append(pd.DataFrame.from_dict(tmp_df, orient = "index"))
    return tables

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_scatter(df, x_col, y_col, fit = True, title = None, **kwargs):
    '''
    Method to plot a scatter plot with the fitted regression line between two variables
    Args:
        df: a pandas DataFrame containing data
        x_col: str indicating the column name of the x variable
        y_col: str indicating the column name of the y variable
        fit: boolean indicating whether to fit or not the regression line
        title (optional): the title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    if fit:
        sns.regplot(x = x_col, y = y_col, data = df, scatter = False)#scatter_kws=dict(zorder =-1), ci = 100)
    ax.scatter(df[x_col], df[y_col], c = df[x_col], cmap='RdBu_r', s=40, edgecolor = 'k', zorder = 100)
    fig.tight_layout()
    if title is not None:
        plt.savefig(title)
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_scatter_by_regime(df, x_col, y_col, regimes, title = None, **kwargs):
    '''
    Method to plot a scatter plot between two variables, under weather regimes
    Args:
        df: a pandas DataFrame containing data
        x_col: str indicating the column name of the x variable
        y_col: str indicating the column name of the y variable
        regimes: list of str containing the name of the regimes
        title (optional): the title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for i, regime in enumerate(sorted(regimes)):
        sns.regplot(x=df[df['Regime'] == regime][x_col], y=df[df['Regime'] == regime][y_col], label=regime,
                    scatter_kws=dict(s=40, edgecolor = 'k'))
    ax.legend()
    fig.tight_layout()
    if title:
        plt.savefig(title)
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def boxplot(df, grouping_cols, cols, title = None, **kwargs):
    '''
    Method to plot the boxplots of some variables
    Args:
        df: a pandas DataFrame containing data
        grouping_cols: list of str containing columns names which will be used to group by the df
        cols: list of str of columns names that will be plotted
        title (optional): title to save the figure
        **kwargs: optonal additional parameters

    Returns:
        A matplotlib figure

    '''
    fig, axs = plt.subplots(1, len(cols), figsize=(len(cols) * 7.5, 7))
    if not hasattr(axs,"flat"):
        axs = np.array([axs])
    for ax, col in zip(axs.flat, cols):
        groups = grouping_cols + [col]
        df[groups].boxplot(by=grouping_cols, meanline=True, showmeans=True, showcaps=True,
                                   showbox=True, showfliers=False, ax=ax, grid=False,
                                   boxprops=dict(color='k'),
                                   capprops=dict(color='k'),
                                   whiskerprops=dict(color='k'),
                                   meanprops=dict(color='r'),
                                   medianprops=dict(color='b'))
        ax.set_ylabel(col)
        ax.set_xlabel('Regime')
    fig.suptitle('')
    if title:
        plt.savefig(title)
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def boxplot_monthly(df, grouping_cols, cols, title = None, **kwargs):
    '''
    Method to plot the boxplots of some variables, under different months
    Args:
        df: a pandas DataFrame containing data
        grouping_cols: list of str containing columns names which will be used to group by the df
        cols: list of str of columns names that will be plotted
        title (optional): title to save the figure
        **kwargs: optonal additional parameters

    Returns:
        A matplotlib figure

    '''
    fig, axs = plt.subplots(len(cols), 1, figsize=(22.5, len(cols) * 7.5))
    if not hasattr(axs,"flat"):
        axs = np.array([axs])
    monthly_stats = df.copy()
    monthly_stats['Month'] = monthly_stats.index.month % 12
    for ax, col in zip(axs.flat, cols):
        groups = grouping_cols + [col]
        monthly_stats[groups].boxplot(by=grouping_cols,meanline=True, showmeans=True, showcaps=True,
                                        showbox=True, showfliers=False, ax=ax, grid=False,
                                        boxprops=dict(color='k'),
                                        capprops=dict(color='k'),
                                        whiskerprops=dict(color='k'),
                                        meanprops=dict(color='r'),
                                        medianprops=dict(color='b'))
        ax.axvline(x=3.5, color='k')
        ax.axvline(x=6.5, color='k')
        ax.axvline(x=9.5, color='k')
        ax.set_ylabel(col, fontsize = 16)
        ax.set_xticklabels([f"{regime}, {month}" for regime, month in itertools.product(sorted(monthly_stats[groups]["Regime"].unique()),['Dec','Jan','Feb'])])
        ax.set_xlabel(','.join(grouping_cols), fontsize = 16)
    fig.suptitle('')
    plt.savefig(title)
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_bars(df, cols, title = None, **kwargs):
    '''
    Method to plot barplots of some variables
    Args:
        df: a pandas DataFrame containing data
        cols: list of str of columns names that will be plotted
        title (optional): title to save the figure
        **kwargs: optonal additional parameters

    Returns:
        A matplotlib figure

        '''
    fig, axs = plt.subplots(1, 4, figsize=(30, 7), sharey=True)  # , sharex = True)
    for ax, regime in zip(axs.flat, df.index.get_level_values(1).unique()):
        df[cols].xs(regime, level=1). \
            plot(kind='bar', ax=ax, xlabel='Month', ylabel='Load Factor', title=regime)
    if title:
        plt.savefig(title)
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_maps(df, country, cols, date, title = None, **kwargs):
    '''
    Method to plot geographic maps of some variables
    Args:
        df: a pandas DataFrame containing data
        country: a str indicating the country whose data is selected
        cols: list of str of columns names that will be plotted
        title (optional): title to save the figure
        **kwargs: optonal additional parameters

    Returns:
        A matplotlib figure

    '''
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection = projection))
    fig = plt.figure(figsize=(7, len(cols)*7.5))
    axs = AxesGrid(fig, 111, axes_class = axes_class, nrows_ncols=(len(cols), 1), axes_pad =1, label_mode ='')
    shps = {"FR": "FRA", "UK": "GBR", "BE": "BEL", "NE": "NLD", "IT": "ITA", "GE": "DEU", "ES": "ESP", "EU-7":"EU7"}
    fname = 'P:/CH/Weather Data/SHP_FILES/gadm36_{}_shp/gadm36_{}_0.shp'.format(shps[country], shps[country])
    shapes = list(shpreader.Reader(fname).geometries())
    df = df.loc[date]
    for ax, col in (zip(axs, cols)):
        color = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin = df[col].min(), vmax = df[col].max()),
                                      cmap = "RdBu_r")
        bounds = {"x0": [], "y0": [], "x1": [], "y1": []}
        if country == "EU-7":
            for cnt in df['Country'].unique():
                fname_country = 'P:/CH/Weather Data/SHP_FILES/gadm36_{}_shp/gadm36_{}_0.shp'.format(shps[cnt], shps[cnt])
                shapes_cnt = list(shpreader.Reader(fname_country).geometries())
                ax.add_geometries(shapes_cnt, ccrs.PlateCarree(),
                                  edgecolor='k', facecolor= np.squeeze(color.to_rgba(df[df['Country'] == cnt][col])), alpha=0.5)
                x0, y0, x1, y1 = shapes_cnt[0].bounds
                bounds["x0"].append(x0)
                bounds["y0"].append(y0)
                bounds["x1"].append(x1)
                bounds["y1"].append(y1)

                text = str(round(df.loc[df['Country'] == cnt,col].values[0], 2))
                ax.text((x0+x1)/2, (y0+y1)/2, text,
                        fontsize = 'large', fontweight = 'black')
            ax.set_extent([min(bounds["x0"])-0.1,max(bounds["x1"])+0.1,min(bounds["y0"])-0.1,max(bounds["y1"])+0.1])
            ax.set_title(col)
        else:
            ax.add_geometries(shapes, ccrs.PlateCarree(),
                              edgecolor='k', facecolor=np.squeeze(color.to_rgba(df[col])),
                              alpha=0.5)
            x0, y0, x1, y1 = shapes[0].bounds
            bounds["x0"].append(x0)
            bounds["y0"].append(y0)
            bounds["x1"].append(x1)
            bounds["y1"].append(y1)

            text = str(round(df[col], 2))
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, text,
                    fontsize='xx-large', fontweight='black')
            ax.set_extent(
                [min(bounds["x0"]) - 0.1, max(bounds["x1"]) + 0.1, min(bounds["y0"]) - 0.1, max(bounds["y1"]) + 0.1])
            ax.set_title(col)
    plt.savefig(title)
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_maps_by_regime(df, country, cols, title = None, **kwargs):
    '''
    Method to plot the geographic maps of some variables, under weather regimes
    Args:
        df: a pandas DataFrame containing data
        country: a str indicating the country whose data is selected
        cols: list of str of columns names that will be plotted
        title (optional): title to save the figure
        **kwargs: optonal additional parameters

    Returns:
        A matplotlib figure

    '''
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection = projection))
    regimes = df['Regime'].unique()
    fig = plt.figure(figsize=(7.5*len(regimes), len(cols)*7.5))
    axs = AxesGrid(fig, 111, axes_class = axes_class, nrows_ncols=(len(cols), len(regimes)), axes_pad =1, label_mode ='')
    shps = {"FR": "FRA", "UK": "GBR", "BE": "BEL", "NE": "NLD", "IT": "ITA", "GE": "DEU", "ES": "ESP", "EU-7":"EU7"}
    fname = 'P:/CH/Weather Data/SHP_FILES/gadm36_{}_shp/gadm36_{}_0.shp'.format(shps[country], shps[country])
    shapes = list(shpreader.Reader(fname).geometries())
    if country == 'EU-7':
        df = df.groupby(['Regime','Country']).mean()
    else:
        df = df.groupby('Regime').mean()

    for i,col in enumerate(cols):
        if country == 'EU-7':
            normal = df.groupby('Country')[col].mean()
        else:
            normal = df[col].mean()
        print(col, normal)
        df[col] = (df[col] - normal) * 100 / normal
        color = mpl.cm.ScalarMappable(
            norm=mpl.colors.TwoSlopeNorm(vmin=df[col].values.min(), vcenter = 0., vmax=df[col].values.max()),
            cmap="RdBu_r")
        for ax, regime in (zip(axs[i*len(regimes): (i+1)*len(regimes)], sorted(regimes))):
            bounds = {"x0": [], "y0": [], "x1": [], "y1": []}
            if country == "EU-7":
                for cnt in df.index.get_level_values(1).unique():
                    #color = mpl.cm.ScalarMappable(
                    #    norm=mpl.colors.Normalize(vmin=df[df['Country'] == cnt][col].values.min(),
                    #                              vmax=df[df['Country'] == cnt][col].values.max()), cmap="RdBu_r")
                    fname_country = 'P:/CH/Weather Data/SHP_FILES/gadm36_{}_shp/gadm36_{}_0.shp'.format(shps[cnt], shps[cnt])
                    shapes_cnt = list(shpreader.Reader(fname_country).geometries())
                    ax.add_geometries(shapes_cnt, ccrs.PlateCarree(),
                                      edgecolor='k', facecolor= np.squeeze(color.to_rgba(df.xs(cnt, level = 1).loc[regime,col])), alpha=0.5)
                    x0, y0, x1, y1 = shapes_cnt[0].bounds
                    bounds["x0"].append(x0)
                    bounds["y0"].append(y0)
                    bounds["x1"].append(x1)
                    bounds["y1"].append(y1)

                    text = str(round(df.xs(cnt, level = 1).loc[regime,col], 1)) + "%"
                    ax.text((x0+x1)/2, (y0+y1)/2, text,
                            fontsize = 'large', fontweight = 'black')
                ax.set_extent([min(bounds["x0"])-0.1,max(bounds["x1"])+0.1,min(bounds["y0"])-0.1,max(bounds["y1"])+0.1])
                ax.set_title(regime, fontsize = 20)
            else:
                ax.add_geometries(shapes, ccrs.PlateCarree(),
                                  edgecolor='k', facecolor=np.squeeze(color.to_rgba(df.loc[regime, col])),
                                  alpha=0.5)
                x0, y0, x1, y1 = shapes[0].bounds
                bounds["x0"].append(x0)
                bounds["y0"].append(y0)
                bounds["x1"].append(x1)
                bounds["y1"].append(y1)

                text = str(round(df.loc[regime, col], 1)) + "%"
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, text,
                        fontsize='xx-large', fontweight='black')
                ax.set_extent(
                    [min(bounds["x0"]) - 0.1, max(bounds["x1"]) + 0.1, min(bounds["y0"]) - 0.1, max(bounds["y1"]) + 0.1])
                ax.set_title(regime, fontsize = 20)
    fig.suptitle(", ".join(cols), fontsize = 22)
    fig.tight_layout()
    plt.savefig(title)
    return fig

'''
MODEL DYNAMICS
'''

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_dynamics(stats):
    '''
    Method creating raw images of the probabilistic dynamics of a model
    Args:
        stats: a pandas DataFrame containing the raw data about the transition probabilities of a model


    '''
    states = REGIMES
    fig, axs = plt.subplots(1, 3, figsize=(22.5, 7))
    for ax, method in zip(axs.flat, stats.index):
        Q = stats.loc[method].values.reshape((4, 4))
        Q = pd.DataFrame(Q, columns=states, index=states)

        def _get_markov_edges(Q):
            edges = {}
            for col in Q.columns:
                for idx in Q.index:
                    edges[(idx, col)] = Q.loc[idx, col]
            return edges

        edges_wts = _get_markov_edges(Q)
        G = nx.MultiDiGraph()
        G.add_nodes_from(states)
        for k, v in edges_wts.items():
            tmp_origin, tmp_destination = k[0], k[1]
            G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
        nx.draw_networkx(G, pos, node_size=500, node_color='#ffffff', ax=ax,
                         cmap='RdBu_r', edge_cmap='RdBu_r', font_weight='medium')
        # create edge labels for jupyter plot but is not necessary
        edge_labels = {(n1, n2): d['label'] for n1, n2, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        nx.drawing.nx_pydot.write_dot(G, 'W:/UK/Research/Private/WEATHER/STAGE_ABALDO/scripts/mc_imgs/' + method + '.dot')
        (graph,) = pydot.graph_from_dot_file('W:/UK/Research/Private/WEATHER/STAGE_ABALDO/scripts/mc_imgs/' + method + '.dot')
        graph.write_png('W:/UK/Research/Private/WEATHER/STAGE_ABALDO/scripts/mc_imgs/' + method + '.png')
    plt.close()

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def compare_dynamics(predictions, targets, model, title = None, **kwargs):
    '''
    Method to compare the dynamics of two models
    Args:
        predictions: a pandas DataFrame containing the predictions of historical daily weather regimes, whose dynamics are plotted
        targets: a pandas DataFrame containign the predictions of historical daily weather regimes, whose dynamics are thresholded and plotted
        model: the model whose dynamics are plotted
        title (optional): title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    fig, ax = plt.subplots(1,1, figsize=(22.5,7))

    prob_model = predictions.xs(model, level = 0, axis = 1).drop('Prediction', axis = 1)
    prob_model.reindex(columns = sorted(prob_model.columns)).plot(kind='line', linewidth = 2.1, ax = ax)
    colors = {line.get_label(): line.get_color() for line in ax.get_lines()}

    for idx, row in targets.iterrows():
        #print(row['Prediction'])
        ax.fill_betweenx([0., -0.02], [idx-timedelta(days=1)], [idx], color = colors[row['Prediction']], alpha = 1.)
    ax.tick_params(axis = 'both', which = 'both', labelsize = 16)
    ax.set_ylabel("Probability", fontsize = 18)
    if title:
        plt.savefig(title)
    return fig

'''
SUB-SEASONAL FORECASTS
'''

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_weights_forecast(forecast, date, title = None, **kwargs):
    '''
    Method to plot the weights of each regime from a pandas DataFramre
    Args:
        forecast: a pandas DataFrame containing a set of predictions
        date: a str to define the figure title
        title (optional): title to save the figure

    Returns:
        A matplotlib figure

    '''
    fig, ax = plt.subplots(1,1, figsize = (15,7))
    forecast.plot(kind = "line", linewidth = 2.1, marker = 'o', ax = ax)
    ax.set_title("Forecast date: "+date.strftime("%d, %B, %Y"))
    if title:
        plt.savefig(title)
    return fig


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_forecast_distributions(variable_df, forecast, predictions, model, thresh = None):
    '''
    Method to get the distributions associated to sub-seasonal forecasts
    Args:
        variable_df: a pandas DataFrame referred to a particular energy variable
        forecast: a pandas DataFrame containing a set of forecasted weights of regimes
        predictions: a pandas DataFrame containing predicitons of historical daily weather regimes
        model: the model whose predictions are considered
        thresh (optional): the threshold on the minimum regime probability to be considered

    Returns:
        the mean value of the normal associated to the variable
        the std value of the normal associated to the variable
        the weighted distributions associated to the variable
        the weighted areas associated to below and above normal points
        the weighted means of the distributions

    '''
    normal_mean = variable_df.mean()
    normal_std = variable_df.std()
    lb, ub = normal_mean - normal_std, normal_mean + normal_std
    points = np.linspace(variable_df.min(), variable_df.max(), 100)
    variable_df.dropna(inplace = True)
    distributions_regimes = dict()
    for regime in sorted(REGIMES):
        weights = predictions.loc[:,(model, regime)]
        if thresh is not None:
            weights = weights[weights >= thresh]
        distributions_regimes[regime] =  scipy.stats.gaussian_kde(variable_df.loc[weights.index.intersection(variable_df.index)],
                                         weights=weights.loc[weights.index.intersection(variable_df.index)].values,
                                         bw_method=.1)

    distributions = {k: v.pdf(points) / sum(v.pdf(points)) for k, v in distributions_regimes.items()}
    distributions = pd.DataFrame.from_dict(distributions)

    areas = {k: np.array([v.integrate_box_1d(points.min(), lb),
                          v.integrate_box_1d(ub, points.max())])
             for k, v in distributions_regimes.items()}
    areas = pd.DataFrame.from_dict(areas)

    means = {k: [sum(points * v.pdf(points)) / sum(v.pdf(points))]
             for k, v in distributions_regimes.items()}
    means = pd.DataFrame.from_dict(means)
    weighted_dist = np.dot(forecast, distributions.values.T)
    weighted_areas = np.dot(forecast, areas.values.T)
    weighted_means = np.dot(forecast, means.values.T)

    return normal_mean, normal_std, weighted_dist, weighted_areas, weighted_means

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_forecast_errors(subseasonal_df, predictions, model, title = None, **kwargs):
    '''
    Method to get the forecast errors associated to a set of sub-seasonal forecasts
    Args:
        subseasonal_df: a pandas DataFrame containing the set of sub-seasonal forecasts
        predictions: a pandas DataFrame containing the predictions of historical daily weather regimes
        model: str indicating the model whose predictions are considered
        title (optional): title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    forecast = subseasonal_df.loc[pd.IndexSlice[:,"2020-12-01":"28-02-2021", :], :].copy()
    #forecast.reset_index(level=2, drop=True, inplace=True)
    #forecast = forecast.groupby([pd.Grouper(freq="W-MON", label = 'left', closed='left', level = 0),
    #                             forecast.index.get_level_values(1)]).agg('last')
    forecast = forecast.apply(lambda x: x + x['Unknown'] * x if x['Unknown'] != 1 else 0.25, axis=1). \
        drop('Unknown', axis=1).apply(lambda x: x / x.sum(), axis=1)

    targets = predictions.loc[forecast.index.get_level_values(1).unique()].xs(model, level = 0, axis = 1).drop('Prediction', axis = 1)
    def compute_error(row, type = 'mae'):
        if type == 'mae':
            res =  abs(row - targets.loc[row.name[1]])
        elif type == 'mse':
            res =  (row - targets.loc[row.name[1]])**2
        else:
            pass
        res['AVG'] = np.mean(res)

        return res

    forecast_errors = forecast.apply(lambda x: compute_error(x), axis = 1)
    '''
    def diff(start, end):
        x = pd.to_datetime(end) - pd.to_datetime(start)
        return int(x / np.timedelta64(1,'W'))

    forecast_errors['Weeks'] = forecast_errors.apply(lambda x: diff(x.name[0], x.name[1]), axis = 1)
    forecast_errors = forecast_errors.groupby('Weeks').mean()
    '''
    forecast_errors = forecast_errors.groupby(forecast_errors.index.get_level_values(2)).mean() #by step
    fig, ax = plt.subplots(1,1, figsize = (15,7))
    forecast_errors.loc[:,sorted(forecast.columns)].plot(kind = 'line', marker = 'o', linewidth = 2, ax = ax)
    forecast_errors.loc[:,'AVG'].plot(kind = 'line', linestyle = '--', linewidth = 1, color = 'k', ax = ax)
    ax.set_xlabel('Step')
    ax.set_ylabel('Error')
    if title:
        fig.savefig(title + ".png")
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_average_weights_forecast(subseasonal_df, predictions, model):
    '''
    Method to evaluate the average weights of each regime in the hisotry of the sub-seasonal forecasts
    Args:
        subseasonal_df: a pandas DataFrame containing the set of sub-seasonal forecasts
        predictions: a pandas DataFrame containing the predictions of historical daily weather regimes

    Returns:
        a pandas DataFrame containing the statistics

    '''
    weights = subseasonal_df.loc[pd.IndexSlice[:, '2020-11-01':'2021-02-28',:],:]
    weights = weights.reindex(sorted(weights.columns), axis=1)
    weights = weights.apply(lambda x: x + x['Unknown'] * x if x['Unknown'] != 1 else 0.25, axis=1,
                              result_type='broadcast')
    weights = weights.drop('Unknown', axis=1).apply(lambda x: x / x.sum(), axis=1)
    weights = weights.groupby(weights.index.get_level_values(2)).mean()
    fig, ax = plt.subplots(1,1, figsize = (15,7))
    weights.plot(kind = 'line', marker = 'o', linewidth = 2, ax = ax)
    ax.set_ylabel('Average Weight')

    '''
    preds = predictions.xs(model, level = 0, axis = 1).drop('Prediction', axis = 1)
    regimes_days_global = preds.sum(0) / 42
    regimes_days_last = preds.loc["2020-12-01":"2021-02-28"].sum(0)
    weights_calibrated = weights * regimes_days_global / regimes_days_last
    weights_calibrated = weights_calibrated.apply(lambda x: x / x.sum(), axis = 1)
    colors = {line.get_label(): line.get_color() for line in ax.get_lines()}
    for regime, color in colors.items():
        weights_calibrated[regime].plot(kind = 'line', marker='^', linewidth=2, linestyle = '-.', ax=ax, c = color)
    '''
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_forecast_distribution(df, forecast, predictions, countries, variables, date, model, title = None, **kwargs):
    '''
    Method to plot the distirbutions of the forecasts
    Args:
        df: a pandas DataFrame containing data about energy variables
        forecast: a pandas DataFrame containing a set of sub-seasonal forecasts
        predictions: a pandas DataFrame containing the predicitons of historical daily weather regimes
        countries: a list of str containing the names of the countries to be considered
        variables: a list of str containing the columns names of the eneegy variables whose distributions will be plotted
        date: the date associated tot he forecasts
        model: str indiciating the model whose predictions are considered
        title (optional): title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    if countries !=  'EU-7':
        df = df.loc[pd.IndexSlice[countries, :], variables]
    df.reset_index(level=0, drop = True, inplace=True)

    fig, axs = plt.subplots(len(variables), len(forecast), figsize=(3*len(forecast), 10 * len(variables)), sharex = 'row', sharey = 'row')

    for i, variable in enumerate(variables):
        variable_df = df.loc[df.index.month == date.month][variable].dropna(how='any')
        normal_mean, normal_std, weighted_dist, weighted_areas, weighted_means = get_forecast_distributions(
            variable_df, forecast, predictions, model)
        lb, ub = normal_mean - normal_std, normal_mean + normal_std
        points = np.linspace(variable_df.min(), variable_df.max(), 100)

        if len(axs.shape) == 1:
            axs = axs[np.newaxis,...]

        weighted_dist = weighted_dist * 1000

        for j, ax in enumerate(axs[i,:]):
            ax.plot(weighted_dist[j], points, color='k')
            #ax.plot(normal_dist, points, '--', color='k')
            ax.fill_betweenx(points, weighted_dist[j], 0, where = points < lb,
                            interpolate = True, alpha = .5, color = 'b', label = 'below normal')
            ax.fill_betweenx(points, 0, weighted_dist[j], where=points > ub,
                            interpolate = True, alpha=.5, color='r', label = 'above normal')
            ax.set_title(str(weighted_dist.shape[0]-j-1) + " weeks before")
            #ax.axhline(variable_df.loc[date], color = '#0D8C8C', label = 'Truth')
            ax.axhline(normal_mean, color = 'k', linestyle = '--', label = 'Normal')
            axis_to_data = (ax.transData + ax.transAxes.inverted())
            text_pos_lb = axis_to_data.transform((ax.get_xlim()[1] /1.2, lb))
            text_pos_ub = axis_to_data.transform((ax.get_xlim()[1] / 1.2, ub))
            ax.text(*text_pos_lb, str(round(weighted_areas[j,0]*100,1)) + "%", transform = ax.transAxes, bbox = dict(facecolor = 'darkgrey', alpha =.5),
                    horizontalalignment = 'center', verticalalignment = 'center', fontsize = 16, weight = 'bold')
            ax.text(*text_pos_ub, str(round(weighted_areas[j, 1] * 100, 1)) + "%",transform = ax.transAxes, bbox = dict(facecolor = 'darkgrey', alpha =.5),
                    horizontalalignment='center', verticalalignment='center', fontsize = 16, weight = 'bold')
            ax.set_ylabel(variable, fontsize = 16)
            ax.legend(title=variable +", "+countries+"\nMean Value: " + str(round(weighted_means[j,0],2)), loc="upper right")

    fig.tight_layout()
    if title:
        fig.savefig(title + f"/distributions_{date.strftime('%Y%m%d')}_{'_'.join(variables)}.png")

    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_forecast_boxplots(df, forecast, predictions, countries, variables, date, model, title = None, **kwargs):
    '''
    Method to plot the distributions of the forecasts as boxplots
    Args:
        df: a pandas DataFrame containing data about energy variables
        forecast: a pandas DataFrame containing a set of sub-seasonal forecasts
        predictions: a pandas DataFrame containing the predicitons of historical daily weather regimes
        countries: a list of str containing the names of the countries to be considered
        variables: a list of str containing the columns names of the eneegy variables whose distributions will be plotted
        date: the date associated tot he forecasts
        model: str indiciating the model whose predictions are considered
        title (optional): title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    '''
    if countries !=  'EU-7':
        df = df.loc[pd.IndexSlice[countries, :], variables]
    df.reset_index(level=0, drop = True, inplace=True)
    '''

    fig, axs = plt.subplots(len(variables),1, figsize=(30, 15 * len(variables)))
    if not hasattr(axs, "flat"):
        axs = np.array([axs])


    for i, variable in enumerate(variables):
        variable_df = df.loc[:,variable].dropna(how='any')
        normal_mean, normal_std, weighted_dist, weighted_areas, weighted_means = get_forecast_distributions(
            variable_df, forecast, predictions, model)
        points = np.linspace(variable_df.min(), variable_df.max(), 100)

        weighted_dist = weighted_dist * 1000

        boxplots = []
        for j in range(len(weighted_dist)):
            boxplot = pd.DataFrame.from_dict(dict(samples = points, freq = weighted_dist[j]))
            boxplot["freq"] = (boxplot["freq"]).astype(int)
            boxplot = boxplot.reindex(boxplot.index.repeat(boxplot["freq"])).drop("freq", axis = 1)
            boxplots.append(boxplot.reset_index(drop=True))

        boxplots = pd.concat(boxplots, axis = 1)
        boxplots.set_axis([dt.strftime("%d-%m-%Y") for dt in pd.to_datetime(forecast.index)], axis=1, inplace = True)
        colors = plt.cm.coolwarm(mpl.colors.TwoSlopeNorm(vmin = normal_mean - normal_std,
                                                        vcenter = normal_mean,
                                                        vmax = normal_mean + normal_std)(boxplots.mean(0)))
        sns.boxplot(data=boxplots, ax=axs[i], palette=colors, showmeans = True,
                    meanprops = dict(marker="o", markerfacecolor="w", markeredgecolor="b", markersize=10))
        #axs2[i].axhline(variable_df.loc[date], color='#0D8C8C', label='Truth')
        axs[i].axhline(normal_mean, color='k', linestyle='--', label='Normal')
        axs[i].set_ylabel(variable, fontsize = 16)
        axs[i].tick_params(axis='both', which='major', labelsize=14)

    fig.tight_layout()
    if title:
        fig.savefig(title + f"/boxplots_{date.strftime('%Y%m%d')}_{'_'.join(variables)}.png")

    return fig
@st.cache(suppress_st_warning=True)#, allow_output_mutation=True)
def bayesian_forecasts(subseasonal_df, forecast, predictions, model):
    '''
    Method to incorporate an historical prior inside forecasts
    Args:
        forecast: a pandas DataFrame containing sub-seasonal forecasts
        predictions: a pandas DataFrame containing predictions of historical daily weather regimes
        model: the model whose predictions are considered

    Returns:
        A pandas DataFrame containing the forecasts with a prior knowledge enclosed

    '''
    '''
    for step in values.index.get_level_values(2).unique():
        vals = values.xs(step, level = 2)
        mean, cov = vals.mean(0), vals.cov() + np.eye(vals.shape[1])*1e-3
        print(mean, cov)
        evidence = scipy.stats.multivariate_normal.pdf(forecast.iloc[step,:],mean, cov)
        print(evidence)
        forecast.iloc[step,:] /= evidence
    '''
    def get_alpha(mu, var):
        alphas = []
        X2 = mu**2 + var
        for i in range(len(mu)-1):
            alphas.append(
                (mu[0] - X2[0])*mu[i] / (X2[0] - mu[0]**2)
            )
        alphas.append(
            (mu[0] - X2[0]) * (1- np.sum(mu[:-1])) / (X2[0] - mu[0] ** 2)
        )

        return np.array(alphas)

    preds = predictions.xs(model, level=0, axis=1).drop('Prediction', axis=1)
    preds = preds.reindex(columns=sorted(preds.columns))
    #prior = prior.sum(0) / 42
    #prior = prior / prior.sum()
    mean, var = preds.mean(0), preds.var(0)
    alphas_prior = get_alpha(mean.values, var.values)

    posterior = []
    for step in subseasonal_df.index.get_level_values(2).unique():
        values = subseasonal_df.xs(step, level = 2).reindex(sorted(subseasonal_df.columns), axis=1)
        values = values.apply(lambda x: x + x['Unknown'] * x if x['Unknown'] != 1 else 0.25, axis=1,
                              result_type='broadcast')
        values = values.drop('Unknown', axis=1).apply(lambda x: x / x.sum(), axis=1)
        multinomial_params = values.sum(0)
        weights = (alphas_prior + multinomial_params) / np.sum(alphas_prior + multinomial_params)
        print(step, weights.T)
        posterior.append(weights.values[np.newaxis, ...])
    posterior = np.concatenate(posterior)
    print(posterior.shape)
    return (forecast * posterior).apply(lambda x: x / x.sum(), axis = 1)


@st.cache(suppress_st_warning=True)#, allow_output_mutation=True)
def actualize(val, country):
    '''
    Method to convert a value of an energy variable to the actual capacity
    Args:
        val: the value to be actualized
        country: the country associated to that value

    Returns:
        value actualized by capacity

    '''
    capacities = dict(GE=63.143, UK=24.068, FR=18.596, IT=11.517, ES=27.966, BE=4.831, NE=7.062)
    return val * capacities[country]

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_forecast_anomalies(df, forecast, short_term, predictions, countries, variables, date, model, title = None, plug = False, **kwargs):
    '''
    Method to plot the distributions of the forecasts as boxplots
    Args:
        df: a pandas DataFrame containing data about energy variables
        forecast: a pandas DataFrame containing a set of sub-seasonal forecasts
        short_term: a pandas DataFrame containing a set of short-term forecasts
        predictions: a pandas DataFrame containing the predicitons of historical daily weather regimes
        countries: a list of str containing the names of the countries to be considered
        variables: a list of str containing the columns names of the eneegy variables whose distributions will be plotted
        date: the date associated tot he forecasts
        model: str indiciating the model whose predictions are considered
        title (optional): title to save the figure
        plug: boolean indicating whtehr to plug short-term forecasts
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    '''
    if countries !=  'EU-7':
        df = df.loc[pd.IndexSlice[countries, :], variables]
    df.reset_index(level=0, drop = True, inplace=True)
    '''
    fig, axs = plt.subplots(len(variables),1, figsize=(30, 15 * len(variables)))

    if not hasattr(axs, "flat"):
        axs = np.array([axs])

    for i, variable in enumerate(variables):
        variable_df = df.loc[:, variable].dropna(how='any')
        normal_mean, normal_std, weighted_dist, weighted_areas, weighted_means = get_forecast_distributions(
            variable_df, forecast, predictions, model)#, thresh = .9)
        #'''
        if 'Load Factor' in variable:
            normal_mean = actualize(normal_mean, countries)
            normal_std = actualize(normal_std, countries)
            weighted_means = actualize(weighted_means, countries)
        #'''

        weighted_anomalies = weighted_means - normal_mean
        colors = mpl.cm.coolwarm(mpl.colors.TwoSlopeNorm(vmin=-normal_std,
                                                         vcenter=0,
                                                         vmax = normal_std)(np.squeeze(weighted_anomalies)))
        if plug:
            xs = list(range(1, len(forecast)-1))
            xs += [xs[-1]+len(short_term)-7, xs[-1]+len(short_term)-4]
        else:
            xs = [dt.strftime("%d-%m-%Y") for dt in pd.to_datetime(forecast.index)]


        bars = axs[i].bar(xs, np.squeeze(weighted_anomalies), width = 1,
                          edgecolor='k', linewidth = 1, color = np.squeeze(colors))

        for bar in bars.patches:
            axs[i].annotate(str(int(bar.get_height()*100/(normal_mean))) + "%",
                             xy=((bar.get_x()+ bar.get_width()/2), bar.get_height()),
                             ha='center',va='center', fontsize = 18, xytext = (0, 20 if bar.get_height()>0 else -20),
                             textcoords = 'offset points', weight = 'bold')
        if countries == 'GE' and variable == 'Wind Load Factor' and plug:
            short_term[['ECMWF_ENS','ECMWF_HRES','GFS','Meteologica']] -= normal_mean
            for col in ['ECMWF_ENS','ECMWF_HRES','GFS','Meteologica']:
                axs[i].plot(range(xs[-1]-len(short_term)+4, xs[-1]+4), actualize(short_term.loc[:, col],countries), linestyle = '--', label = col)
            true_val = short_term.iloc[-1].loc['Observation'] - normal_mean
            axs[i].axhline(true_val, color='#0D8C8C', label='Truth')
        #axs[i].set_xticks(xs[:-1] + list(range(xs[-1]-len(short_term)+4, xs[-1]+4)))
        #axs[i].set_xticklabels([date - timedelta(weeks=len(forecast)-j-1) for j in range(len(forecast)-1)])
        #axs[i].set_xticks([])
        axs[i].set_ylabel(variable + " Anomaly (GW)", fontsize=22)
        axs[i].legend(fontsize = 18)
        axs[i].set_xticks(np.arange(*axs[i].get_xlim(), 8))
        axs[i].tick_params(axis='both', which = 'major', labelsize = 20)


    fig.tight_layout()
    if title:
        fig.savefig(title + f"/anomalies_{date.strftime('%Y%m%d')}_{'_'.join(variables)}.png")

    return fig

'''
METEO-FRANCE
'''

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_conf_matrix(predictions, targets, models, **kwargs):
    '''
    Method to plot the confusion matrix of the predictions of one or more models
    Args:
        predictions: a pandas DataFrame containing the predictions of historical daily weather regimes
        targets: a pandas DataFrame containing the predictions of historical daily weather regimes targets
        models: a list of str containing the models' names whose predictions are used
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    fig, axs = plt.subplots(1, len(models), figsize=(7.5*len(models), 7), sharey = True)
    cbar_ax = fig.add_axes([.91, .1, .02, .9])
    for i, (model, ax)  in enumerate(zip(models, axs)):
        conf_matrix = confusion_matrix(
            np.argmin(targets.values, axis=1),
            np.argmax(predictions.xs(model, level=0, axis=1).values, axis=1),
            normalize='true') * 100
        sns.heatmap(conf_matrix, ax = ax, cbar = i == 0, vmin = 0, vmax = 100, cbar_ax = None if i else cbar_ax,
                    cbar_kws={'format': '%.0f%%'}, annot = True, annot_kws={"size": 20}, cmap='Blues',
                    yticklabels=sorted(np.unique(predictions.columns.get_level_values(1))),
                    xticklabels=sorted(np.unique(predictions.columns.get_level_values(1)))
                    )
        if i == 0:
            ax.set_ylabel("Metéo-France", fontsize = 18, fontdict = {'fontweight': 'bold'})
        ax.set_xlabel(model, fontsize = 18, fontdict = {'fontweight': 'bold'})
        ax.tick_params(axis='x', rotation=0, labelsize = 16)
        ax.tick_params(axis='y', rotation=0, labelsize = 16)

    fig.tight_layout(rect = [0,0,.9,1])
    '''
    conf_matrix = np.concatenate(
        [confusion_matrix(
            np.argmin(targets.values, axis=1),
            np.argmax(predictions.xs(model, level=0, axis=1).values, axis=1),
            normalize='true') * 100
         for model in models
         ], axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(7.5*len(models), 7))
    sns.heatmap(conf_matrix, annot=True, ax=ax, cbar_kws={'format': '%.0f%%'}, annot_kws={"size": 16}, cmap = 'Blues',
                yticklabels=sorted(np.unique(predictions.columns.get_level_values(1))),
                xticklabels=["\n".join(list(e)[::-1]) for e in
                             list(itertools.product(
                                 models,
                                 sorted(np.unique(predictions.columns.get_level_values(1)))))],
                )
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    ax.set_ylabel("Metéo-France")
    for i in range(conf_matrix.shape[0], conf_matrix.shape[1] + 1, conf_matrix.shape[0]):
        ax.axvline(i, color='white')
    '''
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_multiclass_roc(predictions, targets, models, **kwargs):
    '''
    Method to plot the ROC curves of the predictions of one or more models
    Args:
        predictions: a pandas DataFrame containing the predictions of historical daily weather regimes
        targets: a pandas DataFrame containing the predictions of historical daily weather regimes targets
        models: a list of str containing the models' names whose predictions are used
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    cols = targets.columns
    targets = label_binarize(np.argmin(targets.values, axis=1), classes=range(len(cols)))
    n_classes = targets.shape[1]

    fig, axs = plt.subplots(1, len(models), figsize=(len(models)*7.5, 7), sharey = True)
    if not hasattr(axs, 'flat'):
        axs = np.array([axs])

    for j, (ax, model) in enumerate(zip(axs.flat, models)):
        roc_auc, tpr, fpr = dict(), dict(), dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(targets[:, i], predictions.xs(model, level=0, axis=1).iloc[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(targets.ravel(),
                                                  predictions.xs(model, level=0, axis=1).values.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[j] for j in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        '''
        ax.plot(fpr["micro"], tpr["micro"], label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
                color='deeppink', linestyle=":", linewidth=4)

        ax.plot(fpr["macro"], tpr["macro"], label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
                color='navy', linestyle=":", linewidth=4)
        '''
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], linewidth=2.1, label="ROC curve {0}\n(area: {1:0.2f})". \
                    format(cols[i], roc_auc[i]))

        ax.plot([0, 1], [0, 1], 'k--', linewidth=2.1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (FPR)', fontsize = 18)
        if j == 0:
            ax.set_ylabel('True Positive Rate (TPR)', fontsize = 18)
        ax.legend(loc="lower center", fontsize = 14, bbox_to_anchor = (.5, -.37), ncol = 2)

        ax.set_title(model, fontdict = {'fontsize': 18, 'fontweight': 'bold'})
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_historical_probabilities(targets, pivot = False, title = None, **kwargs):
    '''
    Mehtod to plot the historical probabilities associated to a set of predictions
    Args:
        targets: a pandas DataFrame containing the historical predictions of daily weather regimes
        pivot: boolean indicating whether to pivot the targets dataframe
        title (optional): title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

    '''
    if pivot:
        targets.drop(targets.columns.difference(['Prediction']), axis=1, inplace=True)
        targets = targets.pivot_table(index=targets.index, columns='Prediction',
                                        aggfunc=dict(Prediction='count'))
    targets = targets.fillna(0).resample('MS').sum()
    targets = targets[targets.index.month.isin(MONTHS)]

    fig, ax = plt.subplots(1, 1, figsize=(30, 7))
    targets = targets.apply(lambda x: x / x.sum(), axis=1)
    if targets.columns.nlevels == 2:
        targets.columns = targets.columns.droplevel()
    targets.plot.bar(stacked=True, ax=ax, width=1.0, edgecolor='k')
    ax.legend(targets.columns, bbox_to_anchor=(1.1, 1.05))
    ax.set_ylim(0, 1)
    ax.set_ylabel("CDF", fontsize=16)
    ax.set_xlabel("")
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xticklabels([ts.strftime('%Y') if ts.year != targets.index[idx - 1].year
                        else "" for idx, ts in enumerate(targets.index)])
    ax.figure.autofmt_xdate(rotation=90, ha='center')
    if title is not None:
        plt.savefig(title, bbox_inches = 'tight')
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_historical_counts(targets, pivot = False):
    '''
        Mehtod to get the historical counts associated to a set of predictions
        Args:
            targets: a pandas DataFrame containing the historical predictions of daily weather regimes
            pivot: boolean indicating whether to pivot the targets dataframe

        Returns:
            A matplotlib figure

    '''
    def get_season(year, month):
        if month in MONTHS:
            if month == 12:
                return year+1, MONTHS_STR
            return year, MONTHS_STR
        return year,"_"

    if pivot:
        targets.drop(targets.columns.difference(['Prediction']), axis=1, inplace=True)
        targets = targets.pivot_table(index=targets.index, columns='Prediction',
                                      aggfunc=dict(Prediction='count')).fillna(0)

    targets = targets.groupby(lambda x: get_season(x.year, x.month)).sum()

    targets.index = pd.MultiIndex.from_tuples(targets.index, names=["year", "season"])
    targets = targets.xs(MONTHS_STR, level=1)
    if targets.columns.nlevels == 2:
        targets.columns = targets.columns.droplevel()
    return targets


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_historical_counts(targets, pivot = False, title = None, **kwargs):
    '''
    Mehtod to plot the historical counts associated to a set of predictions
    Args:
        targets: a pandas DataFrame containing the historical predictions of daily weather regimes
        pivot: boolean indicating whether to pivot the targets dataframe
        title (optional): title to save the figure
        **kwargs: optional additional parameters

    Returns:
        A matplotlib figure

        '''
    targets = get_historical_counts(targets, pivot)
    fig, ax = plt.subplots(1, 1, figsize=(30, 7))
    targets.plot.bar(ax=ax, edgecolor='k')
    ax.set_ylim(0, 60)
    ax.legend(targets.columns, bbox_to_anchor=(1.1, 1.05))
    ax.set_ylabel("Number of Days", fontsize=16)
    ax.set_xlabel("")
    ax.tick_params(axis='both', which = 'major', labelsize = 16)
    if title is not None:
        plt.savefig(title, bbox_inches = 'tight')
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_hydro_corr(targets, reservoir, inflow, groundwater, pivot = False):
    '''
    Method to plot some correlation statistics of hydro, considering the reservoir filling, inflow, and groundwater variables
    Args:
        targets: a pandas DataFrame containing the predictions of historical daily weather regimes
        reservoir: a pandas DataFrame containing the history of reservoir filling observations
        inflow: a pandas DataFrame containing the history of inflow observations
        groundwater: a pandas DataFrame containing the history of groundwater observations
        pivot: boolean indicating whether to pivot the targets dataframe

    Returns:
        Three matplotlib figures, one for each variable

    '''
    targets = get_historical_counts(targets, pivot)

    reservoir = reservoir.loc[reservoir.index.month.isin(range(3,8))]
    range_reservoir = reservoir.groupby(reservoir.index.year).agg(dict(Reservoir=["min", "max"]))
    range_reservoir[('Reservoir', 'range')] = range_reservoir[('Reservoir', 'max')] - range_reservoir[('Reservoir', 'min')]
    range_reservoir.columns = range_reservoir.columns.droplevel()
    range_reservoir['NAO+'] = targets.loc[range_reservoir.index, 'NAO+']
    fig1, ax = plt.subplots(1,1, figsize=(7,7))
    sns.scatterplot(data = range_reservoir, x = 'range', y = 'NAO+', hue = list(map(str,range_reservoir.index)), ax = ax)

    inflow = inflow.loc[inflow.index.month.isin(range(3, 8))]
    totals = inflow.groupby(lambda x: x.year).sum()
    totals['NAO+'] = targets.loc[totals.index, 'NAO+']
    inflow = inflow.groupby(lambda x: x.year).agg(dict(Inflow=np.cumsum))
    inflow['year'] = inflow.index.year
    inflow.index = inflow.index.map(lambda x: x.strftime("%m-%d"))
    fig2, axs = plt.subplots(1,2, figsize=(15,7))
    sns.lineplot(x=inflow.index, y = inflow['Inflow'], hue = inflow['year'], linestyle = '--', ax = axs[0])
    axs[0].set_xticks(np.arange(*axs[0].get_xlim(), 10))
    axs[0].tick_params(axis='x', labelrotation = 45)
    sns.scatterplot(data = totals, x = 'Inflow', y = 'NAO+', hue = totals.index, ax = axs[1])

    groundwater = groundwater.loc[groundwater.index.month.isin(MONTHS)]
    totals = groundwater.groupby(lambda x: x.year if x.month!=12 else x.year+1).max() - groundwater.groupby(lambda x: x.year if x.month!=12 else x.year+1).min()
    print(totals)
    totals['NAO+'] = targets.loc[totals.index, 'NAO+']
    fig3, axs = plt.subplots(1, 2, figsize=(15, 7))
    sns.regplot(data = totals, x='NAO+', y='SnowGroundWater', scatter_kws=dict(s=40, edgecolor = 'k'), ax = axs[0])
    axs[0].tick_params(axis='both', which = 'both', labelsize = 16)
    axs[0].set_ylabel("Snow Reservoir (TWh)", fontsize = 18)
    normal_mean = totals['SnowGroundWater'].mean()
    normal_std = totals['SnowGroundWater'].std()
    colors = mpl.cm.coolwarm(mpl.colors.TwoSlopeNorm(vmin=-normal_std,
                                                     vcenter=0,
                                                     vmax=normal_std)(totals['SnowGroundWater'] - normal_mean))
    axs[1].bar(totals.index.map(str), totals['SnowGroundWater'] - normal_mean, edgecolor='k', linewidth=1, color=np.squeeze(colors))
    axs[1].set_ylabel("Snow Reservoir Anomaly (TWh)", fontsize = 18)
    axs[1].set_xticks(np.arange(*axs[1].get_xlim(), 3))
    axs[1].tick_params(axis='x', labelrotation=90, which = 'both', labelsize = 16)
    axs[1].tick_params(axis='y', which='both', labelsize=16)
    return fig1, fig2, fig3