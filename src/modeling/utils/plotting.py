from modeling.utils.data import *
from modeling.utils.models import *
import math
from datetime import datetime
from sklearn.metrics import silhouette_score

import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.pyplot as plt
from matplotlib import animation
import plotly.graph_objects as go
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import seaborn as sns
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['mpl_toolkits.legacy_colorbar'] = False
PATH = "W:/UK/Research/Private/WEATHER/STAGE_ABALDO/scripts/imgs/"


def save_anomalies(dt, title):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    lats, longs = np.unique(dt.latitude.values), np.unique(dt.longitude.values)
    gphs = dt.to_array().values.squeeze()
    cset = ax.contourf(longs, lats, gphs, cmap='coolwarm', transform=ccrs.PlateCarree(), levels=100, alpha=.7)
    cset = ax.contour(longs, lats, gphs, transform=ccrs.PlateCarree(), colors='k')
    plt.savefig(title, bbox_inches='tight', pad_inches=0)
    plt.close()

"""
def plot_gph(dt, **kwargs):
    '''
    Method to plot geopotential height
    Args:
    - dt: an xarray DataArray
    '''
    if 'ax' not in kwargs:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    else:
        ax = kwargs['ax']

    if 'ax' not in kwargs:
        axins = inset_axes(ax,
                           width="3%",  # width = 10% of parent_bbox width
                           height="100%",  # height : 50%
                           loc=6,
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=ax.transAxes,
                           borderpad=0,
                           )
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])

    ax.coastlines()
    lats, longs = np.unique(dt.latitude.values), np.unique(dt.longitude.values)
    gphs = np.squeeze(dt.values)
    if 'reshape' in kwargs:
        gphs = np.reshape(gphs, (len(lats), len(longs)))

    cset = ax.contourf(longs, lats, gphs, cmap='RdBu_r', transform=ccrs.PlateCarree(), levels=50, alpha=.7,
                       norm=kwargs['norm'] if 'norm' in kwargs else None)
    if 'ax' not in kwargs:
        plt.colorbar(cset, cax=axins)
    cset = ax.contour(longs, lats, gphs, transform=ccrs.PlateCarree(), colors='k')

    if 'savefig' in kwargs:
        plt.savefig(kwargs["savefig"], bbox_inches='tight')

    if 'ax' not in kwargs:
        plt.show()
    else:
        return cset
"""


def plot_gph(dt, **kwargs):
    '''
    Method to plot geopotential height
    Args:
    - dt: an xarray DataArray
    '''
    xlim, ylim = (dt['longitude'].values.min(), dt['longitude'].values.max()), \
                 (dt['latitude'].values.min(), dt['latitude'].values.max())
    trans = ccrs.PlateCarree()
    proj = ccrs.LambertConformal((xlim[1] + xlim[0]) / 2, (ylim[1] + ylim[0]) / 2)
    vmin = dt.quantile(.05).values
    vmax = dt.quantile(.95).values
    fig, ax = plt.subplots(1, 1, figsize=(7, 7),
                           subplot_kw={'projection': proj})
    cset = dt.plot.contourf(x="longitude", y="latitude", ax=ax, levels=50, alpha=.7,
                                transform=trans, cmap="RdBu_r",
                                norm=mpl.colors.SymLogNorm(vmin=vmin, vmax=vmax,
                                linthresh=max(dt.quantile(.85), 10), base=10),
                                cbar_kwargs=dict(orientation='horizontal', extend='both',
                                                 ticks=[vmin, vmax], format="%d"))

    ax.coastlines()
    rect = mpl.path.Path([[xlim[0], ylim[0]],
                          [xlim[1], ylim[0]],
                          [xlim[1], ylim[1]],
                          [xlim[0], ylim[1]],
                          [xlim[0], ylim[0]],
                          ]).interpolated(20)
    proj_to_data = trans._as_mpl_transform(ax) - ax.transData
    rect_in_target = proj_to_data.transform_path(rect)
    ax.set_boundary(rect_in_target)
    ax.set_extent([xlim[0], xlim[1], ylim[0] - 10, ylim[1]])

    if 'title' in kwargs:
        ax.set_title(kwargs['title'])

    if 'savefig' in kwargs:
        plt.savefig(kwargs['savefig'])

    plt.show()

def plot_normal(dt, locations, freq = 'm', **kwargs):
    '''
    Method to plot the values of the normal across one year for different locations
    Args:
    - dt: an xarray Dataset containing observation values
    - locations: a dictionary containing different locations to be illustrated
        - key: a str containing the location name
        - value: a tuple containing the coordinates of the location
    '''

    # fig, ax = plt.subplots(1,len(locations), figsize=(len(locations)*7,7), sharey = True)
    fig, ax = plt.subplots(len(locations), 1, figsize=(len(locations) * 7, len(locations) * 7))
    if freq == 'm':
        key = 'month'
    elif freq == 'w':
        key = 'week'
    else:
        key = 'day'

    for i, (loc, coords) in enumerate(locations.items()):

        lat, lon = coords
        abslat = np.abs(dt.latitude - lat)
        abslon = np.abs(dt.longitude - lon)
        pos = np.maximum(abslon, abslat)

        ([lonloc], [latloc]) = np.where(pos == np.min(pos))
        local_dt = dt.isel(latitude=latloc, longitude=lonloc)

        for start_year in [1991, 2011]:
            normal_loc = evaluate_normal.__wrapped__(local_dt, domain='local', mode='dynamic', freq = freq,
                                                     start_year=start_year).to_array().squeeze()
            # normal_loc = normal_loc.unstack('month_day')
            # print(normal_loc)
            normal_loc = normal_loc.loc[{
                key: sorted(normal_loc[key].values, key = lambda x: x%12),
            }]
            ax[i].plot(normal_loc[key].to_series().astype(str),
                       normal_loc.values, '--',
                       linewidth=1.2, label=loc + " - " + str(start_year))

            normal_min, normal_mean, normal_max = evaluate_normal.__wrapped__(local_dt, domain='global', mode='dynamic',
                                                                              freq = freq, start_year=start_year)
            normal_min, normal_mean, normal_max = normal_min.to_array().squeeze(), \
                                                  normal_mean.to_array().squeeze(), \
                                                  normal_max.to_array().squeeze()
            ax[i].plot(normal_loc[key].to_series().astype(str),
                       normal_mean.values, '--', linewidth=1.2,
                       label="Normal - " + str(start_year))
            ax[i].fill_between(normal_loc[key].to_series().astype(str),
                               normal_min.values, normal_max.values,
                               label='min-max dev - ' + str(start_year), alpha=.3)
        ax[i].grid(which='both', axis='x')
        ax[i].legend()

        #months = mpl.dates.MonthLocator()
        #days = mpl.dates.DayLocator(bymonthday=(10, 20))
        #formatter_months = mpl.dates.DateFormatter('%b')
        #formatter_days = mpl.dates.DateFormatter('%d')
        #ax[i].xaxis.set_major_locator(months)
        #ax[i].xaxis.set_major_formatter(formatter_months)
        #ax[i].xaxis.set_minor_formatter(formatter_days)
        #ax[i].xaxis.set_minor_locator(days)
        for tick in ax[i].get_xaxis().get_major_ticks():
            tick.set_pad(15)

        ax[i].set_xlabel("Date")
        ax[i].set_ylabel("Geopotential Height [m]")

    fig.tight_layout()
    if 'savefig' in kwargs:
        plt.savefig("imgs/Normal.png")
    plt.show()


def plot_PC(pc, col_name, **kwargs):
    '''
    Method to plot the NAO index, corresponding to the first Principal Component of the Anomaly
    Args:
    - pc: pandas DataFrame containing the PC
    - col_name: the name of the column containing the values to be plotted
    '''
    vmin, vmax = pc[col_name].values.min(), pc[col_name].values.max()
    pc = pd.pivot_table(pc, values=[col_name], index=pc.index.year,
                        columns=pc.index.month, aggfunc=np.mean)

    nao = pd.read_csv("../files/nao.csv", sep=";", names=["year", "month", "idx"])
    nao["day"] = 1
    nao.index = pd.to_datetime(nao[["year", "month", "day"]])
    nao = nao[np.logical_and(nao["month"].isin([1, 2, 12]), nao["year"] >= 1979)]
    nao.drop(["year", "month", "day"], axis=1, inplace=True)

    fig, ax = plt.subplots(1, 1, figsize=(20, 7))

    pc.plot(kind='bar', ax=ax)
    cmap_red = mpl.cm.get_cmap('Reds')
    cmap_blue = mpl.cm.get_cmap('Blues')
    bars = [bar for bar in ax.containers if isinstance(bar, mpl.container.BarContainer)]
    pos = []
    for i, group in enumerate(bars):
        for j, bar in enumerate(group):
            if bar.get_height() > 0:
                color = (bar.get_height() - 0) / (vmax - 0)
                cmap = cmap_blue
            else:
                color = 1 - (bar.get_height() - vmin) / (0 - vmin)
                cmap = cmap_red
            group.patches[j].set_facecolor(cmap(2 * color))
            if not math.isnan(pc.values[j, i]):
                pos.append(bar.get_x())

    ax.minorticks_on()
    ax.plot(sorted(pos), nao['idx'], zorder=0, marker='o', linestyle='dashed')
    ax.set_xlabel("Time")
    ax.set_ylabel(col_name)
    ax.get_legend().remove()
    if 'savefig' in kwargs:
        plt.savefig("../imgs/" + col_name + ".png")
    plt.show()


"""
def plot_EOFS(eofs, **kwargs):
    '''
    Method to plot the 4 EOFs
    Args:
    - eofs: xr.DataArray containing the EOFs
    '''
    vmin, vmax = eofs[:4,:].quantile(.05).values, eofs[:4,:].quantile(.95).values
    norm = mpl.colors.SymLogNorm(vmin=eofs[:4,:].min(), vmax=eofs[:4,:].max(), linthresh=6e-8, base = 10)   

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,dict(map_projection=projection))
    fig = plt.figure(figsize=(15,15))
    axs = AxesGrid(fig, 111, axes_class=axes_class, nrows_ncols=(2, 2), axes_pad=1, label_mode='')
    for neofs in range(4):
        cset = plot_gph(eofs[neofs,:], reshape = True, norm = norm,
                 ax = axs[neofs], title = 'EOF{}'.format(neofs+1))
    cax = fig.add_axes([0.26, 0.25, 0.5, 0.02])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap('RdBu_r'), norm=norm, spacing='uniform',
                                  orientation = 'horizontal', extend = 'both', extendfrac='auto',
                                  ticklocation ='bottom', format = "%.0e",
                                  ticks = [-9e-7, -2e-7, -5e-8, 0., 5e-8, 2e-7, 9e-7])
    if 'savefig' in kwargs:
        plt.savefig(PATH + "eofs.png")
    plt.show()
"""


def plot_EOFS(eofs, **kwargs):
    '''
    Method to plot the 4 EOFs
    Args:
    - eofs: xr.DataArray containing the EOFs
    '''

    xlim, ylim = (eofs['longitude'].values.min(), eofs['longitude'].values.max()), \
                 (eofs['latitude'].values.min(), eofs['latitude'].values.max())
    trans = ccrs.PlateCarree()
    proj = ccrs.LambertConformal((xlim[1] + xlim[0]) / 2, (ylim[1] + ylim[0]) / 2)

    eofs = eofs[:4, :].unstack('latlon').assign_coords(neofs=('time', range(0, 4)))
    eofs = eofs.groupby('neofs').mean()
    vmin, vmax = eofs.quantile(.05).values, eofs.quantile(.95).values

    cset = eofs.plot.contourf(x="longitude", y="latitude", col="neofs", levels=50, col_wrap=2,
                              transform=trans, alpha=.7, cmap="RdBu_r", robust=True,
                              norm=mpl.colors.SymLogNorm(vmin=vmin, vmax=vmax,
                                                         linthresh=eofs.quantile(.85).values, base=10),
                              subplot_kws=dict(projection=proj), figsize=(15, 15),
                              cbar_kwargs=dict(orientation='horizontal', extend='both',
                                               ticks=[vmin, vmax], format="%.0e"))

    for i, ax in enumerate(cset.axes.flat):
        ax.coastlines()
        rect = mpl.path.Path([[xlim[0], ylim[0]],
                              [xlim[1], ylim[0]],
                              [xlim[1], ylim[1]],
                              [xlim[0], ylim[1]],
                              [xlim[0], ylim[0]],
                              ]).interpolated(20)
        proj_to_data = trans._as_mpl_transform(ax) - ax.transData
        rect_in_target = proj_to_data.transform_path(rect)
        ax.set_boundary(rect_in_target)
        ax.set_extent([xlim[0], xlim[1], ylim[0] - 10, ylim[1]])

        ax.set_title(f"EOF-{i + 1}", y=-0.2 if i > 1 else 1)

    if 'savefig' in kwargs:
        plt.savefig(PATH + "eofs.png")
    plt.show()


def plot_density(start_date, end_date, x, y, regimes=True, **kwargs):
    '''
    Method to plot the density of two variables
    Args:
    - start_date: str indicating the start date to consider
    - end_date: str indicating the end date to consider
    - x: pd.Series containing the elements to plot for the variable x
    - y: pd.Series containing the elements to plot for the variable y
    - regimes: boolean indicating whether the reference of the regimes must be inserted
    '''

    x, y = np.squeeze(x.loc[start_date:end_date]), np.squeeze(y.loc[start_date:end_date])

    if 'ax' not in kwargs:
        fig, ax = plt.subplots(figsize=(7, 7))  # subplot_kw={'projection': 'polar'})

    else:
        ax = kwargs['ax']
    sns.kdeplot(x, y, shade=True, ax=ax, cmap='RdBu_r', levels=50, alpha=.7, )
    ax.vlines(0, -3, 3, linewidth=.1, colors='k', zorder=100)
    ax.hlines(0, -3, 3, linewidth=.1, colors='k', zorder=100)
    ax.plot([-3, 3], [-3, 3], linewidth=.1, c='k', zorder=100)
    ax.plot([-3, 3], [3, -3], linewidth=.1, c='k', zorder=100)
    circle = plt.Circle((0, 0), 1, linewidth=.1, edgecolor='k', facecolor=None, fill=False)
    ax.add_patch(circle)
    if regimes:
        ax.text(-2.95, 0., "NAO-", size='large', weight='bold')
        ax.text(0., -2.95, "BL-", size='large', weight='bold')
        ax.text(0., 2.85, "BL+", size='large', weight='bold')
        ax.text(2.4, 0., "NAO+", size='large', weight='bold')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

    if 'savefig' in kwargs:
        plt.savefig(kwargs['savefig'])

    if 'ax' not in kwargs:
        plt.show()


"""
def plot_regimes(pivot_dt, labels, **kwargs):
    '''
    Method to plot centroids after applying clustering algorithm
    Args:
    - pivot_dt: an xarray Dataset with size time x (lat_lon)
    - labels: an array containing the labels after the clustering
    - kwargs: additional parameters such as filename for saving the figure
    '''
    vmin, vmax = pivot_dt.quantile(.05).values, pivot_dt.quantile(.95).values
    norm = mpl.colors.SymLogNorm(vmin=pivot_dt.min(), vmax=pivot_dt.max(), linthresh=20, base=10)

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=projection))
    nb_clusters = max(labels) + 1
    fig = plt.figure(figsize=(15, 7.5 * math.ceil(nb_clusters / 2)))
    axs = AxesGrid(fig, 111, axes_class=axes_class, nrows_ncols=(math.ceil(nb_clusters / 2), 2), axes_pad=1,
                   label_mode='')
    # cbar_mode='single',cbar_location='bottom',cbar_pad=0.1)
    if (nb_clusters % 2) != 0:
        axs[-1].remove()

    for cluster_nb in range(nb_clusters):
        centroid_df = pivot_dt.isel(time=labels == cluster_nb).mean(dim='time')
        cset = plot_gph(centroid_df, reshape=True, ax=axs[cluster_nb], title='Cluster {}'.format(cluster_nb),
                        norm=norm, **kwargs)
    cax = fig.add_axes([0.26, 0.25, 0.5, 0.02])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap('RdBu_r'), norm=norm, spacing='uniform',
                                   orientation='horizontal', extend='both', extendfrac='auto',
                                   ticklocation='bottom', format="%d",
                                   ticks=[-200., -50., -15, 0., 15., 50., 200.])
    plt.show()
"""


def plot_regimes(pivot_dt, labels, **kwargs):
    '''
    Method to plot centroids after applying clustering algorithm
    Args:
    - pivot_dt: an xarray Dataset with size time x (lat_lon)
    - labels: an array containing the labels after the clustering
    - kwargs: additional parameters such as filename for saving the figure
    '''
    xlim, ylim = (pivot_dt['longitude'].values.min(), pivot_dt['longitude'].values.max()), \
                 (pivot_dt['latitude'].values.min(), pivot_dt['latitude'].values.max())
    trans = ccrs.PlateCarree()
    proj = ccrs.LambertConformal((xlim[1] + xlim[0]) / 2, (ylim[1] + ylim[0]) / 2)

    pivot_dt = pivot_dt.unstack('latlon').assign_coords(labels=('time', labels))
    pivot_dt = pivot_dt.groupby('labels').mean()
    vmin, vmax = pivot_dt.quantile(.05).values, pivot_dt.quantile(.95).values

    cset = pivot_dt.plot.contourf(x="longitude", y="latitude", col="labels", levels=50, col_wrap=2,
                                  transform=trans, alpha=.7, cmap="RdBu_r", robust=True,
                                  norm=mpl.colors.SymLogNorm(vmin=vmin, vmax=vmax,
                                                             linthresh=pivot_dt.quantile(.85).values, base=10),
                                  subplot_kws=dict(projection=proj), figsize=(15, 15),
                                  cbar_kwargs=dict(
                                      orientation='horizontal' if len(np.unique(labels)) <= 4 else "vertical",
                                      extend='both', ticks=[vmin, vmax], format="%d"))

    for i, ax in enumerate(cset.axes.flat):
        ax.coastlines()
        rect = mpl.path.Path([[xlim[0], ylim[0]],
                              [xlim[1], ylim[0]],
                              [xlim[1], ylim[1]],
                              [xlim[0], ylim[1]],
                              [xlim[0], ylim[0]],
                              ]).interpolated(20)
        proj_to_data = trans._as_mpl_transform(ax) - ax.transData
        rect_in_target = proj_to_data.transform_path(rect)
        ax.set_boundary(rect_in_target)
        ax.set_extent([xlim[0], xlim[1], ylim[0] - 10, ylim[1]])

        # ax.set_title(f"Cluster-{i+1}",y = -0.2 if i>1 else 1)
        ax.set_title("")

    if 'savefig' in kwargs:
        plt.savefig("../imgs/" + kwargs['savefig'])
    plt.show()

def plot_elbow(anomaly, **kwargs):
    '''
    Method to show the elbow of the sum of squared distance, to evaluate the correct number of clusters to be adopted
    Args:
    - anomaly: a pandas DataFrame containing the historical series of anomalies
    '''
    if 'ax' not in kwargs:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    else:
        ax = kwargs['ax']

    inertias = [extract_regimes(anomaly, method='kmeans', nb_regimes=k)[1] for k in range(1, 8)]

    ax.plot(range(1, 8), inertias, marker='o', linewidth=1.2)
    # set text box to illustrate the position of 5 clusters, as Cassou et al

    ax.set_xlabel("Number of regimes")
    ax.set_ylabel("Inertia")
    if 'savefig' in kwargs:
        plt.savefig("Elbow.png")

    if 'ax' not in kwargs:
        plt.show()


def plot_silhouette(anomaly_train, anomaly_test, method='kmeans', **kwargs):
    '''
    Method to show the silhouette score, to evaluate the correct number of clusters to be adopted
    Args:
    - anomaly_train: a pandas DataFrame containing the historical series of anomalies
    - anomaly_test: a pandas DataFrame containing the historical series of anomalies
    - method: the clustering method adopted to evaluate the labels
    '''
    if 'ax' not in kwargs:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    else:
        ax = kwargs['ax']

    scores = [silhouette_score(anomaly_test,
                               extract_regimes(anomaly_train, method=method,
                                               nb_regimes=k, test=anomaly_test)) for k in range(2, 8)]

    ax.plot(range(2, 8), scores, marker='o', linewidth=1.2)
    # set text box to illustrate the position of 5 clusters, as Cassou et al

    ax.set_xlabel("Number of regimes")
    ax.set_ylabel("Silhouette score")
    if 'savefig' in kwargs:
        plt.savefig("Silhouette.png")
    if 'ax' not in kwargs:
        plt.show()


def plot_elbo(anomaly, **kwargs):
    '''
    Method to show the ELBO (evidence lower bound), to evaluate the correct number of clusters to be adopted
    Args:
    - anomaly: a pandas DataFrame containing the historical series of anomalies
    '''

    if 'ax' not in kwargs:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    else:
        ax = kwargs['ax']

    elbos = [extract_regimes(anomaly, method='bayesian_gmm', nb_regimes=k)[1] for k in range(1, 8)]

    ax.plot(range(1, 8), elbos, marker='o', linewidth=1.2)
    # set text box to illustrate the position of 5 clusters, as Cassou et al

    ax.set_xlabel("Number of regimes")
    ax.set_ylabel("ELBO")
    if 'savefig' in kwargs:
        plt.savefig("Elbo.png")

    if 'ax' not in kwargs:
        plt.show()


def plot_score(anomaly_train, anomaly_test, method='kmeans', **kwargs):
    '''
    Method to plot the score of a method (as defined by .score() method of sklearn), to evaluate the correct number of clusters to be adopted
    Args:
    Args:
    - anomaly_train: a pandas DataFrame containing the historical series of anomalies
    - anomaly_test: a pandas DataFrame containing the historical series of anomalies
    - method: the clustering method adopted to evaluate the labels
    '''

    if 'ax' not in kwargs:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    else:
        ax = kwargs['ax']

    scores = [extract_regimes(anomaly_train, method=method, nb_regimes=k)[-1].score(anomaly_test) for k in range(1, 8)]

    ax.plot(range(1, 8), scores, marker='o', linewidth=1.2)
    # set text box to illustrate the position of 5 clusters, as Cassou et al

    ax.set_xlabel("Number of regimes")
    ax.set_ylabel("Score")
    if 'savefig' in kwargs:
        plt.savefig("Score.png")

    if 'ax' not in kwargs:
        plt.show()


def plot_mixtures(X, Y, means, covariances, **kwargs):
    '''
    Method to display mixture distributions as estimated from a mixture model
    Args:
    - means: the mean vectors of the N mixtures
    - covariances: the covariance matrices of the N mixtures
    '''
    n = means.shape[1]
    s = n - 1
    fig, axs = plt.subplots(s, s, figsize=(10 * s, 10 * s))
    _x, _y = -1, -1
    for i in range(s):
        _x += 1
        _y = -1
        for j in range(s):
            _y += 1
            if _x == _y:
                _y += 1
            mean = means[:, [_x, _y]]
            covariance = covariances[:, [_x, _y]][:, :, [_x, _y]]

            for c, (m, var) in enumerate(zip(mean, covariance)):
                if not np.any(Y == c):
                    continue
                else:

                    xs = X[:,[_x, _y]]
                    xs = xs[Y==c]
                    sc = axs[i,j].scatter(xs[:,0], xs[:,1], edgecolor='k', s=30, label = 'Mixture {}'.format(c))
                    '''
                    xx, yy = np.meshgrid(xs[:,0], xs[:,1])
                    Xplot = np.vstack((xx.flatten(), yy.flatten())).T
                    preds = scipy.stats.multivariate_normal(mean = m, cov = var).pdf(Xplot)
                    #print(xx.shape, Xplot.shape, preds.shape, preds.reshape(*xx.shape).shape)

                    Multivariate distributions

                    cs = axs[i,j].contour(xs[:,0], xs[:,1], preds.reshape(*xx.shape), levels = np.linspace(0,1,10),
                                          colors=sc.get_facecolors(), linewidths=1.2, zorder=0, alpha =0.3)
                    '''

                    '''
                    Contour levels

                    levels = [0, 0.5, 0.75, 0.9, 1]
                    cs = axs[i,j].contour(xx,yy, preds.reshape(*xx.shape), levels,
                                          colors='k', linewidths=1.8, zorder=100)
                    axs[i,j].clabel(cs, inline=1)
                    cs = axs[i,j].contourf(xx, yy, preds.reshape(*xx.shape), levels,
                                    cmap='Purples_r', linewidths=0, zorder=0, alpha=.5)
                    '''

                    # '''
                    # Ellipses

                    v, w = np.linalg.eigh(var)
                    v = 2. * np.sqrt(2.) * np.sqrt(v)
                    u = w[0] / np.linalg.norm(w[0])

                    angle = np.arctan(u[1] / u[0])
                    angle = 180. * angle / np.pi  # convert to degrees
                    ell = mpl.patches.Ellipse(m, v[0], v[1], 180. + angle, color=sc.get_facecolors()[0])
                    ell.set_clip_box(axs[i, j].bbox)
                    ell.set_alpha(0.5)
                    axs[i, j].add_artist(ell)
                    # '''

            axs[i, j].legend(loc='upper right')
            axs[i, j].text(0.05, 0.85, 'Component {} vs. {}'.format(_x, _y), style='italic',
                           transform=axs[i, j].transAxes, fontsize=14, family='fantasy',
                           bbox={'facecolor': 'tomato', 'alpha': .8, 'pad': 5})

    plt.tight_layout()
    if 'savefig' in kwargs:
        plt.savefig("Elbo.png")
    plt.show()


def plot_KL(means, covariances):
    '''
    Method to plot the KL-divergences between the generative distributions of the clusters
    Args:
    - means: the mean vectors of the N mixtures
    - covariances: the covariance matrices of the N mixtures
    '''
    kls = np.zeros((means.shape[0], means.shape[0]))

    for i in range(means.shape[0]):
        for j in range(means.shape[0]):
            kls[i, j] = 0.5 * (
                    np.log(np.linalg.det(covariances[j]) / np.linalg.det(covariances[i])) - means.shape[1] +
                    np.matrix.trace(np.linalg.inv(covariances[j]).dot(covariances[i])) +
                    (means[j] - means[i]).T.dot(np.linalg.inv(covariances[j])).dot(means[j] - means[i])
            )

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    sns.heatmap(kls, cmap='coolwarm', annot=True, cbar=True, ax=ax)
    plt.show()