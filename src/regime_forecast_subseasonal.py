from dashboard.utils.data import *
from dashboard.utils.website import *


### Loading Synthetic Data
synthetic_df = load_synthetic_data.__wrapped__()

###Load predictions
predictions = load_predictions.__wrapped__("predictions_VAE")

### Loading Subseasonal Forecast
subseasonal_df = load_subseasonal.__wrapped__()

### Create Forecast File
create_forecast_file(synthetic_df, subseasonal_df)

### Plot subseasonal forecasts

def plot_forecast(df, forecast, short_term, predictions, countries, variables, date, model, title = None, plug = False, **kwargs):
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
        normal_mean, normal_std, weighted_dist, weighted_areas, weighted_means = get_forecast_distributions.__wrapped__(
            variable_df, forecast, predictions, model)#, thresh = .9)
        #'''
        if 'Load Factor' in variable:
            normal_mean = actualize.__wrapped__(normal_mean, countries)
            normal_std = actualize.__wrapped__(normal_std, countries)
            weighted_means = actualize.__wrapped__(weighted_means, countries)
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




country = 'GE'
model = 'GMM'
variables = ['Wind Load Factor']
date_forecast = datetime(2021, 11, 1).date()
month = date_forecast.month if date_forecast.month in [1, 2, 12] else min([1, 2, 12], key=lambda x: abs(x - date_forecast.month))

forecast = filter_forecast.__wrapped__(subseasonal_df, date_forecast, backward = False)
filtered_df = filter_by_country.__wrapped__(synthetic_df, country).loc[
        filter_by_country.__wrapped__(synthetic_df, country).index.intersection(predictions.loc[predictions.index.month == month].index)]
fig_weights = plot_weights_forecast.__wrapped__(forecast, date_forecast,  title = None)
plt.show()
fig_anomalies = plot_forecast(filtered_df, forecast, None, predictions, country, variables, date_forecast, model,
                                   title = None)
plt.show()



