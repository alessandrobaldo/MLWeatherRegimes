from dashboard.utils.data import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("notebook", rc={"axes.labelsize":16,"axes.titlesize":20, "legend.fontsize":16})
plt.style.use('seaborn')
#from dashboard.utils.website import *


### Loading Synthetic Data
synthetic_df = load_synthetic_data.__wrapped__()

###Load predictions
predictions = load_predictions.__wrapped__("predictions_VAE")
REGIMES = ['AR', 'NAO+', 'NAO-', 'SB']

### Loading Subseasonal Forecast
subseasonal_df = load_subseasonal.__wrapped__()

### Create Forecast File
create_forecast_file(synthetic_df, subseasonal_df)

def plot_weights_forecast2(forecast, date, title = None, **kwargs):
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


### Plot subseasonal forecasts
country = 'GE'
model = 'GMM'
variables = ['Wind Load Factor']
date_forecast = datetime.today().date()
month = date_forecast.month if date_forecast.month in [1, 2, 12] else min([1, 2, 12], key=lambda x: abs(x - date_forecast.month))

forecast = filter_forecast.__wrapped__(subseasonal_df, date_forecast, backward = False)
filtered_df = filter_by_country.__wrapped__(synthetic_df, country).loc[
        filter_by_country.__wrapped__(synthetic_df, country).index.intersection(predictions.loc[predictions.index.month == month].index)]
fig_weights = plot_weights_forecast2(forecast, date_forecast,  title = None)
plt.show()
#fig_anomalies = plot_forecast(filtered_df, forecast, None, predictions, country, variables, date_forecast, model,
#                                   title = None)
#plt.show()


