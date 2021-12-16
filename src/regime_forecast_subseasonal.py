from dashboard.utils.data import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("notebook", rc={"axes.labelsize":16,"axes.titlesize":20, "legend.fontsize":16})
plt.style.use('seaborn')
from dashboard.utils.website import *


### Loading Synthetic Data
synthetic_df = load_synthetic_data.__wrapped__()

###Load predictions
predictions = load_predictions.__wrapped__("predictions_winter")    #predictions_winter_VAE contains Kmeans with VAE which is not the best
REGIMES = ['AR', 'NAO+', 'NAO-', 'SB']

###Create moments regimes
#create_moments_file(synthetic_df.reset_index(level = 0), predictions, model='GMM', variables=['Wind Load Factor', 'Solar Load Factor', 'Load', 'Temperature'], filename = 'moments_4regimes_EU7_winter')

### Loading Subseasonal Forecast
subseasonal_df = load_subseasonal.__wrapped__()



### Plot subseasonal forecasts
months = [1, 2, 12]      #[1, 2, 12] for winter ; [6,7,8]  for summer
date_forecast = datetime.today().date()
month = date_forecast.month if date_forecast.month in months else min(months, key=lambda x: abs(x - date_forecast.month))

### Create Forecast File
create_forecast_file(synthetic_df, subseasonal_df, date_forecast, filename = 'moments_4regimes_EU7_winter.csv')

forecast = filter_forecast.__wrapped__(subseasonal_df, date_forecast, backward = False)

fig_weights = plot_weights_forecast.__wrapped__(forecast, date_forecast,  title = None)
plt.show()
#fig_anomalies = plot_forecast(filtered_df, forecast, None, predictions, country, variables, date_forecast, model,
#                                   title = None)
#plt.show()


