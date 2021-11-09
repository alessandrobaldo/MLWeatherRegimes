import numpy as np
import pandas as pd
import os
import itertools
import streamlit as st
from datetime import timedelta, datetime
path = "energy_imgs"

@st.cache
def load_predictions(filename = 'predictions_VAE.csv'):
    '''
    Method to load the models' predictions from file
    Args:
        filename: the name of the file containing the predictions

    Returns:
        A pandas DataFrame containing the predictions

    '''
    predictions = pd.read_csv(f'W:/UK/Research/Private/WEATHER/STAGE_ABALDO/scripts/predictions/{filename}.csv',
                              header=[0, 1], index_col=0)

    for method in predictions.columns.get_level_values(0).unique():
        predictions[(method, 'Prediction')] = predictions.xs(method, axis=1).apply(
            lambda x: predictions.xs(method, axis=1).columns[np.argmax(x)], axis=1)
    predictions.index = pd.to_datetime(predictions.index, dayfirst=True)

    return predictions

@st.cache
def load_pcs():
    '''
    Method to load the Principal Components of the weather regimes
    Returns:
        A pandas DataFrame contaning the Principal Components of the weather regimes

    '''
    pcs = pd.read_csv('W:/UK/Research/Private/WEATHER/STAGE_ABALDO/scripts/files/pcs.csv', header=0, index_col=0)
    pcs.index = pd.to_datetime(pcs.index)
    return pcs

@st.cache
def filter_by_preds(df, predictions, model):
    '''
    Method to incorporate predictions inside a DataFrame of energy variables for the common dates
    Args:
        df: a pandas DataFrame containing data on energy variables
        predictions: a pandas DataFraem containing predictions of historical daily weather regimes
        model: a str indicating the model whose predictions are considered

    Returns:
        a pandas DataFrame of energy variables with the related regime associated for each of the date

    '''
    preds = predictions.xs('Prediction', level=1, axis=1)
    # accurate_preds = preds[preds.eq(preds.iloc[:, 0], axis=0).all(1)].apply(lambda x: x.unique()[0], axis=1)
    accurate_preds = preds[model]
    df = df.loc[df.index.intersection(accurate_preds.index)]
    df['Regime'] = accurate_preds[df.index.intersection(accurate_preds.index)]

    return df

@st.cache
def filter_by_country(df, country):
    '''
    Method to filter a DataFrame of energy variables by country
    Args:
        df: a pandas DataFrame containing energy-variables, with a MultiIndex whose level 0 contains the reference to the countries
        country: a str indicating which country to query for

    Returns:
        a pandas DataFrame filtered by country

    '''
    country_df = df.xs(country, level=0).copy()
    country_df.index = pd.to_datetime(country_df.index)
    country_df.dropna(how='all', inplace=True, axis=1)
    country_df.dropna(how='all', inplace=True, axis=0)

    return country_df

@st.cache(allow_output_mutation=True)
def filter_forecast(subseasonal_df, date, backward = True):
    '''
    A method to filter a DataFrame containing sub-seasonal forecasts
    Args:
        subseasonal_df: a pandas DataFrame containing an historical series of sub-seasonal forecasts
        date: the date to filter by
        backward: boolean indicating if filtering in backward mode (i.e. fixing the date as the forecasted date) or not (i.e. fixing the date as the forecasting date)

    Returns:
        a pandas DataFrame with the selected sub-seasonal forecasts

    '''
    if backward:
        forecast = subseasonal_df.loc[pd.IndexSlice[:, date.strftime("%Y-%m-%d"), :], :].copy()
        forecast.reset_index(level=[1, 2], drop=True, inplace=True)
        forecast = forecast.resample("W-MON", label='left', closed='left').agg('last')
    else:
        def nearest(index, date):
            return min(index, key=lambda x: abs(x.date()-date))
        forecast = subseasonal_df.loc[pd.IndexSlice[nearest(subseasonal_df.index.get_level_values(0), date).strftime("%Y-%m-%d"), :, :], :].copy()
        forecast.reset_index(level=[0, 2], drop=True, inplace=True)
    forecast = forecast.reindex(sorted(forecast.columns), axis=1)
    forecast = forecast.apply(lambda x: x + x['Unknown'] * x if x['Unknown'] != 1 else 0.25, axis=1, result_type = 'broadcast')
    forecast = forecast.drop('Unknown', axis=1).apply(lambda x: x / x.sum(), axis=1)
    return forecast

@st.cache
def get_state_transitions(predictions, window=1):
    '''
    A method to obtain the transition robabilities associated to the weather regimes predictions
    Args:
        predictions: a pandas DataFrame containing predictions of historical daily weather regimes
        window: the minimum window in number of days to consider a transition valid

    Returns:
        A pandas DataFrame containing the transition probabilities between each pair of regimes, under each model

    '''
    stats = pd.DataFrame(columns=list(itertools.product(('NAO+', 'NAO-', 'SB', 'AR'), ('NAO+', 'NAO-', 'SB', 'AR'))),
                         index=['K-Means', 'Bayesian-GMM', 'GMM']
                         )
    stats.columns = pd.MultiIndex.from_tuples(stats.columns)
    stats.fillna(value=0, inplace=True)

    for method in stats.index:
        tmp = predictions[method].copy()
        # tmp['Prediction'] = tmp.apply(lambda x: tmp.columns[np.argmax(x)], axis = 1)
        tmp['Pred'] = tmp['Prediction'].map({"NAO+": 0, "NAO-": 1, "SB": 2, "AR": 3})
        tmp['mask'] = tmp['Pred'].shift() - tmp['Pred'] == 0
        tmp['inv_mask'] = ~tmp['mask']
        tmp['cumsum'] = tmp['inv_mask'].cumsum()
        dates = []
        for i in range(1, tmp['cumsum'].max() + 1):
            tmp2 = tmp[tmp['cumsum'] == i]
            if len(tmp2) >= window:
                for j in range(window - 1, len(tmp2)):
                    dates.append(tmp2.index[j])

        for i, (ind, row) in enumerate(tmp.loc[dates].iterrows()):
            if i > 0:
                pred = row['Prediction']
                prec = prev_row['Prediction']
                stats.at[method, (prec, pred)] += 1
            prev_row = row

    for regime in stats.columns.get_level_values(0):
        for method in stats.index:
            stats.loc[method, (regime, slice(None))] /= stats.loc[method].xs(regime, level=0).sum()
    stats = stats.round(2)

    return stats

@st.cache
def load_measurements():
    '''
    Method to load the true measurements of energy variable from the repository of observations
    Returns:
        A pandas DataFrame containing the historical series of measurements, indexed by date and country

    '''
    print(os.listdir('.'))
    if "measurements.csv" not in os.listdir('./dashboard'):
        READ_PATH = 'P:/CH/Weather Data/METEOLOGICA/OBSERVATIONS'
        COUNTRIES = ['ES', 'FR', 'GE', 'UK' , 'NE', 'IT']
        TYPE = ["LOAD", "PRICE", "WIND", "SOLAR", "HYDRO"]
        YEARS = list(map(str, range(2011, 2021)))

        true_df = pd.DataFrame()
        for country in os.listdir(READ_PATH):
            if country in COUNTRIES:
                for year in YEARS:
                    for typeof in TYPE:
                        try:
                            cols = ["Start", "End"]

                            if typeof == "WIND":
                                if country == "UK":
                                    cols += ["Wind Embedded Capacity", "Wind Capacity", "Wind Embedded Obs", "Wind Obs"]
                                elif country in ["NE", "IT"]:
                                    cols = ["Start", "Wind Obs", "Wind Capacity"]
                                else:
                                    cols += ["Wind Capacity", "Wind Obs"]


                            elif typeof == "SOLAR":
                                if country == "UK":
                                    cols += ['Solar Photo Capacity', '#solar', 'Solar Photo Obs']
                                elif country in ["NE", "IT"]:
                                    cols = ["Start", "Solar Photo Obs", "Solar Photo Capacity"]
                                else:
                                    cols += ['Solar Photo Capacity', 'Solar Photo Obs']

                                if country == "ES":
                                    cols += ['Solar Thermal Capacity', 'Solar Thermal Obs']


                            elif typeof == "LOAD":
                                if country not in ["NE", "IT"]:
                                    cols += ["Load"]
                                else:
                                    cols = ["Start", "Load"]


                            elif typeof == "HYDRO":
                                if country == "ES":
                                    cols += ["#hydro1", "#hydro2"]

                                if country == "IT":
                                    cols = ["Start", "Hydro RoR Obs", "Hydro Reservoir Obs"]
                                else:
                                    cols += ['Hydro RoR Obs', 'Total Hydro Obs']


                            else:
                                cols += ["Price"]

                            temp_df = pd.read_csv(
                                "/".join([READ_PATH, country, '_'.join([country, typeof, year])]) + '.csv', sep=";",
                                # index_col = [0,1],
                                skiprows=range(0, 7), header=0)
                            temp_df.columns = cols
                            temp_df.drop([label for label in temp_df.columns if '#' in label], axis=1, inplace=True)
                            temp_df['Country'] = country
                            temp_df.set_index(["Start", "Country"], inplace=True)

                            if country in ["IT", "NE"]:
                                temp_df[
                                    [col for col in temp_df.columns if
                                     any(el in col for el in ["Obs", "Capacity", "Load"])]] /= 1000
                            if country == "IT" and typeof == "HYDRO":
                                temp_df['Total Hydro Obs'] = temp_df['Hydro RoR Obs'] + temp_df['Hydro Reservoir Obs']

                            true_df = true_df.append(temp_df)

                        except Exception as e:
                            print(e)
                            print("Corrupted or not existing file: {}".format('_'.join([country, typeof, year]) + '.csv'))

        true_df = true_df.reset_index().set_index('Start')
        true_df.index = pd.to_datetime(true_df.index)
        true_df = true_df.groupby('Country').resample('D').mean()

        nan_mask = true_df['Wind Capacity'].isnull() & true_df['Wind Embedded Capacity'].isnull()
        true_df['Total Wind Capacity'] = true_df['Wind Capacity'].fillna(0) + true_df['Wind Embedded Capacity'].fillna(0)
        true_df['Total Wind Capacity'][nan_mask] = np.nan

        nan_mask = true_df['Wind Obs'].isnull() & true_df['Wind Embedded Obs'].isnull()
        true_df['Total Wind Obs'] = true_df['Wind Obs'].fillna(0) + true_df['Wind Embedded Obs'].fillna(0)
        true_df['Total Wind Obs'][nan_mask] = np.nan

        nan_mask = true_df['Solar Photo Capacity'].isnull() & true_df['Solar Thermal Capacity'].isnull()
        true_df['Total Solar Capacity'] = true_df['Solar Photo Capacity'].fillna(0) + true_df['Solar Thermal Capacity'].fillna(0)
        true_df['Total Solar Capacity'][nan_mask] = np.nan

        nan_mask = true_df['Solar Photo Obs'].isnull() & true_df['Solar Thermal Obs'].isnull()
        true_df['Total Solar Obs'] = true_df['Solar Photo Obs'].fillna(0) + true_df['Solar Thermal Obs'].fillna(0)
        true_df['Total Solar Obs'][nan_mask] = np.nan

        true_df['Hydro Reservoir Obs'] = true_df['Total Hydro Obs'] - true_df['Hydro Reservoir Obs']

        true_df['Wind Load Factor'] = true_df['Total Wind Obs'] / true_df['Total Wind Capacity']
        true_df['Solar Load Factor'] = true_df['Total Solar Obs'] / true_df['Total Solar Capacity']
    else:
        true_df = pd.read_csv("dashboard/measurements.csv", index_col =[0,1], parse_dates = True)
    return true_df

@st.cache
def load_synthetic_data():
    '''
    Method to load the synthetic data measurements of energy variables
    Returns:
        A pandas DataFrame containing the synthetic measurements of energy variables, indexed by date and country

    '''
    if 'synthetic.csv' not in os.listdir('./dashboard'):
        READ_PATH = 'W:/UK/Reserach/Private/WEATHER/STAGE_ABALDO/dataset/Energy_Indicators_Copernicus'
        FOLDERS = ['dataset-sis-energy-derived-reanalysis_ENERGY', 'dataset-sis-energy-derived-reanalysis_WEATHER']
        COUNTRIES = ['BE', 'ES', 'FR', 'DE', 'IT', 'NL', 'UK']
        TYPE = {'dataset-sis-energy-derived-reanalysis_ENERGY': {
            "EDM_PWR": "Load", "HydroReservoir_CFR": "Hydro Load Factor", "HydroReservoir_PWR": "Hydro Reservoir Obs",
            "HydroRunOfRiver_CFR": "Hydro RoR Load Factor", "HydroRunOfRiver_PWR": "Hydro RoR Obs",
            "PV_CFR": "Solar Photo Load Factor", "PV_PWR": "Solar Photo Obs",
            "WindOffshore_CFR": "Wind Offshore Load Factor", "WindOffshore_PWR": "Wind Offshore Obs",
            "WindOnshore_CFR": "Wind Onshore Load Factor", "WindOnshore_PWR": "Wind Onshore Obs"
        },
            'dataset-sis-energy-derived-reanalysis_WEATHER': {
                "AirTemp_2m": "Temperature", "GHI": "Irradiance", "MeanSeaLevel": "Sea Level Pressure",
                "TotalPrecip": "Precipitation", "WindSpeed_10m": "Wind Speed (10m)", "WindSpeed_100m": "Wind Speed (100m)"
            }
        }
        YEARS = "1979-2021"
        synthetic_df = pd.DataFrame(columns=[v for d in TYPE.values() for v in d.values()])

        for folder in FOLDERS:
            for file in TYPE[folder]:
                tmp = pd.read_csv(os.path.join(READ_PATH, folder, file + "_" + YEARS + ".csv"),
                                  comment='#', index_col=0, usecols=['Date'] + COUNTRIES)
                tmp.index = pd.to_datetime(tmp.index)
                # tmp = tmp[tmp.index.month.isin([12,1,2])]
                tmp.reset_index(inplace=True)
                tmp = pd.melt(tmp, id_vars='Date', var_name='Country', value_vars=COUNTRIES, value_name=TYPE[folder][file])
                tmp['Country'] = tmp['Country'].map(
                    {'DE': 'GE', 'NL': 'NE', 'IT': 'IT', 'UK': 'UK', 'FR': 'FR', 'BE': 'BE', 'ES': 'ES'})
                tmp.set_index(['Country', 'Date'], inplace=True)
                if synthetic_df.empty:
                    synthetic_df = tmp.copy()
                else:
                    synthetic_df = synthetic_df.append(tmp)

        synthetic_df = synthetic_df.max(level=[0, 1]).dropna(how='all', axis=1)
        synthetic_df[['Load', 'Hydro Reservoir Obs', 'Hydro RoR Obs', 'Solar Photo Obs', 'Wind Offshore Obs',
                      'Wind Onshore Obs']] /= 1000
        synthetic_df['Temperature'] -= 273
        nan_mask = synthetic_df['Wind Offshore Obs'].isnull() & synthetic_df['Wind Onshore Obs'].isnull()
        synthetic_df['Total Wind Obs'] = synthetic_df['Wind Offshore Obs'].fillna(value=0) + \
                                         synthetic_df['Wind Onshore Obs'].fillna(value=0)
        synthetic_df['Total Wind Obs'][nan_mask] = np.nan
        synthetic_df['Total Wind Capacity'] = synthetic_df['Wind Offshore Obs'].fillna(value=0) / synthetic_df[
            'Wind Offshore Load Factor'] + \
                                              synthetic_df['Wind Onshore Obs'].fillna(value=0) / synthetic_df[
                                                  'Wind Onshore Load Factor']
        synthetic_df['Total Wind Capacity'][nan_mask] = np.nan
        synthetic_df['Wind Load Factor'] = synthetic_df['Total Wind Obs'] / synthetic_df['Total Wind Capacity']
        synthetic_df['Solar Load Factor'] = synthetic_df['Solar Photo Load Factor']
        synthetic_df['Total Solar Capacity'] = synthetic_df['Solar Photo Obs'] / synthetic_df['Solar Load Factor']
        synthetic_df['Total Solar Obs'] = synthetic_df['Solar Photo Obs']
    else:
        synthetic_df = pd.read_csv("dashboard/synthetic.csv", index_col = [0,1], parse_dates = True)
    return synthetic_df

@st.cache
def build_data(true_df, synthetic_df):
    '''
    Method to merge the true and synthetic observation of energy variables
    Args:
        true_df: a pandas DataFrame containing the true measurements
        synthetic_df: a pandas DataFrame containing the synthetic measurements

    Returns:
        A pandas DataFrame containing the merged true and synthetic measurements. For the common dates, the true reference overwrites the synthetic one

    '''
    df = synthetic_df.copy()
    df.loc[true_df.index, true_df.columns.intersection(df.columns)] = true_df
    other_cols = [c for c in true_df.columns if c not in true_df.columns.intersection(df.columns)]
    df = pd.concat([df, true_df[other_cols]], axis=1)
    df.rename_axis(["Country", "Date"], inplace = True)
    return df

@st.cache
def load_subseasonal():
    '''
    Method to load the sub-seasonal forecasts from the repository
    Returns:
        A pandas DataFrame containing the sub-seasonal forecasts indexed by the forecasting and forecasted dates, and the step identifying the shift in days between the two

    '''
    path = 'W:/UK/Research/Private/WEATHER/STAGE_ABALDO/dataset/Weather_Regime_ECMWF'
    df = pd.DataFrame(columns=['Step', 'NAO+', 'SB', 'NAO-', 'AR'])
    for file in os.listdir(path):
        with open(path + '/' + file, "r") as f:
            next(f)
            date = next(f).strip()
            year, month, day = date[:4], date[4:6], date[6:8]
            date = "-".join([year, month, day])

        temp_df = pd.read_csv(path + '/' + file, skiprows=[0, 1], header=0, index_col=False, sep=':',
                              usecols=range(0, 6))
        temp_df.columns = names = ['Step', 'NAO+', 'SB', 'NAO-', 'AR', 'Unknown']
        temp_df['Step'] //= 24
        temp_df.index = [date] * len(temp_df)
        temp_df.fillna(value=0, inplace=True)

        def get_len(x):
            if isinstance(x, str):
                x = len(x.strip().split(",")) - 1
            return x

        temp_df = temp_df.apply(lambda serie: serie.apply(lambda x: get_len(x)))
        df = df.append(temp_df)
    df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Actual Date"}, inplace=True)
    df['Forecast Date'] = df.apply(lambda x: x['Actual Date'] + timedelta(days=x['Step']), axis=1)
    df.set_index(['Actual Date', 'Forecast Date', 'Step'], inplace=True)
    #df = df[np.logical_and(df.index.get_level_values(0).month.isin([1, 2, 12]),
    #                        df.index.get_level_values(1).month.isin([1, 2, 12]))]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])

    df.loc[:, ["NAO+", "SB", "AR", "NAO-", "Unknown"]] = df.loc[:, ["NAO+", "SB", "AR", "NAO-", "Unknown"]].div(
        df.sum(axis=1), axis=0)
    df.dropna(how='any', inplace=True)
    df.to_csv("subseasonal_full.csv")
    return df

@st.cache
def load_shortterm():
    '''
    Method to load the short-term forecasts from the repository
    Returns:
        A pandas DataFrame containing the short-term forecasts indexed by the forecasting and the forecasted dates

    '''
    path = 'P:/CH/Weather Data/METEOLOGICA/HISTORICAL_FORECAST/WIND/Wind_GER_2018-2020_11am.csv'
    df = pd.read_csv(path, header = 0, parse_dates=True,
                     names = ['Forecast Date','_','ECMWF_ENS','ECMWF_HRES','GFS','p10','Meteologica','p90','Capacity','Observation','Leadtime'])
    df.drop(['_','p10','p90'], inplace = True, axis = 1)
    df.dropna(subset = ['ECMWF_ENS','ECMWF_HRES','GFS','Meteologica'], inplace=True)
    df['Forecast Date'] = pd.to_datetime(df['Forecast Date'])
    df['Leadtime'] = df['Leadtime'].map(lambda x: x.split("-")[1]).astype(int)
    df = df.groupby([pd.Grouper(freq='D', key='Forecast Date'), df['Leadtime']]).agg(lambda x : x.mean(skipna=True))
    df.reset_index(inplace = True)
    df['Actual Date'] = df.apply(lambda x: x['Forecast Date'] - timedelta(days = x['Leadtime']), axis = 1)
    df.set_index(['Actual Date','Forecast Date'], inplace = True)
    df.drop('Leadtime', axis = 1, inplace = True)
    for col in ['ECMWF_ENS','ECMWF_HRES','GFS','Meteologica']:
        df[col] /= df['Capacity']

    df.sort_index(level=['Actual Date','Forecast Date'], inplace=True)
    df.to_csv("dashboard/short_term_forecasts_20182020.csv")
    return df


@st.cache
def load_MF_targets(season = 'Winter', filter_dates = False):
    '''
    Method to loas the Meteo-France historical predictions from the repository
    Args:
        filter_dates: boolean indicating whrther to maintain the dates whose predictions are coherent under both the two methods used by Meteo-France

    Returns:
        A pandas DataFrame containing predictions of historical daily weather regimes as predicted by Meteo-France

    '''

    if 'targets_MF.csv' in os.listdir('W:/UK/Research/Private/WEATHER/STAGE_ABALDO/dataset'):
        targets = pd.read_csv('W:/UK/Research/Private/WEATHER/STAGE_ABALDO/dataset/targets_MF.csv', index_col=0,
                              header=[0, 1, 2])
        targets.index = pd.to_datetime(targets.index)
        targets = targets.xs(season, axis = 1, level = 0)
        if filter_dates:
            targets = targets[targets[('Distance', 'Prediction')] == targets[('Correlation', 'Prediction')]]
    else:
        path = 'W:/UK/Research/Private/WEATHER/STAGE_ABALDO/dataset/MeteoFrance'
        dist_df, corr_df = [], []
        for file in os.listdir(path):
            if any(col in file for col in ['EQM', 'COREL']):
                temp_df = pd.read_csv(os.path.join(path, file), sep=r'\s+',
                                      index_col=None)
                col_names = {"H_ZO": ("Winter", "NAO+"), "H_AR": ("Winter", "AR"),
                             "H_EA": ("Winter", "SB"), "H_AL": ("Winter", "NAO-"),
                             "E_GA": ("Summer", "NAO-"), "E_AL": ("Summer", "AL"),
                             "E_EA": ("Summer", "SB"), "E_ZO": ("Summer", "Zonal")}
                temp_df = temp_df[col_names.keys()]
                temp_df.dropna(how='all', axis=1, inplace=True)
                temp_df.rename(columns=col_names, inplace=True)
                temp_df.index = temp_df.index.map(lambda x: datetime.strptime(str(x), "%Y%m%d"))
                if 'EQM' in file:
                    dist_df.append(temp_df)
                else:
                    corr_df.append(temp_df)
        dist_df, corr_df = pd.concat(dist_df), pd.concat(corr_df)
        dist_df.columns = pd.MultiIndex.from_tuples(dist_df.columns)
        dist_df = pd.concat([dist_df], axis=1, keys=['Distance']).swaplevel(0, 1, 1)
        corr_df.columns = pd.MultiIndex.from_tuples(corr_df.columns)
        corr_df = pd.concat([corr_df], axis=1, keys=['Correlation']).swaplevel(0, 1, 1)
        targets = pd.concat([dist_df, corr_df], axis=1)
        for targets in ['Winter', 'Summer']:
            targets[(season, 'Distance', 'Prediction')] = targets.xs(season, axis=1, level=0).xs('Distance', axis=1, level=0). \
                apply(lambda x: df.xs(season, axis=1, level=0).xs('Distance', axis=1, level=0). \
                      columns[np.argmin(x)], axis=1)
        for season in ['Winter', 'Summer']:
            targets[(season, 'Correlation', 'Prediction')] = targets.xs(season, axis=1, level=0).xs('Correlation', axis=1,
                                                                                          level=0). \
                apply(lambda x: targets.xs(season, axis=1, level=0).xs('Correlation', axis=1, level=0). \
                      columns[np.argmax(x)], axis=1)
        targets.to_csv('W:/UK/Research/Private/WEATHER/STAGE_ABALDO/dataset/targets_MF.csv')
    if season == 'Winter':
        targets = targets[targets.index.month.isin([1, 2, 12])]
    else:
        targets = targets[targets.index.month.isin([6, 7, 8])]
    targets = targets.reindex(sorted(targets.columns, key=lambda x: (x[0], x[1])), axis=1)
    return targets

@st.cache
def load_hydro():
    '''
    Method to load data about Hydro energy variables
    Returns:
        Three pandas DataFrames containing historical synthetic values of water reservoir filling, inflow and snow groundwater variables

    '''
    reservoir = pd.read_csv("dashboard/np_hydro_reservoir_water_filling_mwh_d_synthetic.csv", header = 0, parse_dates=True, index_col = 0)
    inflow = pd.read_csv("dashboard/np_hydro_inflow_mwh_d_synthetic.csv", header = 0, parse_dates=True, index_col = 0)
    groundwater = pd.read_csv("dashboard/np_hydro_snowandgroundwater_mwh_d_synthetic.csv", header = 0, parse_dates=True, index_col = 0)
    reservoir /= 1e6
    inflow /= 1e6
    groundwater /= 1e6
    return reservoir, inflow, groundwater

def create_forecast_file(df, subseasonal_df):
    today = datetime.today().date()
    forecast = filter_forecast(subseasonal_df, today, backward = False)
    old_forecast = filter_forecast(subseasonal_df, today - timedelta(days = 3), backward=False)

    distributions = pd.read_csv('moments_4regimes_EU7.csv', index_col = [0,1,2,3], header = 0)
    distribution_month = min(distributions.index.get_level_values(0), key = lambda x: abs(x - today.month))
    distributions = distributions.xs(distribution_month, level = 0)

    def create_file(df, forecast, distributions, distribution_month):
        dfs = []
        for variable in ['Wind Load Factor', 'Solar Load Factor', 'Load', 'Temperature']:
            for country in distributions.index.get_level_values(0).unique():
                print(country)
                if country != 'EU-7':
                    country_df = df.xs(country, level = 0)
                else:
                    if variable != 'Load':
                        country_df = df.groupby(df.index.get_level_values(1)).mean()
                    else:
                        country_df = df.groupby(df.index.get_level_values(1)).sum()

                variable_df = country_df.loc[country_df.index.month == distribution_month, variable]
                normal = variable_df.mean()
                variable_distribution = distributions.xs(country, level = 0).xs('mean', level = 0).loc[:, variable]
                forecasted_values = np.dot(forecast, variable_distribution)
                forecasted_values = (forecasted_values - normal) *100/ normal
                dfs.append(pd.DataFrame(dict(Date = forecast.index.values,
                        Country = [country]*len(forecast),
                        Variable = [variable]*len(forecast),
                        Anomaly = forecasted_values)))
        file = pd.concat(dfs)
        file.set_index('Date', inplace = True)
        file['Variable'] = file['Variable'].map({"Wind Load Factor": "Wind", "Solar Load Factor": "Solar", "Load":"Load", "Temperature":"Temperature"})
        return file

    file = create_file(df, forecast, distributions, distribution_month)
    old_file = create_file(df, old_forecast, distributions, distribution_month)
    file.loc[file.index.intersection(old_file.index),'Anomaly_old'] = old_file.loc[file.index.intersection(old_file.index),'Anomaly']
    file['Difference'] = file['Anomaly'] - file['Anomaly_old']
    file.to_csv(f'W:\\UK\\Research\\Private\\WEATHER\\REGULAR_MONITORING\\PBI_UPDATE\\WEATHER_REGIMES\\ARCHIVES\\forecast_{today.strftime("%Y%m%d")}.csv')
    file.to_csv(f'W:\\UK\\Research\\Private\\WEATHER\\REGULAR_MONITORING\\PBI_UPDATE\\WEATHER_REGIMES\\regime_forecast.csv')



