import os
import itertools
import pandas as pd
from dashboard.utils.plotting import *
from dashboard.utils.data import *

path = "dashboard/energy_imgs"

def plot_comparison(true_df, synthetic_df):
    try:
        os.makedirs(os.path.join(path, 'Comparison'))
    except OSError as oserr:
        print(oserr)
    columns = [c for c in true_df.columns.intersection(synthetic_df.columns)]
    tmp, tmp_synth = true_df[columns], synthetic_df[columns]
    countries = tmp.index.get_level_values(0).intersection(tmp_synth.index.get_level_values(0))
    tmp = tmp.loc[countries]
    tmp_synth = tmp_synth.loc[countries]


    st.subheader("Time series comparison")
    with st.expander('', expanded = True):
        selected = st.multiselect("Select the variables you want to plot", columns, ['Total Wind Obs', 'Total Wind Capacity'], key = "ts_vars")
        years = st.multiselect("Select an year to show", range(2011, 2022), [2020], key = "ts_years")
        st.write(
            compare_timeseries(tmp.reset_index(level=0)[tmp.reset_index(level=0).index.year.isin(years)], tmp_synth.reset_index(level=0)[tmp_synth.reset_index(level=0).index.year.isin(years)],
                               selected, ['True', 'Synthetic'],
                               os.path.join(path, 'Comparison', 'energy_vars_ts.png')))

        st.caption(f"Comparison of the time-series of the {','.join(selected)} variables for synthetic data (1979-2021) and true measurements (2011-2021)")

        st.write(
            compare_timeseries(tmp.reset_index(level=0)[tmp.reset_index(level=0).index.year.isin(years)], tmp.diff().reset_index(level=0)[tmp.reset_index(level=0).index.year.isin(years)],
                               selected,
                               ['True', 'Correlated'],
                               os.path.join(path, 'Comparison', 'energy_vars_ts_correlated.png')))
        st.caption(f"Detrended true time-series  of the {','.join(selected)} variables")

    st.subheader("Distribution comparison")

    with st.expander('', expanded = True):
        period = st.radio("Compare:", ('Same Period','All History'))
        if period == 'Same Period':
            idx = tmp.index
            print(len(idx))
        else:
            idx = tmp_synth.index
            print(len(idx))

        selected = st.multiselect("Select the variables you want to plot", columns, ["Total Wind Obs", "Total Wind Capacity", "Total Solar Obs", "Total Solar Capacity"], key="distributions_vars")
        st.write(
            compare_distributions(tmp, tmp_synth.loc[idx], selected, ['True', 'Synthetic'],
                                  os.path.join(path, 'Comparison', 'energy_vars_distribution.png')))
        st.caption(f"Distribution of the {','.join(selected)} variables across the synthetic (orange) and true measurements (blue) datasets")

        st.write(
            compare_distributions_by_country(tmp, tmp_synth.loc[idx], selected, ['True', 'Synthetic'],
                                             os.path.join(path, 'Comparison', 'energy_vars_distribution_by_country.png')))
        st.caption(f"Distribution of the {','.join(selected)} variables divided by countires across the synthetic (orange) and true measurements (blue) datasets")

        stats = pd.DataFrame(columns=pd.MultiIndex.from_tuples(
            itertools.product(['Mean', 'Std', 'Skewness', 'Kurtosis'], ['True', 'Synthetic'])),
            index=selected)
        for col in stats.index:
            for dtf, ind in zip([tmp, tmp_synth.loc[idx]], ['True', 'Synthetic']):
                val = dtf[col]
                mean = val.mean(skipna=True)
                stats.loc[col, ('Mean', ind)] = mean
                std = val.std(skipna=True)
                stats.loc[col, ('Std', ind)] = std
                stats.loc[col, ('Skewness', ind)] = ((val ** 3).mean(skipna=True) - 3 * mean * (
                        std ** 2) - mean ** 3) / std ** 3
                stats.loc[col, ('Kurtosis', ind)] = ((val - mean) ** 4).mean(skipna=True) / std ** 4

        stats.to_csv("stats.csv")
        st.write(stats)
        st.caption("The four statistical moments of the synthetic-data and true-data distributions")

def plot_EU7(df, predictions, pcs):
    try:
        os.makedirs(os.path.join(path, 'EU-7'))
    except OSError as oserr:
        print(oserr)
    columns = df.columns

    st.subheader('Distributions')
    with st.expander('', expanded = True):
        selected = st.multiselect("Select the variables you want to plot", columns, ["Total Wind Obs", "Total Wind Capacity", "Total Solar Obs", "Total Solar Capacity"], key="distribution_vars")
        st.write(
            plot_distribution(df, selected,
                          os.path.join(path, 'EU-7', 'energy_vars_distribution.png')))
        st.caption(f"Distribution of {','.join(selected)} variables for EU-7")

    st.subheader('Monthly trends')
    with st.expander('', expanded = True):
        selected = st.multiselect("Select the variables you want to plot", columns, ['Wind Load Factor', 'Solar Load Factor', 'Load'], key="trends_vars")
        st.write(
            plot_bands(df, df.index.get_level_values(1).month, selected,
                   'Month', '', os.path.join(path, 'EU-7', 'bands_energy_vars.png')))
        st.caption(f"Monthly trends of the {','.join(selected)} variables")

    st.subheader('Variable vs. Variable')
    with st.expander('', expanded = True):
        col1, col2 = st.columns(2)
        x_col = col1.selectbox("Select the x variable", columns)
        y_col = col2.selectbox("Select the y variable", columns)
        col1.write(
            plot_scatter(df[df.index.get_level_values(1).map(pd.tseries.offsets.BDay().onOffset)], x_col, y_col,
                         os.path.join(path, 'EU-7', f'{x_col}_vs_{y_col}_bdays.png'),
                         c=df[df.index.get_level_values(1).map(pd.tseries.offsets.BDay().onOffset)][x_col],
                         cmap='RdBu_r')
        )
        col1.caption(f"Distribution {x_col} vs. {y_col} on Business Days")

        col2.write(
            plot_scatter(df[[not elem for elem in df.index.get_level_values(1).map(pd.tseries.offsets.BDay().onOffset)]],
                         x_col, y_col, os.path.join(path, 'EU-7', f'{x_col}_vs_{y_col}_holidays.png'),
                         c=df[[not elem for elem in df.index.get_level_values(1).map(pd.tseries.offsets.BDay().onOffset)]][x_col], cmap='RdBu_r')
        )
        col2.caption(f"Distribution {x_col} vs. {y_col} on Day Offs")

    st.subheader("Maps")
    with st.expander('', expanded=True):
        selected_date = st.selectbox("Select one date", df.index.get_level_values(1), format_func = lambda x: x.strftime("%d, %B, %Y"), key="maps_dates")
        selected_cols = st.multiselect("Select the variable(s)", columns, ['Load'], key="maps_columns")

        st.write(
            plot_maps(df.reset_index(level=0), 'EU-7', selected_cols, selected_date,
                      os.path.join(path, 'EU-7', f'maps_{"_".join(selected_cols)}.png')))

    st.subheader("Regimes")
    with st.expander('', expanded = True):
        model = st.selectbox("Select the model", predictions.columns.get_level_values(0).unique(), index = 2)
        try:
            os.makedirs(os.path.join(path, 'EU-7', model))
        except OSError as oserr:
            print(oserr)

        eu_df = filter_by_preds(df.reset_index(level=0), predictions, model)
        selected = st.multiselect("Select one or more variables", columns, ['Wind Load Factor', 'Solar Load Factor', 'Load'], key = "regimes")

        for sel, statistic in zip(selected, get_distribution_by_regime(eu_df, selected, predictions, model)):
            st.write(statistic)
            statistic.to_csv(f"{sel}.csv")
            st.caption(f"Moments of the distribution of the {sel} variable under regimes")
        st.write(
            plot_distribution_by_regime_days(eu_df, selected, predictions, model,
                                             os.path.join(path, 'EU-7', model,f'distributions_{"_".join(selected)}_regimes_days.png')))
        st.caption(f"Distribution of the {','.join(selected)} variables under regimes on Business Days (left) and Day Offs (right)")

        st.write(
            plot_distribution_by_regime_monthly(eu_df, selected,
                                            predictions, model, os.path.join(path, 'EU-7', model,f'distributions_monthly_{"_".join(selected)}_regimes.png')))
        st.caption(f"Distribution of the {','.join(selected)} variables under regimes at a monthly level")

        for sel, table in zip(selected, get_extreme_events(eu_df, selected, predictions, model)):
            st.write(table)
            st.caption(f"Extreme events of the {sel} variable on EU-7 countries")

        '''
        col1, col2 = st.columns(2)
        x_col = col1.selectbox("Select the x variable", columns, key = "regimes_densitiy_x")
        y_cols = col2.multiselect("Select the y variable(s)", [c for c in columns if c!= x_col], ['Wind Load Factor', 'Solar Load Factor'], key = "regimes_density_y")

        
        st.write(
            plot_densities_vs_variable(eu_df, x_col, y_cols,
                                       os.path.join(path, 'EU-7', model, f'densities_{x_col}_{"_".join(y_cols)}_regimes.png')))
        st.caption(f"Probability densities of {','.join(y_cols)} variables against {x_col} variable by regime")
        '''

        selected = st.multiselect("Select the variables you want to plot", columns, ['Wind Load Factor', 'Solar Load Factor'], key = "boxplot")
        st.write(
            boxplot(eu_df, ['Regime'], selected,
                    os.path.join(path, 'EU-7', model, f'boxplots_{"_".join(selected)}_regimes.png'))
        )
        st.caption("Boxplots of variables under regimes")

        st.write(
            boxplot_monthly(eu_df, ['Regime', 'Month'], selected,
                            os.path.join(path, 'EU-7', model, f'boxplots_{"_".join(selected)}_regimes_monthly.png'))
        )
        st.caption("Monthly Boxplots of variables under regimes")

        monthly_stats = eu_df.groupby([eu_df.index.month, 'Regime']).mean()
        st.dataframe(monthly_stats)

        selected = st.multiselect("Select the variables you want to plot", columns, ['Wind Load Factor', 'Solar Load Factor'], key = "barplot")
        st.write(
            plot_bars(monthly_stats, selected,
                      os.path.join(path, 'EU-7', model, f'barplot_{"_".join(selected)}_regimes_monthly.png'))
        )

        st.caption(f"Bar plots of the {','.join(selected)} variables under regimes")

        ### SCATTER
        regimes = st.multiselect("Select one or more regimes", eu_df['Regime'].unique().tolist(), eu_df['Regime'].unique().tolist(), key = "regimes_select")
        col1, col2 = st.columns(2)
        x_col = col1.selectbox("Select the x variable", columns, key="regimes_x")
        y_col = col2.selectbox("Select the y variable", columns, key="regimes_y")
        col1.write(
            plot_scatter_by_regime(eu_df[eu_df.index.map(pd.tseries.offsets.BDay().onOffset)],
                                        x_col, y_col, regimes, os.path.join(path, 'EU-7', model,f'distributions_{x_col}_{y_col}_regimes_bdays.png')))

        col1.caption(f"{x_col} and {y_col} variables against under regimes on Busines Days")
        col2.write(
            plot_scatter_by_regime(eu_df[[not elem for elem in eu_df.index.map(pd.tseries.offsets.BDay().onOffset)]],
                                   x_col, y_col, regimes, os.path.join(path, 'EU-7', model, f'distributions_{x_col}_{y_col}_regimes_bdays.png')))
        col2.caption(f"{x_col} and {y_col} variables against under regimes on Day Offs")

        selected_cols = st.multiselect("Select the variable(s)", columns, ['Load'], key="maps_columns_regimes")

        st.write(
            plot_maps_by_regime(eu_df.reset_index(level=0), 'EU-7', selected_cols,
                      os.path.join(path, 'EU-7', model, f'maps_{"_".join(selected_cols)}_regimes.png')))

def plot_country(country, df, predictions, pcs):
    try:
        os.makedirs(os.path.join(path, country))
    except OSError as oserr:
        print(oserr)


    country_df = filter_by_country(df, country)
    columns = country_df.columns

    st.subheader("Principal Components of the Regimes")
    with st.expander('', expanded=True):
        selected = st.multiselect("Select one or more variables", columns, ['Wind Load Factor', 'Solar Load Factor', 'Load'], key = "pcs")
        st.write(
            plot_3Dpcs(country_df, pcs, selected,
                       os.path.join(path, country, 'scatter_measure_pcs.png'))
        )
        st.caption(f"Point distributions of the {','.join(selected)} variables with respect to the PCS")

    st.subheader('Variable vs. Variable')
    with st.expander('', expanded=True):
        col1, col2 = st.columns(2)
        x_col = col1.selectbox("Select the x variable", columns)
        y_col = col2.selectbox("Select the y variable", columns)
        col1.write(
            plot_scatter(country_df[country_df.index.map(pd.tseries.offsets.BDay().onOffset)], x_col, y_col,
                         os.path.join(path, country, f'{x_col}_vs_{y_col}_bdays.png'),
                         c=country_df[country_df.index.map(pd.tseries.offsets.BDay().onOffset)][x_col],
                         cmap='RdBu_r')
        )
        col1.caption(f"Distribution {x_col} vs. {y_col} on Business Days")

        col2.write(
            plot_scatter(country_df[[not elem for elem in country_df.index.map(pd.tseries.offsets.BDay().onOffset)]],
                         x_col, y_col, os.path.join(path, country, f'{x_col}_vs_{y_col}_holidays.png'),
                         c=country_df[[not elem for elem in country_df.index.map(pd.tseries.offsets.BDay().onOffset)]][
                             x_col], cmap='RdBu_r')
        )
        col2.caption(f"Distribution {x_col} vs. {y_col} on Day Offs")

    st.subheader("Regimes")
    with st.expander('', expanded = True):
        model = st.selectbox("Select the model", predictions.columns.get_level_values(0).unique(), index = 2)
        try:
            os.makedirs(os.path.join(path, country, model))
        except OSError as oserr:
            print(oserr)

        country_df = filter_by_preds(country_df, predictions, model)

        selected = st.multiselect("Select one or more variables", columns, ['Wind Load Factor', 'Solar Load Factor', 'Load'], key = "regimes")
        st.write(
            plot_distribution_by_regime_days(country_df, selected, predictions, model,
                                             os.path.join(path, country, model, f'distributions_{"_".join(selected)}_regimes_days.png')))
        st.caption(f"Distribution of the {','.join(selected)} variables under regimes on Business Days (left) and Day Offs (right)")

        st.write(
            plot_distribution_by_regime_monthly(country_df, selected,
                                            predictions, model, os.path.join(path, country, model,f'distributions_monthly_{"_".join(selected)}_regimes.png')))
        st.caption(f"Distribution of the {','.join(selected)} variables under regimes at a monthly level")

        col1, col2 = st.columns(2)
        col1.write(
            plot_extreme_events(country_df[country_df.index.map(pd.tseries.offsets.BDay().onOffset)], selected,
                                predictions, model, os.path.join(path, country, model,f'extreme_events_{"_".join(selected)}_regimes_bdays.png')))
        col1.caption(f"Extreme events of the {','.join(selected)} variables under regimes on Busines Days")

        col2.write(
            plot_extreme_events(country_df[[not elem for elem in country_df.index.map(pd.tseries.offsets.BDay().onOffset)]],
                                selected, predictions, model,
                                os.path.join(path, country, model, f'extreme_events_{"_".join(selected)}_regimes_holidays.png')))
        col2.caption(f"Extreme events of the {','.join(selected)} variables under regimes on Days Off")

        col1, col2 = st.columns(2)
        x_col = col1.selectbox("Select the x variable", columns, key = "regimes_densitiy_x")
        y_cols = col2.multiselect("Select the y variable(s)", [c for c in columns if c!= x_col], ['Wind Load Factor', 'Solar Load Factor'], key = "regimes_density_y")
        st.write(
            plot_densities_vs_variable(country_df, x_col, y_cols,
                                       os.path.join(path, country, model, f'densities_{x_col}_{"_".join(y_cols)}_regimes.png')))
        st.caption(f"Probability densities of {','.join(y_cols)} variables against {x_col} variable by regime")

        selected = st.multiselect("Select the variables you want to plot", columns, ['Wind Load Factor', 'Solar Load Factor'], key = "boxplot")
        st.write(
            boxplot(country_df, ['Regime'], selected,
                    os.path.join(path, country, model, f'boxplots_{"_".join(selected)}_regimes.png'))
        )
        st.caption(f"Boxplots of {','.join(selected)} variables under regimes")

        st.write(
            boxplot_monthly(country_df, ['Regime', 'Month'], selected,
                            os.path.join(path, country, model, f'boxplots_{"_".join(selected)}_regimes_monthly.png'))
        )
        st.caption(f"Monthly Boxplots of {','.join(selected)} variables under regimes")

        monthly_stats = country_df.groupby([country_df.index.month, 'Regime']).mean()
        st.dataframe(monthly_stats)

        selected = st.multiselect("Select the variables you want to plot", columns, ['Wind Load Factor', 'Solar Load Factor'], key = "barplot")
        st.write(
            plot_bars(monthly_stats, selected,
                      os.path.join(path, country, model, f'barplot_{"_".join(selected)}_regimes_monthly.png'))
        )

        st.caption(f"Bar plots of the {','.join(selected)} variables under regimes")

        ### SCATTER
        regimes = st.multiselect("Select one or more regimes", country_df['Regime'].unique().tolist(), country_df['Regime'].unique().tolist(), key = "regimes_select")
        col1, col2 = st.columns(2)
        x_col = col1.selectbox("Select the x variable", columns, key="regimes_x")
        y_col = col2.selectbox("Select the y variable", columns, key="regimes_y")
        col1.write(
            plot_scatter_by_regime(country_df[country_df.index.map(pd.tseries.offsets.BDay().onOffset)],
                                        x_col, y_col, regimes, os.path.join(path, country, model,f'distributions_{x_col}_{y_col}_regimes_bdays.png')))

        col1.caption(f"{x_col} and {y_col} variables against under regimes on Busines Days")
        col2.write(
            plot_scatter_by_regime(country_df[[not elem for elem in country_df.index.map(pd.tseries.offsets.BDay().onOffset)]],
                                   x_col, y_col, regimes, os.path.join(path, country, model, f'distributions_{x_col}_{y_col}_regimes_bdays.png')))
        col2.caption(f"{x_col} and {y_col} variables against under regimes on Day Offs")


        selected_cols = st.multiselect("Select the variable(s)", columns, ['Load'], key="maps_columns_regimes")

        st.write(
            plot_maps_by_regime(country_df.reset_index(level=0), country, selected_cols,
                      os.path.join(path, country, model, f'maps_{"_".join(selected_cols)}_regimes.png')))

def plot_subseasonal_forecasts(df, subseasonal_df, shortterm_df, predictions):
    model = st.selectbox("Select the model", predictions.columns.get_level_values(0).unique(), index = 2)
    col1, col2 = st.columns(2)
    countries = col1.selectbox("Select the coutry", df.index.get_level_values(0).unique().tolist() + ['EU-7'], key = "countries_subseasonal")
    try:
        os.makedirs(os.path.join(path, "Subseasonal_forecast", countries, model))
        os.makedirs(os.path.join(path, "Subseasonal_forecast", "ECMWF_weights"))
    except OSError as oserr:
        print(oserr)

    variables = col2.multiselect("Select the variables", df.columns.values.tolist(), ['Load','Wind Load Factor', 'Solar Load Factor'], key = "variables_subseasonal")

    st.write(
        get_forecast_errors(subseasonal_df, predictions, model,
                            os.path.join(path, "Subseasonal_forecast", "ECMWF_weights", "forecast_errors")))
    st.write(get_average_weights_forecast(subseasonal_df, predictions, model))

    dates = subseasonal_df.index.get_level_values(0)#.unique().intersection(predictions.index)
    date = st.slider("Select a forecast date",
                     min_value=dates[0].date()+timedelta(7), max_value=dates[-1].date(), step=timedelta(1))
    st.subheader("Historical distributions")
    month = date.month if date.month in [1,2,12] else min([1,2,12], key=lambda x: abs(x-date.month))

    filtered_df = filter_by_country(df, countries).loc[
        filter_by_country(df, countries).index.intersection(predictions.loc[predictions.index.month == month].index)]
    st.write(
        plot_distribution_by_regime(filtered_df, variables, predictions, model, os.path.join(path, "Subseasonal_forecast", countries, model, "historical_distribution.png")))

    forecast = filter_forecast(subseasonal_df, date, backward = False)
    st.write(plot_weights_forecast(forecast, date,  os.path.join(path, "Subseasonal_forecast","ECMWF_weights", date.strftime("%Y%m%d")+".png")))
    forecast = bayesian_forecasts(subseasonal_df, forecast, predictions, model)
    st.write(plot_weights_forecast(forecast, date,  os.path.join(path, "Subseasonal_forecast","ECMWF_weights", date.strftime("%Y%m%d")+".png")))

    try:
        short_term = shortterm_df.xs(date.strftime("%Y-%m-%d"), level = 1)
    except:
        short_term = None
    predictions_weights = predictions.xs(model, level = 0, axis = 1).loc[date:date+timedelta(days = len(forecast))].drop('Prediction', axis = 1)
    st.write(plot_weights_forecast(predictions_weights.reindex(columns = sorted(predictions_weights.columns)),
                                    date, os.path.join(path, "Subseasonal_forecast","model_weights", date.strftime("%Y%m%d")+".png")))
    #st.write(
    #    plot_forecast_distribution(df, forecast, predictions, countries, variables, date, model, os.path.join(path, "Subseasonal_forecast", countries, model)))

    st.write(
        plot_forecast_boxplots(filtered_df, forecast, predictions, countries, variables, date, model,
                                   os.path.join(path, "Subseasonal_forecast", countries, model)))
    st.write(
        plot_forecast_anomalies(filtered_df, forecast, short_term, predictions, countries, variables, date, model,
                                   os.path.join(path, "Subseasonal_forecast", countries, model)))

def plot_MF(df, predictions, targets, reservoir, inflow, groundwater):
    st.subheader("Comparison with Metéo-France targets")

    targets = targets.xs('Distance', level=0, axis=1)
    predictions = predictions.loc[predictions.index.intersection(targets.index)]
    models = predictions.columns.get_level_values(0).unique().tolist()

    st.write(
        plot_conf_matrix(predictions.loc[:, (models, ['AR', 'NAO+', 'NAO-', 'SB'])], targets.drop('Prediction', axis=1),
                         models))
    st.caption(f"Confusion matrices of the {', '.join(models)} against Metéo-France targets")

    st.write(
        plot_multiclass_roc(predictions.loc[:, (models, ['AR', 'NAO+', 'NAO-', 'SB'])],
                            targets.drop('Prediction', axis=1), models))
    st.caption(f"ROC curves of the {', '.join(models)} against Metéo-France targets")

    model = st.selectbox("Select one or more models", models, index = 2, key = 'model_multiselect')
    predictions = predictions.xs(model, level=0, axis=1).reindex(sorted(targets.columns), axis=1)
    targets = targets.loc[targets.index.intersection(predictions.index)]

    st.write(
        plot_historical_probabilities(targets, pivot = True, title = '../imgs/stacked_bar_MF.png'))
    st.caption(f"Historical monthly regimes probabilities of Metéo-France targets")

    st.write(
        plot_historical_probabilities(predictions.drop('Prediction', axis = 1), title = '../imgs/stacked_bar_predictionsVAE.png'))
    st.caption(f"Historical monthly regimes probabilities of {model}")

    st.write(
        plot_historical_counts(targets, pivot = True, title = '../imgs/count_winter_MF.png'))
    st.write(
        plot_historical_counts(predictions.drop('Prediction', axis = 1), title = '../imgs/count_winter_predictionsVAE.png'))


    f1, f2, f3 = plot_hydro_corr(predictions.drop('Prediction', axis = 1), reservoir, inflow, groundwater)
    st.write(f1)
    st.write(f2)
    st.write(f3)

def plot_dynamics_model(predictions, targets):
    st.subheader("Dynamics")
    model = st.selectbox("Select a model", predictions.columns.get_level_values(0).unique().tolist(), index = 2)
    days = st.slider("Select a range of days to evaluate the transitions", min_value = 1, max_value= 30, step = 1)
    stats = get_state_transitions(predictions,days)
    plot_dynamics(stats)
    image = Image.open(f"W:/UK/Research/Private/WEATHER/STAGE_ABALDO/scripts/mc_imgs/{model}.png")
    st.image(image, caption=f"{model} dynamics")

    model = st.selectbox("Select a model", ['GMM', 'Bayesian-GMM'])
    year = st.selectbox("Select a winter", predictions.index.year.unique()[:-1])
    st.write(
        compare_dynamics(predictions.loc[str(year)+"-12-01":].iloc[:90],
                         targets.xs('Distance', level = 0, axis = 1).loc[str(year)+"-12-01":].iloc[:90],
                         model, '../imgs/compare_dynamics_MF.png'))


def build_webpage(true_df, synthetic_df, synthetic_mapped, subseasonal_df, shortterm_df, predictions, targets_MF, reservoir, inflow, groundwater, pcs):
    st.title("Energy and Weather Variables Dashboard")
    countries = synthetic_df.index.get_level_values(0).unique().tolist()
    options = ['EU-7'] + countries + ['Sub-seasonal Forecasts', 'Metéo-France', 'Model Dynamics', 'Comparison True Measurements and Synthetic Data']
    sel_opt = st.selectbox("Select an option", options)

    if sel_opt in countries:
        plot_country(sel_opt, true_df, predictions, pcs)

    else:
        if sel_opt == "Comparison True Measurements and Synthetic Data":
            plot_comparison(true_df, synthetic_mapped)
        elif sel_opt == "EU-7":
            plot_EU7(synthetic_mapped, predictions, pcs)
        elif sel_opt == "Sub-seasonal Forecasts":
            plot_subseasonal_forecasts(synthetic_df, subseasonal_df, shortterm_df, predictions)
        elif sel_opt == "Metéo-France":
            plot_MF(synthetic_df, predictions, targets_MF, reservoir, inflow, groundwater)
        else:
            plot_dynamics_model(predictions, targets_MF)