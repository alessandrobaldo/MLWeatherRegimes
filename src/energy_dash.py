from dashboard.utils.website import *
from dashboard.utils.data import *
from dashboard.utils.config import *

#true_df = load_measurements.__wrapped__()
true_df = load_measurements()
synthetic_df = load_synthetic_data()
synthetic_mapped = pd.read_csv("dashboard/synthetic_mapped.csv", index_col = [0,1], parse_dates = True)
#df = build_data(true_df, synthetic_df)
predictions = load_predictions(PREDICTION_FILE)
targets_MF = load_MF_targets(season = SEASON, filter_dates=False)
reservoir, inflow, groundwater = load_hydro()
pcs = load_pcs()
subseasonal_df = load_subseasonal()
#create_forecast_file(synthetic_df, subseasonal_df)
#shortterm_df = load_shortterm()
build_webpage(true_df, synthetic_df, synthetic_mapped, subseasonal_df, None, predictions, targets_MF, reservoir, inflow, groundwater, pcs)
