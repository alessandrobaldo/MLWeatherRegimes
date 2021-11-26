import json

with open('W:/UK/Research/Private/WEATHER/STAGE_ABALDO/scripts/src/dashboard/utils/config.json', 'r') as f:
    config = json.load(f)

SEASON = config['season']
PREDICTION_FILE = config['predictions']
MONTHS = config['months']
MONTHS_STR = config['months_str']
REGIMES = config['regimes']