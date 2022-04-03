import json

with open('src/modeling/utils/config.json', 'r') as f:
    config = json.load(f)


READ_PATH = config['readpath']
physical_qty = config['variable']
G = 9.80665
freq = config['frequency']
months = config['months']
obs_years = config['years']
LAT, LONG = config['lat'], config['long']
reduction = config['reduction']
model = config['model']
season = config['season']
training = bool(config['training'])

