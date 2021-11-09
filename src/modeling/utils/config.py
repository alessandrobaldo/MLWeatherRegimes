import json

with open('W:/UK/Research/Private/WEATHER/STAGE_ABALDO/scripts/src/modeling/utils/config.json', 'r') as f:
    config = json.load(f)

'''
READ_PATH = 'P:\CH\Weather Data\ERA-5\GEOPOTENTIAL'
physical_qty = 'Geopotential-500hPa'
G = 9.80665
freq = 'hourly' # 'monthly'
months = 'DecJanFeb'#'DecJanFeb'  #'MayJunJulAugSep'
obs_years = '1979-2020'#'1979-2020'
LAT, LONG = (20.,80.), (-90., 30.)
'''

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

