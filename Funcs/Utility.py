import pytz
import os
import pandas as pd
import numpy as np
import scipy.stats as st
import cloudpickle
import ray
from datetime import datetime
from contextlib import contextmanager
import warnings
import time


DEFAULT_TZ = pytz.FixedOffset(540)  # GMT+09:00; Asia/Seoul

PATH_DATA = "/var/nfs_share/D#1"
PATH_ESM = os.path.join(PATH_DATA, 'EsmResponse.csv')
PATH_PARTICIPANT = os.path.join(PATH_DATA, 'UserInfo.csv')
PATH_SENSOR = os.path.join(PATH_DATA, 'Sensor')

PATH_INTERMEDIATE = '/var/nfs_share/Stress_Detection_D-1/Intermediate'


#LABEL_THRESHOLD = 87  # D#1: 31, D#2: 31, D#3: 108, D#4: 87

DATA_TYPES = {
    'Acceleration': 'ACC',
    'AmbientLight': 'AML',
    'Calorie': 'CAL',
    'Distance': 'DST',
    'EDA': 'EDA',
    'HR': 'HRT',
    'RRI': 'RRI',
    'SkinTemperature': 'SKT',
    'StepCount': 'STP',
    'UltraViolet': 'ULV',
    'ActivityEvent': 'ACE',
    'ActivityTransition': 'ACT',
    'AppUsageEvent': 'APP',
    'BatteryEvent': 'BAT',
    'CallEvent': 'CAE',
    'Connectivity': 'CON',
    'DataTraffic': 'DAT',
    'InstalledApp': 'INS',
    'Location': 'LOC',
    'MediaEvent': 'MED',
    'MessageEvent': 'MSG',
    'WiFi': 'WIF',
    'ScreenEvent': 'SCR',
    'RingerModeEvent': 'RNG',
    'ChargeEvent': 'CHG',
    'PowerSaveEvent': 'PWS',
    'OnOffEvent': 'ONF'
}


def load(path: str):
    with open(path, mode='rb') as f:
        return cloudpickle.load(f)

    
def dump(obj, path: str):
    with open(path, mode='wb') as f:
        cloudpickle.dump(obj, f)
        
    
def log(msg: any):
    print('[{}] {}'.format(datetime.now().strftime('%y-%m-%d %H:%M:%S'), msg))


def summary(x):
    x = np.asarray(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        n = len(x)
        # Here, uppercase np.dtype.kind corresponds to non-numeric data.
        # Also, we view the boolean data as dichotomous categorical data.
        if x.dtype.kind.isupper() or x.dtype.kind == 'b': 
            cnt = pd.Series(x).value_counts(dropna=False)
            card = len(cnt)
            cnt = cnt[:20]                
            cnt_str = ', '.join([f'{u}:{c}' for u, c in zip(cnt.index, cnt)])
            if card > 30:
                cnt_str = f'{cnt_str}, ...'
            return {
                'n': n,
                'cardinality': card,
                'value_count': cnt_str
            }
        else: 
            x_nan = x[np.isnan(x)]
            x_norm = x[~np.isnan(x)]
            
            tot = np.sum(x_norm)
            m = np.mean(x_norm)
            me = np.median(x_norm)
            s = np.std(x_norm, ddof=1)
            l, u = np.min(x_norm), np.max(x)
            conf_l, conf_u = st.t.interval(0.95, len(x_norm) - 1, loc=m, scale=st.sem(x_norm))
            n_nan = len(x_nan)
            
            return {
                'n': n,
                'sum': tot,
                'mean': m,
                'SD': s,
                'med': me,
                'range': (l, u),
                'conf.': (conf_l, conf_u),
                'nan_count': n_nan
            }


@contextmanager
def on_ray(*args, **kwargs):
    try:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(*args, **kwargs)
        yield None
    finally:
        ray.shutdown()
        
        
transform = {
    'GAME': 'ENTER',
    'GAME_TRIVIA': 'ENTER',
    'GAME_CASINO': 'ENTER',
    'GAME-ACTION': 'ENTER',
    'GAME_SPORTS': 'ENTER',
    'GAME_PUZZLE': 'ENTER',
    'GAME_SIMULATION': 'ENTER',
    'GAME_STRATEGY': 'ENTER',
    'GAME_ROLE_PLAYING': 'ENTER',
    'GAME_ACTION': 'ENTER',
    'GAME_ARCADE': 'ENTER',
    'GAME_RACING': 'ENTER',
    'GAME_CASUAL': 'ENTER',
    'GAME_MUSIC': 'ENTER',
    'GAME_CARD': 'ENTER',
    'GAME_ADVENTURE': 'ENTER',
    'GAME_BOARD': 'ENTER',
    'GAME_EDUCATIONAL': 'ENTER',
    'GAME_RACING': 'ENTER',
    'PHOTOGRAPHY': 'ENTER',
    'ENTERTAINMENT': 'ENTER',
    'SPORTS': 'ENTER',
    'MUSIC_AND_AUDIO': 'ENTER',
    'COMICS': 'ENTER',
    'VIDEO_PLAYERS_AND_EDITORS': 'ENTER',
    'VIDEO_PLAYERS': 'ENTER',
    'ART_AND_DESIGN': 'ENTER',
    'TRAVEL_AND_LOCAL': 'INFO',
    'FOOD_AND_DRINK': 'INFO',
    'NEWS_AND_MAGAZINES': 'INFO',
    'MAPS_AND_NAVIGATION': 'INFO',
    'WEATHER': 'INFO',
    'HOUSE_AND_HOME': 'INFO',
    'BOOKS_AND_REFERENCE': 'INFO',
    'SHOPPING': 'INFO',
    'LIBRARIES_AND_DEMO': 'INFO',
    'BEAUTY': 'INFO',
    'AUTO_AND_VEHICLES': 'INFO',
    'LIFESTYLE': 'INFO',
    'PERSONALIZATION': 'SYSTEM',
    'TOOLS': 'SYSTEM',
    'COMMUNICATION': 'SOCIAL',
    'SOCIAL': 'SOCIAL',
    'DATING': 'SOCIAL',
    'PARENTING':'SOCIAL',
    'FINANCE': 'WORK',
    'BUSINESS': 'WORK',
    'PRODUCTIVITY': 'WORK',
    'EDUCATION': 'WORK',
    'HEALTH_AND_FITNESS': 'HEALTH',
    'MEDICAL': 'HEALTH',
    'SYSTEM': 'SYSTEM',
    'MISC': 'SYSTEM', # ABC logger
     None: 'UNKNOWN',
    'UNKNOWN':'UNKNOWN'
}