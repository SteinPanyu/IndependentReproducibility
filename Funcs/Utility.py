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
from typing import Optional


DEFAULT_TZ = pytz.FixedOffset(540)  # GMT+09:00; Asia/Seoul

PATH_DATA = "/var/nfs_share/D#1"
PATH_ESM = os.path.join(PATH_DATA, 'EsmResponse.csv')
PATH_PARTICIPANT = os.path.join(PATH_DATA, 'UserInfo.csv')
PATH_SENSOR = os.path.join(PATH_DATA, 'Sensor')
PATH_RESULTS = '/var/nfs_share/Stress_Detection_D-1/Results'

PATH_INTERMEDIATE = '/var/nfs_share/Stress_Detection_D-1/Intermediate'
RANDOM_STATE =42


#LABEL_THRESHOLD = 87  # D#1: 31, D#2: 31, D#3: 108, D#4: 87
seed= RANDOM_STATE
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
        
def _load_data(
    name: str
) -> Optional[pd.DataFrame]:
    paths = [
        (d, os.path.join(PATH_SENSOR, d, f'{name}.csv'))
        for d in os.listdir(PATH_SENSOR)
        if d.startswith('P')
    ]
    return pd.concat(
        filter(
            lambda x: len(x.index), 
            [
                pd.read_csv(p).assign(pcode=pcode)
                for pcode, p in paths
                if os.path.exists(p)
            ]
        ), ignore_index=True
    ).assign(
        timestamp=lambda x: pd.to_datetime(x['timestamp'], unit='ms', utc=True).dt.tz_convert(DEFAULT_TZ)
    ).set_index(
        ['pcode', 'timestamp']
    )

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


param = {
    "predictor": 'cpu_predictor',
    "early_stopping_rounds": 200,
    "reg_alpha": 0,
    "colsample_bytree": 1,
    "colsample_bylevel": 1,
    "scale_pos_weight": 1,
    "learning_rate": 0.3,
#     "nthread": 10,
#     "nthread": 1,
    "min_child_weight": 1,
#     "n_estimators": 1000, #In current version, it is replaced by num_boost_round
    "subsample": 1,
    "reg_lambda": 1.72,
#     "reg_lambda": 1,
    "reg_alpha":0,
    "seed": seed,
    "objective": 'binary:logistic',
    "max_depth": 6,
    "gamma": 0,
    'eval_metric': 'auc',
    'verbosity': 0,
#     'tree_method': 'exact
    'tree_method': 'gpu_hist',
#     'tree_method': 'hist',
#     'debug': 0,
#     'use_task_gain_self': 0,
#     'when_task_split': 1,
#     'how_task_split': 0,
#     'min_task_gain': 0.0,
#     'task_gain_margin': 0.0,
#     'max_neg_sample_ratio': 0.4,
#     'which_task_value': 2,
#     'baseline_alpha': 1.0,
#     'baseline_lambda': 1.0,
#     'tasks_list_': (0, 1),
#     'task_num_for_init_vec': 3,
#     'task_num_for_OLF': 2,
}