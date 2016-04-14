"""
by Qingxian Lai
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from subprocess import check_output

print(check_output(['ls', './data']).decode("utf8"))


def main():
    # load the data
    df_train = pd.read_pickle('./data/train.pkl')
    df_test = pd.read_pickle('./data/test.pkl')

    # gradient boosting cv training
    param = {
        "missing": np.nan,
        "max_depth": 20,
        "eval_metrix": 'auc',
        "early_stopping_rounds": 20,
        "learning_rate": 0.03,
        "nthread": 4,
        "subsample":0.95,
        "colsample_bytree": 0.85,
        "seed":1234
    }

    xgb.cv(param, )

