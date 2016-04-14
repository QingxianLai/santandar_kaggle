"""
by Qingxian Lai
"""

import numpy as np
import pandas as pd

from subprocess import check_output
import logging
import os
try:
    os.makedirs('log')
except:
    pass
logging.basicConfig(filename='./log/data.log', level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


def save_data(df_train, df_test):
    logging.info("save data to pickle...")
    df_train.to_pickle("./data/train.pkl")
    df_test.to_pickle("./data/test.pkl")
    logging.info("done.\n")


def remove_constant_columns(df_train, df_test):
    logging.info("remove constant columns...")

    remove = []
    for col in df_train.columns:
        if df_train[col].std() == 0:
            remove.append(col)

    df_train.drop(remove, axis=1, inplace=True)
    df_test.drop(remove, axis=1, inplace=True)


def remove_duplicated_columns(df_train, df_test):
    logging.info("remove duplicated_columns...")

    remove = []
    c = df_train.columns
    for i in range(len(c) - 1):
        v = df_train[c[i]].values
        for j in range(i+1, len(c)):
            if np.array_equal(v, df_train[c[j]].values):
                remove.append(c[j])

    df_train.drop(remove, axis=1, inplace=True)
    df_test.drop(remove, axis=1, inplace=True)


def main():
    print(check_output(["ls", "./data"]).decode("utf8"))

    # load data
    logging.info("loading data...")
    df_train = pd.read_csv('./data/train.csv')
    df_test = pd.read_csv('./data/test.csv')

    print df_train.shape

    remove_constant_columns(df_train, df_test)
    remove_duplicated_columns(df_train, df_test)

    print df_train.shape
    save_data(df_train, df_test)


if __name__ == '__main__':
    main()
