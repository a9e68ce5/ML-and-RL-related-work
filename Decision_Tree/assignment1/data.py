''' This file provides utility functions for loading data that you may find useful.
    You don't need to change this file.
'''

from glob import glob

import pandas as pd


def load_data() -> dict:
    '''
    Loads all the data required for this assignment.

    Returns:
        dict: a dictionary containing the train, test, and cross validation data as pandas dataframes
    '''

    # load train dataset
    train = pd.read_csv('data/train.csv')

    # load test dataset
    test = pd.read_csv('data/test.csv')

    # load cross validation datasets
    cv_folds = []
    for cv_fold_path in glob('data/cv/*'):
        fold = pd.read_csv(cv_fold_path)
        cv_folds.append(fold)

    return {
        'train': train,
        'test': test,
        'cv_folds': cv_folds}
