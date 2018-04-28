#For cleaning and transforming data
import numpy as np
import collections

def log_transform(df):
    '''
    This function takes in a data frame and returns a new data frame that's transformed in natural log format
    :param df: data frame
    :return: log-transformed data
    '''
    for column in df:
        df[column] = np.log(df[column]+1)


def detect_outliers(df):
    '''
    This function takes in data frame, calculates outliers for each field using the IQR rule, and return the indices of the records that are considered outliers
    :param df: data frame
    :return: a collection of fields and their indices of outlier records showing overlapping fields that have outlier records
    '''

    outlier = {}
    for feature in df.keys():
        q1 = np.percentile(df[feature], 25)
        q3 = np.percentile(df[feature], 75)
        iqr = q3 - q1
        step = iqr * 1.5
        outlier[feature] = sorted(df[((df[feature] < q1 - step) | (df[feature] > q3 + step))].index.values.astype(int))

    # include outlier records that have multiple fields
    overlap = collections.defaultdict(list)
    for k, v in outlier.iteritems():
        for i in v:
            overlap[i].append(k)

    return overlap






