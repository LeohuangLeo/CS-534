import numpy as np
import pandas as pd

# data preprocessing

# read csv file and drop 'veil-type_p' feature
def Read_data(filename):
    df = pd.read_csv(filename)
    df.drop("veil-type_p", axis=1)
    return df

# split the data into left and right parts depending on the feature we selected
def split_data(data, split_column):
    split_column_value = data[:, split_column]
    data_left = data[split_column_value == 0]
    data_right = data[split_column_value == 1]
    return data_left,data_right