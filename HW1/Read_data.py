import pandas as pd

#Read data
def read_data(filename):
    df = pd.read_csv(filename)
    del df['id'] #Remove ID feature
    #Split date feature into three separate numerical features
    df['day'] = pd.DatetimeIndex(df['date']).day
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['year'] = pd.DatetimeIndex(df['date']).year
    del df['date']

    return df