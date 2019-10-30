import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Read_data as rd

#Part 0: Preprocessing and simple analysis

#Statistics for numerical features
def data_numerical(data):
    categorical_features = ['waterfront', 'grade', 'condition']
    numerical_column = [i for i in data if i not in categorical_features]
    df_numerical = data[numerical_column]
    numerical_mean = df_numerical.mean()
    numerical_std = df_numerical.std()
    numerical_range = df_numerical.max() - df_numerical.min()
    numerical_all = pd.concat([numerical_mean, numerical_std, numerical_range], axis = 1)
    numerical_all.columns = ['mean', 'std', 'range']

    return numerical_all

#Statistics for categorical features
def data_categorical(data):
    df_categorical = data[['waterfront', 'grade', 'condition']]
    c_waterfront = df_categorical['waterfront'].value_counts().sort_index(ascending=True)
    c_grade = df_categorical['grade'].value_counts().sort_index(ascending=True)
    c_condition = df_categorical['condition'].value_counts().sort_index(ascending=True)
    c_waterfront_perc = pd.Series(c_waterfront.values / c_waterfront.values.sum())
    c_grade_perc = pd.Series(c_grade.values / c_grade.values.sum(), index = list(range(4,14)))
    c_condition_perc = pd.Series(c_condition.values / c_condition.values.sum(), index = list(range(1,6)))
    c_class = pd.concat([c_waterfront_perc,c_grade_perc,c_condition_perc],axis = 1)
    c_class.columns = ['waterfront', 'grade', 'condition']

    return c_class

#visualize the data and analyze it
def data_visualize(df):
    corr_matrix = df.corr() #Calculate data correlation
    print(corr_matrix['price'].abs().sort_values(ascending=False))
    plt.bar(df["zipcode"], df["price"])
    plt.show()
    plt.bar(df["floors"], df["price"])
    plt.show()
    plt.bar(df["view"], df["price"])
    plt.show()

def main():
    df = rd.read_data('PA1_train.csv')
    X_df = df.drop(columns=['price','dummy'])
    df_numerical = data_numerical(X_df)
    df_categorical = data_categorical(X_df)
    print(df_numerical.T)
    print(df_categorical.T)
    data_visualize(df)

main()