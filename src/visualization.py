import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_hist(df: pd.DataFrame, X:int = 10, Y:int = 8):
    df.hist(figsize=(X, Y))
    plt.show

def show_bloxplots (df: pd.DataFrame):
    for col in df.columns[:-1]:
        sns.boxplot(x=df[col])
        plt.title(col)
        plt.show()

def show_relation_length_width (df: pd.DataFrame, X:int = 10, Y:int = 8, y_val: str ='', x_val : str= ''):
    plt.figure(figsize=(X, Y))
    sns.scatterplot(
        data=df,
        x=x_val,
        y=y_val,
        hue='Species',
        palette='deep'
    )
    plt.title('Relation between sepal lenght and width')
    plt.show