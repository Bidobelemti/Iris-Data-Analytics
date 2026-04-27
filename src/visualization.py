import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_hist(df: pd.DataFrame, X:int = 10, Y:int = 8):
    '''
    Show the histogram of the dataset, excluding the last column which is the target variable.
    Parameters:
        - df (pd.DataFrame) : The input dataframe containing the dataset.
        - X (int) : The width of the histogram figure. Default is 10.
        - Y (int) : The height of the histogram figure. Default is 8.
    Returns:
        - None: This function does not return anything. It displays the histogram plot.
    '''
    df.hist(figsize=(X, Y))
    plt.show()

def show_bloxplots (df: pd.DataFrame):
    '''
    Show the boxplots of the dataset, excluding the last column which is the target variable.
    Parameters:
        - df (pd.DataFrame) : The input dataframe containing the dataset.
    Returns:
        - None: This function does not return anything. It displays the boxplot for each feature in the dataset.
    '''
    for col in df.columns[:-1]:
        sns.boxplot(x=df[col])
        plt.title(col)
        plt.show()

def show_relation_length_width (df: pd.DataFrame, X:int = 10, Y:int = 8, y_val: str ='', x_val : str= ''):
    '''
    Show the relation between sepal length and width using a scatter plot, with different colors for each species.
    Parameters:
        - df (pd.DataFrame) : The input dataframe containing the dataset.
        - X (int) : The width of the scatter plot figure. Default is 10.
        - Y (int) : The height of the scatter plot figure. Default is 8.
        - y_val (str) : The column name for the y-axis variable. Default is an empty string.
        - x_val (str) : The column name for the x-axis variable. Default is an empty string.
    Returns:
        - None: This function does not return anything. It displays the scatter plot.
    '''
    plt.figure(figsize=(X, Y))
    sns.scatterplot(
        data=df,
        x=x_val,
        y=y_val,
        hue='Species',
        palette='deep'
    )
    plt.title('Relation between sepal length and width')
    plt.show()

def show_pca (df : pd.DataFrame, hue: str):
    '''
    Show the PCA scatter plot of the dataset, with different colors for each hue.
    Parameters:
        - df (pd.DataFrame) : The input dataframe containing the PCA components and the target variable.
        - hue (str) : The column name for the target variable to be used for coloring the points in the scatter plot.
    Returns:
        - None: This function does not return anything. It displays the PCA scatter plot.
    '''
    sns.scatterplot(data=df, x='PC1', y = 'PC2', hue=hue)
    plt.title('PCA Graphic IRIS')
    plt.show()