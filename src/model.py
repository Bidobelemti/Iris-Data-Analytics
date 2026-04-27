from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import pandas as pd


class ModelIris:
    def __init__(self, X:pd.DataFrame, y:pd.DataFrame, hue:str):
        '''
        ModelIris class initialization
        Parameters
            - X (pd.DataFrame) : X values of dataset
            - y (pd.DataFrame) : y values of dataset
            - hue (str) : Identification of thing to predict
        '''
        self.X = X
        self.y = y
        self.hue = hue

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)

    def fit_transform_train(self):
        '''
        Fit and transform the training data using StandardScaler and PCA.
        This method scales the training data and then applies PCA to reduce its dimensionality to 2
        components. The transformed training data is stored in the attribute X_train_pca.
        '''
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)

        self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)

    def transform_pca(self, X_input):
        '''
        Transform the input data using the fitted PCA model.
        Parameters:
        - X_input: The input data to be transformed, typically the test set features.   
        Returns:
        - X_pca: The PCA-transformed version of the input data, which has been
        scaled and reduced to 2 principal components.
        '''

        X_scaled = self.scaler.transform(X_input)
        X_pca = self.pca.transform(X_scaled)
        return X_pca

    def get_pca_dataframe(self):
        '''
        Get the PCA-transformed dataframe for the entire dataset, including the target variable.
        This method scales the entire feature set, applies PCA to reduce it to 2 components,
        and combines the results with the target variable.
        '''
        X_scaled = StandardScaler().fit_transform(self.X)
        X_pca = PCA(n_components=2).fit_transform(X_scaled)

        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df[self.hue] = self.y

        return pca_df

    def train_model(self, test_size=0.2, random_state=42, n_neighbors=3):
        '''
        Train a K-Nearest Neighbors (KNN) classifier on the PCA-transformed training data and evaluate its performance on the test set.
        Parameters:
            - test_size : The proportion of the dataset to include in the test split. Default is 0.2 (20%).
            - random_state : Controls the randomness of the train-test split. Default is 42.
            - n_neighbors : The number of neighbors to use for the KNN classifier. Default is 3.
        Returns:
            - model : The trained KNN model.

            - accuracy : The accuracy of the model on the test set.
        '''
        # 1. Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        # 2. Fit + transform en TRAIN
        self.fit_transform_train()

        # 3. Transform TEST
        self.X_test_pca = self.transform_pca(self.X_test)

        # 4. Modelo
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(self.X_train_pca, self.y_train)

        # 5. Predicción
        self.y_pred = self.model.predict(self.X_test_pca)

        # 6. Evaluación
        self.evaluate_model()

        return self.model, self.accuracy

    def evaluate_model(self):
        self.accuracy = accuracy_score(self.y_test, self.y_pred)