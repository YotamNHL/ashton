import pandas as pd
import numpy as np
import shap
import logging
from kmodes.kmodes import KModes
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier


def get_cat_features_static(df):
    lst = list(df.select_dtypes(include=['object']).columns.values)
    return [str(i) for i in lst]


class Ashton(object):
    def __init__(self, X, y, data_to_ashtonize, model):
        logging.basicConfig(level=logging.INFO, filename='log.txt', filemode='w', format='%(asctime)s %(message)s')
        self.logger = logging.getLogger()

        # X, y
        self.X = X
        self.y = y
        self.X_cat_grouped = None
        self.data_to_ashtonize = data_to_ashtonize
        self.data_to_ashtonize_cat_grouped = None

        # models
        self.target_column = 'y'
        self.model = model
        self.clusters_for_kmodes = 5
        self.kmodes_model = None

    def get_cat_features(self):
        lst = list(self.X.select_dtypes(include=['object']).columns.values)
        return [str(i) for i in lst]

    def group_cat_features_for_data_to_ashtonize(self):
        # Using KModes, replace all cat features with one categorical id for each instance
        cat_features = self.get_cat_features()
        data_to_ashtonize_cat = self.data_to_ashtonize[cat_features]
        clusters = self.kmodes_model.predict(data_to_ashtonize_cat)
        data_to_ashtonize_none_cat = self.data_to_ashtonize[[col for col in self.data_to_ashtonize if col not in cat_features]]
        self.data_to_ashtonize_cat_grouped = pd.concat([data_to_ashtonize_none_cat, pd.Series(clusters)], axis=1)

    def group_cat_features(self):
        # Using KModes, replace all cat features with one categorical id for each instance
        cat_features = self.get_cat_features()
        X_cat = self.X[cat_features]
        # init KModes
        self.kmodes_model = KModes(n_clusters=self.clusters_for_kmodes, init='Huang', n_init=5, verbose=1)
        # get clusters
        clusters = self.kmodes_model.fit_predict(X_cat)
        X_none_cat = self.X[[col for col in self.X.columns if col not in cat_features]]
        # Replace all the categorical features with a calculated group cluster id
        self.X_cat_grouped = pd.concat([X_none_cat, pd.Series(clusters)], axis=1)
        self.group_cat_features_for_data_to_ashtonize()

    def get_correct_neighbors_per_instance(self, instance):
        # Get the population with the opposite of the predicted value for that instance
        X_opp = self.X_cat_grouped.iloc[self.y[self.y != (instance[self.target_column])].index, :]
        # Out of that population, take only the population with the same cluster value
        return X_opp[:, X_opp['cluster'] == instance[0, 'cluster']]

    def ashtonize(self):
        self.group_cat_features()
        for index, instance in self.data_to_ashtonize_cat_grouped.iterrows():
            possible_neighbors = self.get_correct_neighbors_per_instance(instance)
            possible_neighbors_and_instance = pd.concat([possible_neighbors, instance], axis=0, ignore_index=True)
            possible_neighbors_and_instance.drop(['cluster', self.target_column], axis=1, inplace=True)
            scaler = MinMaxScaler()
            possible_neighbors_and_instance_scaled = pd.DataFrame(scaler.fit_transform(possible_neighbors_and_instance),
                                                                  columns=possible_neighbors_and_instance.columns)
            possible_neighbors_scaled = possible_neighbors_and_instance_scaled.iloc[:-1, :]
            instance_scaled = possible_neighbors_and_instance_scaled.iloc[-1, :]
            dists = [euclidean(instance_scaled, possible_neighbors_scaled.iloc[i]) for i in range(possible_neighbors_scaled.shape[0])]
            closet_obs = self.X.iloc[np.argmin(dists)]
            # TODO: Nothing
            print(closet_obs)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv('data/aefd.csv')
    df.rename(columns={'STATUS': 'y'}, inplace=True)
    df_train = df.iloc[0:17401, :]
    data_to_ashtonize = df.iloc[17401:17442, :]
    X = df_train[[col for col in list(df.columns) if col != 'y']]
    y = df_train['y']
    cbc = CatBoostClassifier(cat_features=get_cat_features_static(X),
                             random_state=42,
                             bootstrap_type='Bayesian',
                             rsm=0.1,
                             verbose=False)
    cbc.fit(X, y)
    ashton = Ashton(X=X, y=y, data_to_ashtonize=data_to_ashtonize, model=cbc)
    ashton.ashtonize()