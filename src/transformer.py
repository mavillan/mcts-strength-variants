# code taken from:
# https://github.com/DataCanvasIO/Hypernets/blob/1fc49015567655d4c01b7307f88fcf109979547b/hypernets/tabular/sklearn_ex.py#L646

import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin

class LGBMLeavesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols ,cat_cols, task, **params):
        super(LGBMLeavesEncoder, self).__init__()

        self.lgbm = None
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.new_columns = []
        self.task = task
        self.lgbm_params = params

    def fit(self, X, y):
        X = X.copy()
        X[self.num_cols] = X[self.num_cols].astype('float')
        X[self.cat_cols] = X[self.cat_cols].astype('int')

        if self.task == 'regression':
            self.lgbm = LGBMRegressor(**self.lgbm_params)
        else:
            self.lgbm = LGBMClassifier(**self.lgbm_params)
        self.lgbm.fit(X, y)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.num_cols] = X[self.num_cols].astype('float')
        X[self.cat_cols] = X[self.cat_cols].astype('int')

        leaves = self.lgbm.predict(X, pred_leaf=True, num_iteration=self.lgbm.best_iteration_)
        new_columns = [f'lgbm_leaf_{i}' for i in range(leaves.shape[1])]
        df_leaves = pd.DataFrame(leaves, columns=new_columns, index=X.index)
        result = pd.concat([X, df_leaves], axis=1)
        self.new_columns = new_columns
        return result