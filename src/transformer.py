# code taken from:
# https://github.com/DataCanvasIO/Hypernets/blob/1fc49015567655d4c01b7307f88fcf109979547b/hypernets/tabular/sklearn_ex.py#L646

import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
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

        self.lgbm.fit(
            X, y,
            # categorical_feature=self.cat_cols
        )
        return self

    def transform(self, X, verbose=False):
        X = X.copy()
        X[self.num_cols] = X[self.num_cols].astype('float')
        X[self.cat_cols] = X[self.cat_cols].astype('int')

        leaves = self.lgbm.predict(X, pred_leaf=True, num_iteration=self.lgbm.best_iteration_)
        new_columns = [f'lgbm_leaf_{i}' for i in range(leaves.shape[1])]
        if verbose:
            print(f'Adding {len(new_columns)} new columns from LGBM leaves')
        df_leaves = pd.DataFrame(leaves, columns=new_columns, index=X.index)
        result = pd.concat([X, df_leaves], axis=1)
        self.new_columns = new_columns
        return result

class XGBLeavesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols ,cat_cols, task, **params):
        super(XGBLeavesEncoder, self).__init__()

        self.xgb = None
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.new_columns = []
        self.task = task
        self.xgb_params = params

    def fit(self, X, y):
        X = X.copy()
        X[self.num_cols] = X[self.num_cols].astype('float')
        X[self.cat_cols] = X[self.cat_cols].astype('int')

        if self.task == 'regression':
            self.xgb = XGBRegressor(**self.xgb_params)
        else:
            self.xgb = XGBClassifier(**self.xgb_params)

        self.xgb.fit(X, y)
        return self

    def transform(self, X, verbose=False):
        X = X.copy()
        X[self.num_cols] = X[self.num_cols].astype('float')
        X[self.cat_cols] = X[self.cat_cols].astype('int')

        leaves = self.xgb.predict(X, pred_leaf=True)
        new_columns = [f'xgb_leaf_{i}' for i in range(leaves.shape[1])]
        if verbose:
            print(f'Adding {len(new_columns)} new columns from XGB leaves')
        df_leaves = pd.DataFrame(leaves, columns=new_columns, index=X.index)
        result = pd.concat([X, df_leaves], axis=1)
        self.new_columns = new_columns
        return result

class CATBLeavesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols ,cat_cols, task, **params):
        super(CATBLeavesEncoder, self).__init__()

        self.catb = None
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.new_columns = []
        self.task = task
        self.catb_params = params

    def fit(self, X, y):
        X = X.copy()
        X[self.num_cols] = X[self.num_cols].astype('float')
        X[self.cat_cols] = X[self.cat_cols].astype('int')

        if self.task == 'regression':
            self.catb = CatBoostRegressor(**self.catb_params)
        else:
            self.catb = CatBoostClassifier(**self.catb_params)

        self.catb.fit(X, y, cat_features=self.cat_cols)
        return self

    def transform(self, X, verbose=False):
        X = X.copy()
        X[self.num_cols] = X[self.num_cols].astype('float')
        X[self.cat_cols] = X[self.cat_cols].astype('int')

        leaves = self.catb.calc_leaf_indexes(X)
        new_columns = [f'catb_leaf_{i}' for i in range(leaves.shape[1])]
        if verbose:
            print(f'Adding {len(new_columns)} new columns from CATB leaves')
        df_leaves = pd.DataFrame(leaves, columns=new_columns, index=X.index)
        result = pd.concat([X, df_leaves], axis=1)
        self.new_columns = new_columns
        return result
