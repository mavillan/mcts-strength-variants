import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def split_agent_fields(df):
    agent1_cols = ['agent1_selection', 'agent1_exploration_const', 'agent1_playout', 'agent1_score_bounds']
    agent2_cols = ['agent2_selection', 'agent2_exploration_const', 'agent2_playout', 'agent2_score_bounds']
    df[agent1_cols] = df['agent1'].str.split('-', expand=True).iloc[:, 1:]
    df[agent2_cols] = df['agent2'].str.split('-', expand=True).iloc[:, 1:]
    return df


def process_train_data(
    df_train: pd.DataFrame,
    scale: bool = False,
    numerical_cols: list = None,
    categorical_cols: list = None,
):
    df_train = split_agent_fields(df_train)

    # Identify numerical and categorical columns
    if numerical_cols is None:
        numerical_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_cols = [
            col for col in numerical_cols 
            if col not in ['Id', 'num_wins_agent1', 'num_draws_agent1', 'num_losses_agent1', 'utility_agent1']
        ]
    if categorical_cols is None:
        categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [
            col for col in categorical_cols 
            if col not in ['GameRulesetName','EnglishRules', 'LudRules']
        ]

    # Remove all NaN/null numerical columns
    all_nan_cols = df_train[numerical_cols].columns[df_train[numerical_cols].isna().all()]
    numerical_cols = [col for col in numerical_cols if col not in all_nan_cols.tolist()]
    print("number of all nan cols: ", len(all_nan_cols))

    # Remove constant columns
    constant_cols = df_train[numerical_cols].columns[df_train[numerical_cols].std() == 0]
    numerical_cols = [col for col in numerical_cols if col not in constant_cols]
    print("number of constant cols: ", len(constant_cols))

    # Apply ordinal encoding to categorical columns
    encoder = OrdinalEncoder()
    df_train[categorical_cols] = encoder.fit_transform(df_train[categorical_cols])
    df_train[categorical_cols] = df_train[categorical_cols].astype(int)

    # Fit and transform the numerical columns of df_train
    if scale:
        scaler = StandardScaler()
        df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])
    else:
        scaler = None

    df_train[numerical_cols] = df_train[numerical_cols].astype(np.float32)
    df_train[categorical_cols] = df_train[categorical_cols].astype(np.int32)

    return df_train, numerical_cols, categorical_cols, encoder, scaler


def process_test_data(
    df_test: pd.DataFrame,
    numerical_cols: list,
    categorical_cols: list,
    encoder: OrdinalEncoder,
    scaler: StandardScaler = None
):
    df_test = split_agent_fields(df_test)

    # Apply ordinal encoding to categorical columns
    df_test[categorical_cols] = encoder.transform(df_test[categorical_cols])

    # Fit and transform the numerical columns of df_test
    if scaler is not None:
        df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])

    df_test[numerical_cols] = df_test[numerical_cols].astype(np.float32)
    df_test[categorical_cols] = df_test[categorical_cols].astype(np.int32)

    return df_test
