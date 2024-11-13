import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import re


def split_agent_fields(df):
    agent1_cols = ['agent1_selection', 'agent1_exploration_const', 'agent1_playout', 'agent1_score_bounds']
    agent2_cols = ['agent2_selection', 'agent2_exploration_const', 'agent2_playout', 'agent2_score_bounds']
    df[agent1_cols] = df['agent1'].str.split('-', expand=True).iloc[:, 1:]
    df[agent2_cols] = df['agent2'].str.split('-', expand=True).iloc[:, 1:]
    return df


def feat_engineering(df):
    df['area'] = df['NumRows'] * df['NumColumns']
    df['row_equal_col'] = (df['NumColumns'] == df['NumRows']).astype(np.int8)
    df['Playouts/Moves'] = df['PlayoutsPerSecond'] / (df['MovesPerSecond'] + 1e-15)
    df['EfficiencyPerPlayout'] = df['MovesPerSecond'] / (df['PlayoutsPerSecond'] + 1e-15)
    df['TurnsDurationEfficiency'] = df['DurationActions'] / (df['DurationTurnsStdDev'] + 1e-15)
    df['AdvantageBalanceRatio'] = df['AdvantageP1'] / (df['Balance'] + 1e-15)
    df['ActionTimeEfficiency'] = df['DurationActions'] / (df['MovesPerSecond'] + 1e-15)
    df['StandardizedTurnsEfficiency'] = df['DurationTurnsStdDev'] / (df['DurationActions'] + 1e-15)
    df['AdvantageTimeImpact'] = df['AdvantageP1'] / (df['DurationActions'] + 1e-15)
    df['DurationToComplexityRatio'] = df['DurationActions'] / (df['StateTreeComplexity'] + 1e-15)
    df['NormalizedGameTreeComplexity'] = df['GameTreeComplexity'] / (df['StateTreeComplexity'] + 1e-15)
    df['ComplexityBalanceInteraction'] = df['Balance'] * df['GameTreeComplexity']
    df['OverallComplexity'] = df['StateTreeComplexity'] + df['GameTreeComplexity']
    df['ComplexityPerPlayout'] = df['GameTreeComplexity'] / (df['PlayoutsPerSecond'] + 1e-15)
    df['TurnsNotTimeouts/Moves'] = df['DurationTurnsNotTimeouts'] / (df['MovesPerSecond'] + 1e-15)
    df['Timeouts/DurationActions'] = df['Timeouts'] / (df['DurationActions'] + 1e-15)
    df['OutcomeUniformity/AdvantageP1'] = df['OutcomeUniformity'] / (df['AdvantageP1'] + 1e-15)
    df['ComplexDecisionRatio'] = df['StepDecisionToEnemy'] + df['SlideDecisionToEnemy'] + df['HopDecisionMoreThanOne']
    df['AggressiveActionsRatio'] = df['StepDecisionToEnemy'] + df['HopDecisionEnemyToEnemy'] + df['HopDecisionFriendToEnemy'] + df['SlideDecisionToEnemy']
    added_cols = [
        'area', 'row_equal_col', 'Playouts/Moves', 'EfficiencyPerPlayout',
        'TurnsDurationEfficiency', 'AdvantageBalanceRatio', 'ActionTimeEfficiency',
        'StandardizedTurnsEfficiency', 'AdvantageTimeImpact', 'DurationToComplexityRatio',
        'NormalizedGameTreeComplexity', 'ComplexityBalanceInteraction', 'OverallComplexity',
        'ComplexityPerPlayout', 'TurnsNotTimeouts/Moves', 'Timeouts/DurationActions',
        'OutcomeUniformity/AdvantageP1', 'ComplexDecisionRatio', 'AggressiveActionsRatio'
    ]
    return df, added_cols


def agent_position_feature(df):
    all_agents = [
        'MCTS-ProgressiveHistory-0.1-MAST-false', 'MCTS-ProgressiveHistory-0.1-MAST-true', 
        'MCTS-ProgressiveHistory-0.1-NST-false', 'MCTS-ProgressiveHistory-0.1-NST-true',
        'MCTS-ProgressiveHistory-0.1-Random200-false', 'MCTS-ProgressiveHistory-0.1-Random200-true',
        'MCTS-ProgressiveHistory-0.6-MAST-false', 'MCTS-ProgressiveHistory-0.6-MAST-true',
        'MCTS-ProgressiveHistory-0.6-NST-false', 'MCTS-ProgressiveHistory-0.6-NST-true',
        'MCTS-ProgressiveHistory-0.6-Random200-false', 'MCTS-ProgressiveHistory-0.6-Random200-true',
        'MCTS-ProgressiveHistory-1.41421356237-MAST-false', 'MCTS-ProgressiveHistory-1.41421356237-MAST-true',
        'MCTS-ProgressiveHistory-1.41421356237-NST-false', 'MCTS-ProgressiveHistory-1.41421356237-NST-true',
        'MCTS-ProgressiveHistory-1.41421356237-Random200-false', 'MCTS-ProgressiveHistory-1.41421356237-Random200-true',
        'MCTS-UCB1-0.1-MAST-false', 'MCTS-UCB1-0.1-MAST-true', 'MCTS-UCB1-0.1-NST-false',
        'MCTS-UCB1-0.1-NST-true', 'MCTS-UCB1-0.1-Random200-false', 'MCTS-UCB1-0.1-Random200-true',
        'MCTS-UCB1-0.6-MAST-false', 'MCTS-UCB1-0.6-MAST-true', 'MCTS-UCB1-0.6-NST-false',
        'MCTS-UCB1-0.6-NST-true', 'MCTS-UCB1-0.6-Random200-false', 'MCTS-UCB1-0.6-Random200-true',
        'MCTS-UCB1-1.41421356237-MAST-false', 'MCTS-UCB1-1.41421356237-MAST-true',
        'MCTS-UCB1-1.41421356237-NST-false', 'MCTS-UCB1-1.41421356237-NST-true',
        'MCTS-UCB1-1.41421356237-Random200-false', 'MCTS-UCB1-1.41421356237-Random200-true',
        'MCTS-UCB1GRAVE-0.1-MAST-false', 'MCTS-UCB1GRAVE-0.1-MAST-true', 'MCTS-UCB1GRAVE-0.1-NST-false',
        'MCTS-UCB1GRAVE-0.1-NST-true', 'MCTS-UCB1GRAVE-0.1-Random200-false', 'MCTS-UCB1GRAVE-0.1-Random200-true',
        'MCTS-UCB1GRAVE-0.6-MAST-false', 'MCTS-UCB1GRAVE-0.6-MAST-true', 'MCTS-UCB1GRAVE-0.6-NST-false',
        'MCTS-UCB1GRAVE-0.6-NST-true', 'MCTS-UCB1GRAVE-0.6-Random200-false', 'MCTS-UCB1GRAVE-0.6-Random200-true',
        'MCTS-UCB1GRAVE-1.41421356237-MAST-false', 'MCTS-UCB1GRAVE-1.41421356237-MAST-true',
        'MCTS-UCB1GRAVE-1.41421356237-NST-false', 'MCTS-UCB1GRAVE-1.41421356237-NST-true',
        'MCTS-UCB1GRAVE-1.41421356237-Random200-false', 'MCTS-UCB1GRAVE-1.41421356237-Random200-true',
        'MCTS-UCB1Tuned-0.1-MAST-false', 'MCTS-UCB1Tuned-0.1-MAST-true', 'MCTS-UCB1Tuned-0.1-NST-false',
        'MCTS-UCB1Tuned-0.1-NST-true', 'MCTS-UCB1Tuned-0.1-Random200-false', 'MCTS-UCB1Tuned-0.1-Random200-true',
        'MCTS-UCB1Tuned-0.6-MAST-false', 'MCTS-UCB1Tuned-0.6-MAST-true', 'MCTS-UCB1Tuned-0.6-NST-false',
        'MCTS-UCB1Tuned-0.6-NST-true', 'MCTS-UCB1Tuned-0.6-Random200-false', 'MCTS-UCB1Tuned-0.6-Random200-true',
        'MCTS-UCB1Tuned-1.41421356237-MAST-false', 'MCTS-UCB1Tuned-1.41421356237-MAST-true',
        'MCTS-UCB1Tuned-1.41421356237-NST-false', 'MCTS-UCB1Tuned-1.41421356237-NST-true',
        'MCTS-UCB1Tuned-1.41421356237-Random200-false', 'MCTS-UCB1Tuned-1.41421356237-Random200-true'
    ]

    # Create agent position features more efficiently using numpy operations
    agent1_arr = df['agent1'].values
    agent2_arr = df['agent2'].values
    n_samples = len(df)
    
    for agent in all_agents:
        # Use numpy boolean operations instead of loops
        value = np.zeros(n_samples)
        value += (agent1_arr == agent)
        value -= (agent2_arr == agent)
        df[f'agent_{agent}'] = value

    # Create list of agent position feature columns
    added_cols = [f'agent_{agent}' for agent in all_agents]

    return df, added_cols


def clean(df, col):
    df[col] = df[col].fillna("nan")
    
    def clean_text(text):
        # Convert to lowercase
        text = text.lower()
        # Replace multiple spaces with single space
        text = ' '.join(text.split())        
        # Replace remaining punctuation with spaces
        text = re.sub(r'[^\w\s_]', ' ', text)
        # Remove numeric values that appear alone (but keep them if part of a token)
        text = re.sub(r'\b\d+\b', '', text)
        # Final cleanup of multiple spaces
        text = ' '.join(text.split())
        return text
    
    df[col] = df[col].apply(clean_text)
    return df


def remove_game_name(rule):
    """
    Remove only the game name from the rule string.
    Input:  '(game "Kaua Dorki" (players 2) ...'
    Output: '(game (players 2) ...'
    """
    try:
        # Find the position of first and second quote
        first_quote = rule.find('"')
        second_quote = rule.find('"', first_quote + 1)
        
        # Remove everything between quotes (including quotes)
        return rule[:first_quote - 1] + rule[second_quote + 2:]
    except (IndexError, AttributeError):
        return rule
    

def preproc_text_features(df):
    df['LudRules'] = df['LudRules'].apply(remove_game_name)
    df = clean(df, 'LudRules')
    df = clean(df, 'EnglishRules')

    df['LudRules_len'] = df['LudRules'].apply(len)
    df['EnglishRules_len'] = df['EnglishRules'].apply(len)
    added_cols = ['LudRules_len', 'EnglishRules_len']
    return df, added_cols
    

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

    df_train, added_cols = agent_position_feature(df_train)
    numerical_cols = numerical_cols + added_cols

    df_train, added_cols = preproc_text_features(df_train)
    numerical_cols = numerical_cols + added_cols

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

    # for CV purposes
    df_train = df_train.copy()
    df_train["utility_agent1_rank"] = (
        df_train["utility_agent1"].rank(method='dense', ascending=True).astype(int)
    )

    return df_train, numerical_cols, categorical_cols, encoder, scaler


def process_test_data(
    df_test: pd.DataFrame,
    numerical_cols: list,
    categorical_cols: list,
    encoder: OrdinalEncoder,
    scaler: StandardScaler = None
):
    df_test = split_agent_fields(df_test)
    df_test,_ = agent_position_feature(df_test)
    df_test,_ = preproc_text_features(df_test)

    # Apply ordinal encoding to categorical columns
    df_test[categorical_cols] = encoder.transform(df_test[categorical_cols])

    # Fit and transform the numerical columns of df_test
    if scaler is not None:
        df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])

    df_test[numerical_cols] = df_test[numerical_cols].astype(np.float32)
    df_test[categorical_cols] = df_test[categorical_cols].astype(np.int32)

    return df_test
