from functools import wraps
import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from sklearn.metrics import SCORERS
from sklearn.preprocessing import MultiLabelBinarizer



'''
Low level functions used in machine learning. Especially in preprocessing and
general data manipulation.

These functions are "stateless" and are used in the High-Level API pipeline.py
'''


def baseline(func):
    @wraps(func)
    def wrapper(
        data: pd.DataFrame,
        columns: list=None,
        test=None,
        *args,
        **kwargs
        ):
        '''
        Decorator used for most functions that apply a transformation on
        X_train/X_test OR for the whole X.

        Then you need to simply implement a function decorating like:

        Ex.
            @baseline
            def custom_function(*args, **kawrgs):

        Args:
        - data (pandas.DataFrame): Dataframe for training.
        - columns (list): Columns used for the transformation
        - test (pandas.DataFrame, default=None): Dataframe for predictions.
            It can be None when training on full dataset
        - *args
        - **kwargs       

        Returns:
            - target_dataframe (pandas.DataFrame): It overwrites
                the dataframe with the transformations applied
                to it.
                It can return data OR (data, test) depending on the test parameter.

        '''
        data = data.copy()

        if test is not None:
            test = test.copy()

        for col in columns:
            if col in data.columns:
                data[col] = func(data, col, *args, **kwargs)

                if test is not None:
                    test[col] = func(test, col, *args, **kwargs)

        if test is not None:
            return data, test

        return data
    return wrapper


def time_it(text=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            '''
            Simple timer for functions, methods, allowing optional text.

            Args:
                - text (str, default=None): If given, it is appended in the output of the timer.

            Returns:
                - result: Function of method invoked being decorated.
            '''
            t1 = time.time()
            result = func(*args, **kwargs)
            time_out = f"{func.__qualname__} time: {(time.time() - t1):.7f}"
            if text:
                time_out += ", " + str(text)
            print(time_out)
            return result
        return wrapper
    return decorator


def dummies_with_limit(
    dataframe,
    col,
    limit=None,
    verbose=False,
    prefix=True,
    use_cols=None,
    return_unique_list=False,
    return_value_counts=False
    ):
    '''
    One hot enconding limiting the number of options.
    
    Uses df[col].value_counts().head() to know how many
    options of that column to use
        OR
    You can pass use_cols and strictly tell it what columns to use.

    Simple to analyze with feature importance.

    Args:
        - dataframe (pandas.DataFrame): Target dataframe.
        - col (str): Column to be transformed.
        - limit (int, default=None): Number of options
            to keep. drop_one takes place.
        - verbose (bool, default=False): print the value counts used
            for reference.
        - prefix (bool, default=True): prefix the new columns with {col}_{value}
            or just {value}.
        - use_cols (list, default=None): Keep desired columns instead of
            hierarchy according to value_counts()
        -return_unique_list (bool, default=False): Instead of returning the dataframe
            with the new columns, the list of unique values is returned instead.
        - return_value_counts(bool, default=False): Similar to return_unique_list,
            the value counts are returned as a dataframe with limit set.

    Returns:
        - dataframe (pandas.DataFrame): It overwrites
            the dataframe with the transformations applied
            to it.
    '''
    dataframe = dataframe.copy()

    vc = dataframe[col].value_counts()

    if use_cols:
        vc = pd.DataFrame(vc.loc[use_cols])
    else:
        vc = vc.head(limit)
    
    if verbose:
        print(vc)

    if return_unique_list is True:
        return vc.index.tolist()
    
    elif return_value_counts is True:
        return vc

    cols = vc.index
    dataframe.loc[~dataframe[col].isin(cols), col] = 'to_drop'

    dataframe = pd.get_dummies(
        dataframe,
        prefix=col if prefix else '',
        prefix_sep='_' if prefix else '',
        dummy_na=False,
        columns=[col],
        sparse=False,
        drop_first=False,
        dtype=None)

    if f'{col}_to_drop' in dataframe.columns:
        dataframe.drop(f'{col}_to_drop', axis=1, inplace=True)

    return dataframe


def dummies_with_options_and_limit(
    dataframe,
    col,
    limit=None,
    verbose=False,
    prefix=True,
    sep=',',
    use_cols=None,
    return_unique_list=False,
    return_value_counts=False
    ):
    '''
    Similar to dummies_with_limit(), in this case each value has a "sep"
    separated string with different options.

    The dummy columns are constructed from the pool of options.

    Note:
        - It assumes that the options comes in a "sep" separated string.
            Future implementations and options can be added.

    Args:
        - dataframe (pandas.DataFrame): Target dataframe.
        - col (str): Column to be transformed.
        - limit (int, default=None): Number of options
            to keep. drop_one takes place.
        - verbose (bool, default=False): print the value counts used
            for reference.
        - prefix (bool, default=True): prefix the new columns with {col}_{value}
            or just {value}.
        - sep (str, default=','): Separator to use to come up with the
            list of options per row.
        - use_cols (list, default=None): Keep desired columns instead of
            hierarchy according to value_counts()
        -return_unique_list (bool, default=False): Instead of returning the dataframe
            with the new columns, the list of unique values is returned instead.
        - return_value_counts(bool, default=False): Similar to return_unique_list,
            the value counts are returned as a dataframe with limit set.

    Returns:
        - dataframe (pandas.DataFrame): It overwrites
            the dataframe with the transformations applied
            to it.
    '''
    dataframe = dataframe.copy()

    dataframe[col].fillna('to_drop', inplace=True)

    dataframe[col] = dataframe[col].apply(lambda x: [i.strip() for i in x.split(sep)] if type(x) == str else [x])

    # subset = dataframe[col].str.join('|').str.get_dummies() # OLD Implementation. Much slower.
    
    mlb = MultiLabelBinarizer()
    subset = pd.DataFrame(mlb.fit_transform(dataframe[col]), columns=mlb.classes_, index=dataframe.index)

    if 'to_drop' in subset.columns:
        subset.drop('to_drop', axis=1, inplace=True)

    vc = subset.sum().sort_values(ascending=False)

    if use_cols:
        vc = pd.DataFrame(vc.loc[use_cols])
    else:
        vc = vc.head(limit)

    subset = subset[vc.index]

    if return_unique_list is True:
        return subset.columns.tolist()

    elif return_value_counts is True:
        return vc

    if prefix:
        subset = subset.add_prefix(f'{col}_')

    if verbose:
        print(vc)

    dataframe = dataframe.join(subset)
    dataframe.drop(col, axis=1, inplace=True)

    return dataframe


@baseline
def apply_function(data, col, function):
    '''
    Allows the application of a custom function on a column of the dataframe

    Note:
        Include the @baseline decorator, which means you can return:
            X_train, X_test or just X.
    '''
    return data[col].apply(function) 


@baseline
def fillna(data, col, value):
    return  data[col].fillna(value)


@baseline
def boolean_pipeline(df, col, mapping):
    '''
    Example: mapping = {False: "No", True: "Yes"}
    '''
    mapping = {
        mapping[False]: 0.,
        mapping[True]: 1.
    }

    return df[col].replace(mapping)


def keep_trainer_columns(
    trainer_columns: list,
    target_dataframe: pd.DataFrame
    ):
    '''
    Only use the columns used in training to match shapes.

    Args:
        - trainer_columns (list): Columns used in training.
        - target_dataframe (pandas.DataFrame): Dataframe for predictions.

    Returns:
        - target_dataframe (pandas.DataFrame): It overwrites
            the dataframe with the transformations applied
            to it.

    '''
    target_dataframe = target_dataframe.copy()

    columns_to_keep = [col for col in target_dataframe if col in trainer_columns]

    target_dataframe = target_dataframe[columns_to_keep]

    missing_columns = [col for col in trainer_columns if col not in target_dataframe.columns]

    for col in missing_columns:
        target_dataframe[col] = 0.0

    return target_dataframe


def cat_feature_pipeline(
    data: pd.DataFrame,
    mapping: dict,
    test: pd.DataFrame=None,
    options=False,
    final_columns=None,
    verbose=False
    ) -> pd.DataFrame:
    '''
    Only use the columns used in training to match shapes.

    Args:
        - data (pandas.DataFrame): Dataframe for training.
        - test (pandas.DataFrame): Dataframe for predictions.
            It can be None when training on full dataset
        - mapping (dict): Follows the logic
            mapping = {
                'feature_1': 2 # limit
                'feature_2': 1,
                etc....
            }
        - options (bool, default=False): choose between
            dummies_with_limit() and dummies_with_options_and_limit()
        - final_columns (list, default=None): When predicting,
            this list is passed to filter columns not used.
        - verbose (bool, default=False): print the value counts used
            for reference.

    Notes:
        ** Maybe add separator logic when with options


    Returns:
        - target_dataframe (pandas.DataFrame): It overwrites
            the dataframe with the transformations applied
            to it.  
    '''
    data = data.copy()

    if test is not None:
        test = test.copy()

    function = dummies_with_limit if options is False else dummies_with_options_and_limit

    for key, value in mapping.items():
        if key in data.columns:

            if isinstance(value, list):
                use_cols = value
                limit = None

            elif isinstance(value, int) or value is None:
                use_cols = None
                limit = value

            data = function(
                dataframe=data,
                col=key,
                limit=value if test is not None else None,
                verbose=verbose,
                use_cols=use_cols
            )
            if test is not None:
                test = function(
                    dataframe=test,
                    col=key,
                    limit=None,
                    use_cols=use_cols
                )
        else:
            continue

    if test is not None:
        return data, test

    return data


def find_high_corr(
    dataframe: pd.DataFrame,
    threshold: float=0.85,
    verbose: bool=False,
    save=False,
    save_path=None
    ) -> pd.DataFrame:
    '''
    Find correlation in dataframe above a threshold

    Args:
        - dataframe (pd.DataFrame): Feature(s) dataframe.
        - th (float, default=0.85): Show correlation which absolute value is >= th.
        - verbose (bool, default=False): Print the result.
        - save (bool, default=False): Save or not the result in save_path
        - save_path (str, default=None): If save == True, path to save the result.
    
    Returns:
        - df_corr (pd.DataFrame): Correlation dataframe.
    

    '''
    results = []
    pairs_used = []
    correlations = dataframe.corr()

    for index, value in correlations.iterrows():
        for i, corr_value in value.items():
            if index == i:
                continue
            if abs(corr_value) > threshold and \
            sorted([index, i]) not in pairs_used:
                pair = {
                    'value': corr_value,
                    'columns': [index, i]
                }
                pairs_used.append(sorted([index, i]))
                results.append(pair)
    
    df_corr = pd.DataFrame(
        data={
            'corr': [i['value'] for i in results],
            'corr_abs': [abs(i['value']) for i in results],
            'column_1': [i['columns'][0] for i in results],
            'column_2': [i['columns'][1] for i in results]
        }
    )
    df_corr = df_corr.sort_values('corr_abs', ascending=False).drop('corr_abs', axis=1).reset_index(drop=True)

    if verbose and results:
        print('-------------------')
        print(f'Showing correlations < - {threshold} and > {threshold} ')
        print(df_corr)

        print('-------------------')

    if save is True:
        df_corr.to_csv(os.path.join(save_path, 'feature_corr.csv'), index=False)

    return df_corr


def feature_importance(
    feature_importances: list or np.array,
    data: pd.DataFrame,
    verbose=False
    ):
    '''
    Convinient dataframe visualization of the features importances.

    Args:
        - feature_importances (list or np.array): Features importances.
        - data (pd.DataFrame): Original feature data where to extract the feature importances.
        - verbose (bool, default=False): Print the result.
    
    Returns:
        - df_fi (pd.DataFrame): Feature importance dataframe.

    '''
    df_fi = pd.DataFrame(
        feature_importances,
        index=data.columns,
        columns=['feature_importance']
    ).sort_values('feature_importance', ascending=False)
    
    df_fi['cum_fi'] = df_fi['feature_importance'].cumsum()
    
    if verbose:
        print('-------------------')
        print(df_fi)
        print('-------------------')

    return df_fi


def plot_learning_curve(
    estimator,
    X: pd.DataFrame or np.array,
    y: pd.DataFrame or np.array,
    estimator_name='estimator',
    scoring='accuracy',
    cv=10,
    n_jobs=-1,
    save_path=None
    ):
    '''
    Plots the {scoring} performance of the model as more data is used for training and testing.
    Very good indicator for overfitting.

    Args:
        - estimator: Model already fitted.
        - X (pd.DataFrame or np.array): Feature(s)
        - y (pd.DataFrame or np.array)
        - estimator_name(str, default='estimator'): Name of your model.
        - scoring (str, default='accuracy'): Performance indicator, it could be roc_auc among others.
            It has to exist in sklearn.metrics.SCORES
        - cv (int, default=10): Number of cross validation splits.
        - n_jobs (int, default=-1): How many processors to use.
        - save_path (int, default=None): If != None, then you have to provide the path where the
            image (plt) will be saves.

    Returns:
        - plt: pyplot figure.

    '''
    f, ax1 = plt.subplots(1, 1, figsize=(10, 7), sharey=True)

    scoring = scoring if scoring in SCORERS.keys() else 'accuracy'

    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        groups=None,
        train_sizes=np.linspace(0.1, 1.0, cv),
        cv=cv,
        scoring=scoring,
        exploit_incremental_learning=False,
        n_jobs=n_jobs,
        pre_dispatch='all',
        verbose=0,
        shuffle=True,
        random_state=None,
        error_score=np.nan,
        return_times=False
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="#ff9124")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124", label=f"Training {scoring}")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff", label=f"Cross-validation {scoring}")

    ax1.set_title(f"{estimator_name}", fontsize=14)
    ax1.set_xlabel('Training size')
    ax1.set_ylabel(scoring)
    ax1.grid(True)
    ax1.legend(loc="best")

    if save_path:
        path = os.path.join(save_path, f'lc_{estimator_name}.png')
        plt.savefig(path)

    return plt


# def clean_columns(cols_or_df: str or list or pd.DataFrame) -> str or list or pd.DataFrame:
#     '''
#     # DEPRECATED. Mssing with the original names, can bring problems.
#     Better formatting for dataframe columns or list of columns or single col (str)

#     Args:
#         - cols_or_df (str or list or pd.DataFrame)

#     Return
#         - columns or dataframe

#     '''
#     def clean(x):
#         return x.lower().strip().replace(" ", "_")

#     if isinstance(cols_or_df, list):
#         columns = [clean(i) for i in cols_or_df]
#         return columns

#     elif isinstance(cols_or_df, str):
#         return clean(cols_or_df)

#     else:
#         dataframe = cols_or_df.copy()
#         dataframe.columns = [clean(i) for i in dataframe.columns]
#         return dataframe