import os
from joblib import dump, load
import inspect
import importlib
from datetime import date
import json
from copy import deepcopy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #, MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import (
    roc_auc_score, 
    confusion_matrix, 
    multilabel_confusion_matrix,
    r2_score,
    mean_squared_error,
)

from . import utility_functions


'''
High level class for a pipeline using the low-level tool in utility_functions.py

3 majors Areas:

I. Train and test your model.
    1. Load Data | Pipeline.load_data()
    2. Load feature(s) and target(s) | Pipeline.load_X_y
        Columns to be used. (** you can directly use the setters for both)
    2.5 Apply custom transformation to column(s) in self.X
    3. Split into Train and Test | Pipeline.split()
    4. Preprocessing | Pipeline.preprocessing()
        This step is by far the most complex. You need to load the different
        mappings for different features.
    5. Scale data. | Pipeline.scale()
    6. Train your model | Pipeline.train()

(I option B):
    You can use self.pipeline() to automate this first step.
    The big advantages are:
        - K-fold.
        - Custom learning curves.

II. Deploy your model.
    This will save all the parameters and functions into the corresponding directory in
    ./models/{model_name}/


III. Make predictions
    You only need to specify the name of the model to load and the path of the data used.
    All the results will be saved in ./models/{model_name}/results.csv

    For predictions you could do:
    A)
        self.predict()
    B)
        set self._deployment = True
        self.load_data()
        self.load_model()
        self.load_X_y() | (self.X_columns = [] & self.y_columns = '' or [])
        self.preprocessing()
        self.scale()

        and you can make predictions with sef.model.predict()
'''


class Pipeline:
    # Set to None to try different values (cross validation)
    RANDOM_STATE = 4

    def __init__(self, nrows=None, random_state=-1):
        '''
        Args:
            nrows (int, default=None): Used in self.load_data()
                Sometimes we want to try things fast and the data is too big.
                This is why we load the data partially to try things out.
                It has to be set to None when you want to train on the whole dataset.
            random_state (int, default=-1): The actual default os set to cls.RANDOM_STATE.
                We actually want the option of None for randomness, so we use a proxy -1.
        '''
        self.nrows = nrows
        self.random_state = type(self).RANDOM_STATE if random_state == -1 else random_state
        
        # Property for self.model
        self._model = None

        # Internals for attributes
        self._X = None
        self._y = None

        # Dimension for target. It gets defined along self.y
        self.ydim = None
        
        # Important flag to separate training VS deployment | predictions
        self._deployment = False
        
        # Internal dictionary used to save all the informantion necessary
        # when self._deployment = True
        self._information = {'apply_X': []}
        
        # This flag is used for predictions is extra self.df
        # manipulation is needed and done manually.
        self._data_loaded = False
        
        # The following flags are used to mark steps
        # applied in training and help validate the right order.
        self._preprocessed = False
        self._split = False
        self._scale = False
        self._calculate_class_weights = True
        
        # Flag to know if X_columns has been set.
        self._X_columns_set = False

        ### UNDER REVIEW ###
        # Model type
        self.model_type = 'classification'
        
    @property
    def scoring(self):
        return 'roc_auc' if self.model_type == 'classification' else 'neg_root_mean_squared_error'

        ##########

    def load_data(self, path_or_dataframe: str or pd.DataFrame) -> None:
        '''
        Loads the data from csv file to pd.DataFrame or
            pass a pd.DataFrame directly. 

        Note:
            This method is used for both Deployment and Prediction.
        Args:
            - path_or_dataframe (str or pd.DataFrame): Filepath for csv file with data
                or pd.DataFrame
        Returns:
            - None
        '''
        # Load data
        if type(path_or_dataframe) == pd.DataFrame:
            df = path_or_dataframe
        else:
            df = pd.read_csv(filepath_or_buffer=path_or_dataframe, nrows=self.nrows)
        
        # Set dataframe.
        self.df = df
        
        # Flag (used for predictions mainly).
        self._data_loaded = True

    @property
    def X_columns(self):
        return self._X_columns

    @X_columns.setter
    def X_columns(self, value: list):
        '''
        Automatically sets self.X with all the proper validations.
        '''
        # List validation.
        if value is not None and type(value) != list:
            raise ValueError('You must pass a list of features for X_columns.')
        
        # Retrieve values if deployment is True (also for predictions)
        if self._deployment is True:
            value = self._information['X_columns']
        
        # Validate value integrity.
        for val in value:
            if val not in self.df.columns:
                raise ValueError(f"{val} not present in self.df.columns.")
        
        # Setting self.X
        self._X = self.df[value]
        
        # Setter
        self._X_columns = value
        
        # Write X_columns in self._information
        self._information['X_columns'] = self._X_columns
        
        # Flag this event
        self._X_columns_set = True

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def y_columns(self):
        return self._y_columns

    @y_columns.setter
    def y_columns(self, value: str or list):
        '''
        Automatically sets self.y with all the proper validations.
        '''
        # Checking right flag.
        if self._deployment is True:
            raise ValueError('You cannot set self.y_columns when self._deployment is True.')
        
        # Validate value integrity
        if isinstance(value, list):
            for val in value:
                if val not in self.df.columns:
                    raise ValueError(f"{val} not present in self.df.columns.")
        else:
            if value not in self.df.columns:
                raise ValueError(f"{value} not present in self.df.columns.")
        
        # Small workaround when target is 1 column.
        if isinstance(value, list) and len(value) < 2:
            value = value[0]

        # Setter
        self._y_columns = value
        
        # Write y columns(s) in self._information
        self._information['y_columns'] = self._y_columns
        
        # Setting self.y
        self._y = self.df[self._y_columns]

        # N Dimensions for target
        self.ydim = self._y.ndim 

    def load_X_y(self, X_columns: list=None, y_columns: list or str=None):
        '''
        For compatibility with other older versions you can either use:

            a)
                self.X_columns = ['x1','x2', ....]
                self.y_columns = 'y' OR ['y1', 'y2']
            b)
                self.load_X_y('x1','x2', ....], 'y' OR ['y1', 'y2'])

        ** All validations take place in the setters.
        '''
        self.X_columns = X_columns
        self.y_columns = y_columns

    def apply_X(self, function, columns: list):
        '''
        Custom transformation to self.X

        **This transformation is done after self.load_X_y()

        Args:
            - func (function): Function applied.
            - columns (list): List of columns to which the transformation applies.
        '''
        # Order validation
        if self._X_columns_set is False:
            raise AssertionError('You need to set self.X_columns or self.load_X_y() before.')
        
        # Application of custom transformer.
        self.X = function(self.X, columns)
        
        # Write information necessary in self._information
        # to be used when deploying or predicting.
        self._information['apply_X'].append({
            'columns': columns,
            'function_name': function.__name__,
            'function': function
        })

    def split(self, test_size: float=0.10, stratify=False):
        '''
        Simple splitter for train and test data.

        Args:
            - test_size (float, default=0.10)
            - stratify (bool, default=False): Keep class imbalance for
                splitting.
        Returns:
            - None
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            train_size=None,
            random_state=self.random_state,
            shuffle=True,
            stratify=self.y if stratify else None
        )
        self._split = True

    def preprocessing(
        self,
        cat_mapping: dict=None,
        cat_options_mapping: dict=None,
        boolean_mapping: dict=None,
        continuous_mapping: dict=None
        ):
        '''
        Handles preprocessing of X_train & X_test OR X (predicting)

        6 STEPS
            1. Categorical feature mapping
            2. Categotorical feature with options mapping
            3. Booleans mapping
            4. Continuous feature mapping
            5. Cleaning and keeping columns
            6. Flag self._preprocessed as True

        Args:
            - cat_mapping (dict, default=None): Mapping for category columns with
                single values per row. Ex: {'feature_1': 1} 
            - cat_options_mapping (dict, default=None): Mapping for category that have
                many options per row. Ex: {'feature_1': 1}
            - boolean_mapping (dict, default=None): Mapping for booleans, to 1.0 and 0.0
            - continuous_mapping (dict, default=None): Mapping for NaN.

        Returns:
            - None
        '''
        # I. CATEGORICAL FEATURES
        if self._deployment is False and cat_mapping:
            self.X_train, self.X_test = utility_functions.cat_feature_pipeline(
                data=self.X_train,
                mapping=cat_mapping,
                test=self.X_test
            )
            self._information['cat_mapping'] = cat_mapping
        elif self._information.get('cat_mapping'):
            self.X = utility_functions.cat_feature_pipeline(
                data=self.X,
                mapping=self._information.get('cat_mapping'),
                test=None
            )
        
        # II. CATEGORIES WITH OPTIONS
        if self._deployment is False and cat_options_mapping:
            self.X_train, self.X_test = utility_functions.cat_feature_pipeline(
                data=self.X_train,
                mapping=cat_options_mapping,
                test=self.X_test,
                options=True
            )
            self._information['cat_options_mapping'] = cat_options_mapping
        elif self._information.get('cat_options_mapping'):
            self.X = utility_functions.cat_feature_pipeline(
                data=self.X,
                mapping=self._information.get('cat_options_mapping'),
                test=None,
                options=True
            )
        
        # III. BOOLEAN
        if self._deployment is False and boolean_mapping:
            for feature, mapping in boolean_mapping.items():
                self.X_train, self.X_test = utility_functions.boolean_pipeline(
                    data=self.X_train,
                    columns_or_dict=[feature],
                    test=self.X_test,
                    mapping=mapping
                )
            self._information['boolean_mapping'] = boolean_mapping
        elif self._information.get('boolean_feature_list'):
            self.X = utility_functions.boolean_pipeline(
                data=self.X,
                columns_or_dict=self._information.get('boolean_feature_list'),
                test=None,
                mapping=self._information.get('boolean_feature_mapping')
            )
        
        # IV. CONTINUOUS
        if self._deployment is False and continuous_mapping:
            self.X_train, self.X_test = utility_functions.continuous_feature_pipeline(
                data=self.X_train,
                mapping=continuous_mapping,
                test=self.X_test
            )
            self._information['continuous_mapping'] = continuous_mapping
        elif self._information.get('continuous_mapping'):
            self.X = utility_functions.continuous_feature_pipeline(
                data=self.X,
                mapping=self._information.get('continuous_mapping'),
                test=None
            )
        
        # V. KEEP COLUMNS
        if self._deployment is False:
            self.X_test = utility_functions.keep_trainer_columns(
                trainer_columns=self.X_train.columns,
                target_dataframe=self.X_test
            )
            self._information['final_columns'] = self.X_train.columns.tolist()
        else:
            self.X = utility_functions.keep_trainer_columns(
                trainer_columns=self._information['final_columns'],
                target_dataframe=self.X
            )

        # VI. This flags the order of custom function for deployment and predictions
        self._preprocessed = True

    def scale(self, scaler=StandardScaler()):
        '''
        Depending on self_deployment, we scaled self.X_train and self.X_test
        OR self.X (predictions).

        Args:
            - scaler (default="sklearn.preprocessing.StandardScaler")

        '''
        if self._deployment is True:
            self.X_scaled = pd.DataFrame(
                data=self.scaler.transform(self.X),
                columns=self.X.columns
            )
        else:
            self.scaler = scaler
            self.X_train_scaled = pd.DataFrame(
                data=self.scaler.fit_transform(self.X_train),
                columns=self.X_train.columns
            )
            self.X_test_scaled = pd.DataFrame(
                data=self.scaler.transform(self.X_test),
                columns=self.X_test.columns
            )

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        '''
        This could be refactored.

        Some considerations:
            - If model accepts class weight for imbalanced
                classification, it automatically tunes it.

        Notes:
            - There is a distinction here for ndim == 1 or 2
            for y. This can be refactored.

        '''
        if self._deployment is False:
            if 'random_state' in dir(value):
                value.random_state = self.random_state

            if self._calculate_class_weights:
                # Predicting sequences
                if self.ydim == 2:
                    class_weights = []
                    for col in self.y_columns:
                        vc = self.y_train[col].value_counts()
                        vc  = vc / self.y_train.shape[0]
                        vc = 1 - vc
                        vc = vc.to_dict()
                        class_weights.append(vc)
                    self.class_weight_ = class_weights

                else:
                    # Recheck this part.
                    classes = np.unique(self.y_train)
                    class_weight_array = class_weight.compute_class_weight(
                        class_weight='balanced',
                        classes=classes,
                        y=self.y_train
                    )
                    self.class_weight_ = dict(zip(classes, class_weight_array))

                    # Fix for dtypes no json serializable
                    temp_dict = {}
                    keys_ = list(self.class_weight_.keys())
                    for key in keys_:
                        temp_dict[key.item()] = self.class_weight_.pop(key)
                    self.class_weight_ = temp_dict
                
                if 'class_weight' in dir(value):
                    value.class_weight = self.class_weight_
        # Setter
        self._model = value
        
        # Saving model parameters in self._information
        if 'get_params' in dir(self._model):
            self._information['model_params'] = self._model.get_params()

    def train(
        self,
        show_plots=True,
        corr_th=0.8,
        scoring=None,
        verbose=True
        ):
        '''
        Args:
            - show_plots
        '''
        # Validation of steps.
        if self._model is None:
            raise ValueError('You have not loaded a model.')
        
        # Fitting thr model
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Running internal method for analysis.
        self._results(
            show_plots,
            corr_th=corr_th,
            scoring=scoring or self.scoring,
            verbose=verbose
        )

    def pipeline(
        self,
        path_or_dataframe: str or pd.DataFrame,
        X_columns: list,
        y_columns: str or list,
        model,
        X_all_but: bool=False,
        k_folds=5,
        test_size=0.2,
        stratify=True,
        apply_X=None,
        cat_mapping=None,
        cat_options_mapping=None,
        boolean_mapping=None,
        continuous_mapping=None,
        verbose=True,
        corr_th=0.8,
        scoring=None,
        learning_curve=False
        ):
        '''
        Easy wrapper to cross-validate using kfold and have more
        robust results.

        Args:
            - path_or_dataframe (str or pd.DataFrame): If str, path for data.
                You can also pass the dataframe directly.
            - X_columns (list): List of features to use.
                You can pass a list of the features, or consider all features except for {list}.
                You need to pass X_all_but = True in that case.
            - y_columns (str or list): Target variables(s).
            - model: Model to be used. It triggers setter self.model = model.
            - X_all_but (bool, default=False): Include all columns but the ones
                specified in X_columns.
            - apply_X (dict or list(dict), default=None): If provided (list of
                ) dictionaries with {"function": val, "columns": []}

        Pending:
            Learning curves.

        '''
        # 1. Load data -> self.df (You can previosuly load data if necessary)
        if self._data_loaded is False:
            self.load_data(path_or_dataframe=path_or_dataframe)

        stratify = stratify if self.model_type == 'classification' else False

        # 2. Setters for features and target(s)
        self.X_columns = X_columns if X_all_but is False else self.df.drop(X_columns, axis=1).columns.tolist()
        self.y_columns = y_columns
        
        # apply_x transformations
        if apply_X:
            if isinstance(apply_X, dict):
                apply_X = [apply_X]

            for d in apply_X:
                self.apply_X(function=d['function'], columns=d['columns'])

        # ALL AGGREGATE SCORING
        self.agg_ras_train = []
        self.agg_ras_test = []

        self.agg_train_score = []
        self.agg_test_score = []

        self.agg_train_cm = []
        self.agg_test_cm = []

        self.agg_train_r2_score = []
        self.agg_test_r2_score = []

        self.agg_train_rsme = []
        self.agg_test_rsme = []

        self.agg_fi = []

        self.agg_df_corr = []

        for k_fold in range(k_folds):
            self.random_state = k_fold

            # 3. Split
            self.split(test_size=test_size, stratify=stratify)

            # 4. Preprocessing
            self.preprocessing(
                cat_mapping=cat_mapping,
                cat_options_mapping=cat_options_mapping,
                boolean_mapping=boolean_mapping,
                continuous_mapping=continuous_mapping
            )

            # 5. Scale
            self.scale()

            # 6. Model
            self.model = model

            # 7. Train
            self.train(
                show_plots=False,
                corr_th=0,
                scoring=scoring,
                verbose=False
            )
            # Aggregates and averages
            if self.model_type == 'classification':
                self.agg_ras_train.append(self.ras_train)
                self.agg_ras_test.append(self.ras_test)

                self.agg_train_cm.append(self.train_cm)
                self.agg_test_cm.append(self.test_cm)
                
            else:
                self.agg_train_r2_score.append(self.train_r2)
                self.agg_test_r2_score.append(self.test_r2)
                                   
                self.agg_train_rsme.append(self.train_rsme)
                self.agg_test_rsme.append(self.test_rsme)

            self.agg_train_score.append(self.train_score)
            self.agg_test_score.append(self.test_score)

            self.agg_fi.append(self.df_fi.T)

            self.agg_df_corr.append(self.df_corr)

        # Scores
        self.avg_train_score = np.mean(self.agg_train_score)
        self.avg_test_score = np.mean(self.agg_test_score)

        if self.model_type == 'classification':
            # ROC AUC Scores
            self.avg_ras_train = np.mean(self.agg_ras_train)
            self.avg_ras_test = np.mean(self.agg_ras_test)

            # Confusion Matrices
            self.avg_train_cm = np.mean(self.agg_train_cm, axis=0).round(0)
            self.avg_test_cm = np.mean(self.agg_test_cm, axis=0).round(0)

        else:
            self.avg_train_r2_score = np.mean(self.agg_train_r2_score)
            self.avg_test_r2_score = np.mean(self.agg_test_r2_score)

            self.avg_train_rsme = np.mean(self.agg_train_rsme)
            self.avg_test_rsme = np.mean(self.agg_test_rsme)

        # Feature Importances.
        agg_fi_features = self.agg_fi[0].columns
        agg_fi_columns = self.agg_fi[0].index

        # Temporary fix due to unmatching feature importance.
        r = []
        for df in self.agg_fi:
            try:
                r.append(df[agg_fi_features].T)
            except:
                pass
        self.agg_fi = r.copy()
        # self.agg_fi = [df[agg_fi_features].T for df in self.agg_fi]
        self.avg_fi = pd.DataFrame(index=agg_fi_features, columns=agg_fi_columns, data=np.mean(self.agg_fi, axis=0))
        self.avg_fi.sort_values(by='feature_importance', ascending=False, inplace=True)

        # Correlation Matrix.
        self.avg_df_corr = pd.concat(self.agg_df_corr, axis=0).groupby(['column_1', 'column_2']).mean().reset_index()
        self.avg_df_corr['temp'] = self.avg_df_corr['corr'].abs()
        self.avg_df_corr.sort_values('temp', ascending=False, inplace=True)
        self.avg_df_corr = self.avg_df_corr[self.avg_df_corr['temp'] >= corr_th]
        self.avg_df_corr.drop('temp', axis=1, inplace=True)
        self.avg_df_corr.reset_index(inplace=True, drop=True)

        if learning_curve:
            lc = utility_functions.plot_learning_curve(
                estimator=self.model,
                X=self.X_train_scaled,
                y=self.y_train,
                scoring=scoring or self.scoring,
                cv=10,
                n_jobs=-1,
                save_path=None
            )
            lc.show()

        if verbose:
            spaces = 5
            model_str = ' '.join(str(model).replace("\n", "").split())

            ### Target Imbalance formatting

            ### !!! Rewrite this part with better var names.

            if self.model_type == 'classification':

                target_imbalances = []
                iter_y_cols = self.y_columns if self.ydim == 2 else [self.y_columns]
                for y_col in iter_y_cols:
                    temp_fix = self.y if self.ydim == 1 else self.y[y_col]
                    val = (temp_fix.value_counts() / self.y.shape[0]).to_dict()
                    fmt = '\n'.join([f"\tClass {i}: {round(j, 4)}" for i, j in val.items()])
                    fmt = f'{y_col}\n{fmt}\n'
                    target_imbalances.append(fmt)
                target_imbalances = '\n'.join(target_imbalances)

                # Confusion matrix imbalances
                iter_cm_train = self.avg_train_cm if self.ydim == 2 else [self.avg_train_cm]
                iter_cm_test = self.avg_test_cm if self.ydim == 2 else [self.avg_test_cm]

                all_fmt_cm_train = []
                for cm, y_ in zip(iter_cm_train, iter_y_cols):
                    fmt_cm = f'''
{y_}
   {' ' * (spaces - 2)}PN | PP
TN {' ' * (spaces - len(str(cm[0][0]).split('.')[0]))}{cm[0][0] :0.0f} | {cm[0][1] :0.0f}
TP {' ' * (spaces - len(str(cm[1][0]).split('.')[0]))}{cm[1][0] :0.0f} | {cm[1][1] :0.0f}
           '''
                    all_fmt_cm_train.append(fmt_cm)
                all_fmt_cm_train = ''.join(all_fmt_cm_train)

                all_fmt_cm_test = []
                for cm, y_ in zip(iter_cm_test, iter_y_cols):
                    fmt_cm = f'''
{y_}
   {' ' * (spaces - 2)}PN | PP
TN {' ' * (spaces - len(str(cm[0][0]).split('.')[0]))}{cm[0][0] :0.0f} | {cm[0][1] :0.0f}
TP {' ' * (spaces - len(str(cm[1][0]).split('.')[0]))}{cm[1][0] :0.0f} | {cm[1][1] :0.0f}
           '''
                    all_fmt_cm_test.append(fmt_cm)
                all_fmt_cm_test = ''.join(all_fmt_cm_test)


                analysis = f'''
########################################

PIPELINE ANALYSIS - DITAT_ML - ditat.io

DESCRIPTION:

- Model: {model_str}
- KFold: n = {k_folds}
- Data shape: {self.df.shape}
- Test Size %: {test_size}


INDICATORS:

########################
    Target(s)
{target_imbalances}

########################

    - Train Score   : {self.avg_train_score.round(4)}
    - Test Score    : {self.avg_test_score.round(4)}

    - Auc Train     : {self.avg_ras_train.round(4)}
    - Auc  Test     : {self.avg_ras_test.round(4)}

########################
    Confusion Matrix:
** TRAIN **
{all_fmt_cm_train}
    
** TEST **
{all_fmt_cm_test}

########################

    - Feature Importance (Displaying max. 60 rows)
{self.avg_fi.head(60)}

########################

    - Feature Correlation (th >= {corr_th}) (Displaying max. 60 rows)
{self.avg_df_corr.head(60)}

########################
'''
            else:
                analysis = f'''
########################################

PIPELINE ANALYSIS - DITAT_ML - ditat.io

DESCRIPTION:

- Model: {model_str}
- KFold: n = {k_folds}
- Data shape: {self.df.shape}
- Test Size %: {test_size}


INDICATORS:

########################
    Target(s)
(Pending information about target distribution)

########################

    - rsme Score   : {self.avg_train_rsme.round(4)}
    - rsme Score    : {self.avg_test_rsme.round(4)}

    - r2 Train    : {self.avg_train_r2_score.round(4)}
    - r2 Test     : {self.avg_test_r2_score.round(4)}

########################

    - Feature Importance (Displaying max. 60 rows)
{self.avg_fi.head(60)}

########################

    - Feature Correlation (th >= {corr_th}) (Displaying max. 60 rows)
{self.avg_df_corr.head(60)}

########################
'''
            print(analysis)

    def _results(
        self,
        show_plots=True,
        save=False,
        save_path=None,
        corr_th=0.8,
        verbose=True,
        scoring=None,
        cv=5,
        other_cols=None
        ):
        '''
        '''
        # Scoring depends on self.model_type
        scoring = scoring or self.scoring

        # Mapping variables according to self._deployment
        if self._deployment:
            verbose = False
            corr_th = 0
            save_path = self.model_path
        
        # Create 3 dataframes for results.
        self.train_results = pd.DataFrame()
        self.test_results = pd.DataFrame()
        self.full_results = pd.DataFrame()
        
        # Mapping features and target(s) according to self._deployment
        X_ = self.X_scaled if self._deployment else self.X_train_scaled
        y_  = self.y if self._deployment else self.y_train
        
        # Process feature importance if self.model has that attribute.
        if 'feature_importances_' in dir(self.model):
            self.df_fi = utility_functions.feature_importance(
                feature_importances=self.model.feature_importances_,
                data=X_,
                verbose=verbose
            )
            if save is True:
                self.df_fi.to_csv(os.path.join(self.model_path, 'feature_importances.csv'))
        
        # Populating train and test.
        if not self._deployment:
            # Train dataframes

            if self.ydim == 1:
                train_predict_cols = 'train_predict'
                test_predict_cols = 'test_predict'

                train_predict_proba_cols = 'train_predict_proba'
                test_predict_proba_cols = 'test_predict_proba'

                self.train_results['y_train'] = self.y_train
                self.train_results['train_predict'] = self.model.predict(self.X_train_scaled)
            
                self.test_results['y_test'] = self.y_test
                self.test_results['test_predict'] = self.model.predict(self.X_test_scaled)


            else:
                train_predict_cols = [f'train_predict_{i}' for i in self.y_columns]
                test_predict_cols = [f'test_predict_{i}' for i in self.y_columns]

                train_predict_proba_cols = [f'train_predict_proba_{i}' for i in self.y_columns]
                test_predict_proba_cols = [f'test_predict_proba_{i}' for i in self.y_columns]

                self.train_results[[f'y_train_{i}' for i in self.y_columns]] = self.y_train
                self.train_results[train_predict_cols] = self.model.predict(self.X_train_scaled)

                self.test_results[[f'y_test_{i}' for i in self.y_columns]] = self.y_test
                self.test_results[test_predict_cols] = self.model.predict(self.X_test_scaled)

            # Test dataframes
            
            # Train and test accuracy scores.
            self.train_score = self.model.score(self.X_train_scaled, self.y_train)
            self.test_score =self.model.score(self.X_test_scaled, self.y_test)

            # Verbose scores
            if verbose:
                print(f'Score Train {round(self.train_score, 4)}')
                print(f'Score Test {round(self.test_score, 4)}')
            
            if self.model_type == 'classification':
                # Confusion matrices
                if self.ydim == 1:
                    self.train_cm = confusion_matrix(self.y_train, self.train_results[train_predict_cols])
                    self.test_cm = confusion_matrix(self.y_test, self.test_results[test_predict_cols])
                
                else:
                    self.train_cm = multilabel_confusion_matrix(self.y_train, self.train_results[train_predict_cols])
                    self.test_cm = multilabel_confusion_matrix(self.y_test, self.test_results[test_predict_cols])

                # Verbose confusion matrices
                if verbose:
                    print('Train\n', self.train_cm)
                    print('Test\n', self.test_cm)

            else:
                # Only one output supported for now
                self.train_r2 = r2_score(self.y_train, self.train_results[train_predict_cols])
                self.test_r2 = r2_score(self.y_test, self.test_results[test_predict_cols])

                self.train_rsme = mean_squared_error(self.y_train, self.train_results[train_predict_cols], squared=False)
                self.test_rsme = mean_squared_error(self.y_test, self.test_results[test_predict_cols], squared=False)

                if verbose:
                    print(f'Train r2 Score {round(self.train_r2, 4)}')
                    print(f'Test r2 Score {round(self.test_r2, 4)}')

                    print(f'Train rsme Score {round(self.train_rsme, 4)}')
                    print(f'Test rsme Score {round(self.test_rsme, 4)}')
            
            # If predict_proba is a method of self.model
            if 'predict_proba' in dir(self.model):
                # Adding predict_proba to both dataframes
                if self.ydim == 1:
                    self.train_results[train_predict_proba_cols] = self.model.predict_proba(self.X_train_scaled)[:, 1]
                    self.test_results[test_predict_proba_cols] = self.model.predict_proba(self.X_test_scaled)[:, 1]

                else:
                    # self.train_results[train_predict_proba_cols] = np.concatenate(self.model.predict_proba(self.X_train_scaled), axis=1)[:, (1, 3)]
                    # self.test_results[test_predict_proba_cols] = np.concatenate(self.model.predict_proba(self.X_test_scaled), axis=1)[:, (1, 3)]
                    self.train_results[train_predict_proba_cols] = np.concatenate(self.model.predict_proba(self.X_train_scaled), axis=1)[:, 1::2]
                    self.test_results[test_predict_proba_cols] = np.concatenate(self.model.predict_proba(self.X_test_scaled), axis=1)[:, 1::2]

                # Creating Roc Auc Scores as attributes for both sets.
                self.ras_train = roc_auc_score(self.y_train, self.train_results[train_predict_proba_cols])
                self.ras_test = roc_auc_score(self.y_test, self.test_results[test_predict_proba_cols])
                
                # Verbose AUC
                if verbose:
                    print(f"Roc Auc score Training: {round(self.ras_train, 4)}")
                    print(f"Roc Auc score Testing: {round(self.ras_test, 4)}")
        else:
            # Similar to the previous if, but for self._deployment == True
            if self.ydim == 1:
                predict_cols = 'predict'
                predict_proba_cols = 'predict_proba'

                self.full_results['y'] = self.y
            
            else:
                predict_cols = [f'predict_{i}' for i in self.y_columns]
                predict_proba_cols = [f'predict_proba_{i}' for i in self.y_columns]

                self.full_results[[f'y_{i}' for i in self.y_columns]] = self.y
            
            self.full_results[predict_cols] = self.model.predict(self.X_scaled)
            
            # Accuracy score as attribute.
            self.full_score = self.model.score(self.X_scaled, self.y)

            if verbose:
                print('Score Full', round(self.full_score, 4))


            if self.model_type == 'regression':
                self.full_r2 = r2_score(self.y, self.full_results[predict_cols]) 
                self.full_rsme = mean_squared_error(self.y, self.full_results[predict_cols], squared=False)

                if verbose:
                    print(f'r2 Score Full {round(self.full_score, 4)}')
                    print(f'rsme Score Full {round(self.full_score, 4)}')
                    
            
            # If predict_proba is a method of 
            if 'predict_proba' in dir(self.model):
                # This next section could be only one
                if self.ydim == 1:
                    self.full_results[predict_proba_cols] = self.model.predict_proba(self.X_scaled)[:, 1]
                
                else:
                    self.full_results[predict_proba_cols] = np.concatenate(self.model.predict_proba(self.X_scaled), axis=1)[:, 1::2]

                self.ras_full = roc_auc_score(self.y, self.full_results[predict_proba_cols])
                
                if verbose:
                    print(f"Roc Auc score Full: {round(self.ras_full, 4)}")
            
            # Add extra columns if needed
            if other_cols and all(item in self.df.columns for item in other_cols):
                for col in other_cols:
                    self.full_results[col] = self.df[col]
            
            # Saving results
            if save:
                self.full_results.to_csv(os.path.join(self.model_path, 'full_results.csv'), index=False)

        # Correlation analysis
        self.df_corr = utility_functions.find_high_corr(
            dataframe=X_,
            threshold=corr_th,
            verbose=verbose,
            save=save,
            save_path=save_path
        )
        # Learning Curve
        if show_plots:
            plot1 = utility_functions.plot_learning_curve(
                estimator=self.model,
                X=X_,
                y=y_,
                scoring=scoring,
                estimator_name='estimator',
                cv=cv,
                save_path=save_path
            )
            if not self._deployment:
                plot1.show()

    def deploy(
        self,
        name,
        directory='models',
        overwrite=False,
        other_cols=None,
        save_plots=True,
        add_date_to_save_path=False
        ):
        '''
        We train the model on the whole dataset once we have decided
        we are satified with the model performance. This will save the model
        and all its parameters in /{directory}/{name}/(_{self._information['daye']})/

        1. Create path for deployed model.
        2. Preprocess, scale and train on the full dataset.
        3. Save model in path.

        Args:
            - name (str): Name of your model. It will be save with name     # (not anymore)+ "_YYYYMMDD"
            - directory (str, default='models'): folder to use for creating the path.
            - overwrite (bool, default=False): Raise error if folder exists or overwrite.
            - other_cols (list, default=None): Other columns in the original self.df
                to include in the full_results.csv
            - save_plots (bool, default=True): Similar tro training, whether you want to save
                the learning curves and general data.
            - add_date_to_save_path (bool, default=False): Add date to model name for self.path

        '''
        # Set environment
        self._deployment = True

        # Set model timestamp
        self._information['date'] = str(date.today())
        
        # Save attributes.
        self.model_name = name
        self._information['model_name'] = self.model_name

        self._information['model_type'] = self.model_type

        self.model_path = os.path.join(os.getcwd(), directory, self.model_name)
        if add_date_to_save_path:
            self.model_path += f"_{self._information['date']}"
        
        # Handle pre-existing directory and overwrite it if needed.
        if overwrite and os.path.exists(self.model_path):
            for f in os.listdir(self.model_path):
                if os.path.isdir(f) and f != '__pycache__':
                    os.remove(os.path.join(self.model_path, f))
        else:
            os.makedirs(self.model_path)
        
        # Save all custom transformation to be applied for predictions
        with open(os.path.join(self.model_path, 'custom.py'), 'w') as file:
            for custom_function_info in self._information['apply_X']:
                f = custom_function_info['function']
                f_code = inspect.getsource(f)
                file.write(f_code)
        
        # All the following steps are a replication of the steps applied to the testing data.
        # but now on the whole dataset.

        # Preprocessing
        self.preprocessing()
        
        # Scaling
        self.scale()
        
        # Fitting model.
        self.model.fit(self.X_scaled, self.y)
        
        # Running Analytics
        self._results(other_cols=other_cols, show_plots=save_plots, save=save_plots)
        
        # Save model and scaler.
        dump(self.model, os.path.join(self.model_path, 'model.joblib'))
        dump(self.scaler, os.path.join(self.model_path, 'scaler.joblib'))
        
        # Saver self._information as as json (deepcopy)
        with open(os.path.join(self.model_path, 'information.json'), 'w') as file:
            # Creation of a deep copied self._information is necessary if we want to deploy the model more than once.
            _information_to_save = deepcopy(self._information)
            for custom_function_info in _information_to_save['apply_X']:
                del custom_function_info['function']
            json.dump(_information_to_save, file)

    def predict(
        self,
        model_name,
        model_dir='models',
        path_or_dataframe=None,
        save=True,
        other_cols=None
        ):
        '''
        *** For prediction ONLY ***

        1. Load data to make predictions on self.df
        2. Load model and information.
        3. Set self.X_columns as a subset of self.df
        3.5 Apply_X if there are any on self.X
        4. Preprocess data.
        5. Scale data.
        6. Predict

        Args:
            - model_name (str): Name of the model to load. It 
            - path_or_dataframe (str or pd.DataFrame, default=None): Where the data for the predicting is.
                You can also pass a dataframe
            - save (bool, default=True): Whether to save the prediction dataframe as
                results.csv
            - other_cols (list, default=None): Include other columns from the dataframe to
                be present in the predictions

        Return
            - results (pd.DataFrame): Dataframe containing predictions and other columns
                according to parameters passed.
        '''
        # Setting environment fro predictions
        self._deployment = True
        
        # Check and load data self.load_data() has not been triggered before.
        if self._data_loaded is False:
            if path_or_dataframe is None:
                raise ValueError('You have to provide a path_or_dataframe or self.load_data() to predict.')
            self.load_data(path_or_dataframe)
        
        # Loading model
        self.load_model(model_name, model_dir=model_dir)
        
        # Setting X_columns. It will look in self._information['X_columns']
        self.X_columns = None
        
        # Apply custom transformation to data if present in self._information['apply_X']
        for info in self._information['apply_X']:           
            custom_module = importlib.import_module(f'{model_dir}.{self.model_name}.custom')
            custom_function = getattr(custom_module, info['function_name'])
            self.X = custom_function(self.X, info['columns'])   
        
        # Setting ydim and y_columns for predictions
        self.ydim = 2 if isinstance(self._information['y_columns'], list) else 1
        predict_cols = [f"predict_{col}" for col in self._information['y_columns']] if self.ydim == 2 else 'predict'
        predict_proba_cols = [f"predict_proba_{col}" for col in self._information['y_columns']] if self.ydim == 2 else 'predict_proba'


        # Preprocess self.X
        self.preprocessing()
        
        # Scale self.X using the loaded scaler.                        
        self.scale(scaler=self.scaler)
        
        # Dataframe with predictions (and predict_proba if available).
        results = pd.DataFrame()
        
        if self.ydim == 2:
            results[predict_cols] = pd.DataFrame(self.model.predict(self.X_scaled))
        else:
            results[predict_cols] = self.model.predict(self.X_scaled)
        
        if 'predict_proba' in dir(self.model):
            if self.ydim == 2:
                results[predict_proba_cols] = np.concatenate(self.model.predict_proba(self.X_scaled), axis=1)[:, 1::2]
            else:
                results[predict_proba_cols] = self.model.predict_proba(self.X_scaled)[:, 1]

        # Add column(s) to predictions if needed. Used for indexing mostly.
        if other_cols and all(item in self.df.columns for item in other_cols):
            for col in other_cols:
                results[col] = self.df[col].values

        # Choose to save results if needed.
        if save is True:
            results.to_csv(os.path.join(self.model_path, 'results.csv'), index=False)
        return results

    def load_model(self, model_name: str, model_dir='models'):
        '''
        *** For prediction ONLY ***

        Loading of all the information of the model
        previously deployed and saved.

        Args:
            model_name (str): Name of the model to use.
        '''

        # Validate environment.
        if self._deployment is not True:
            raise AssertionError('self._deployment must be True.')
        
        # Setting model name and path.
        self.model_name = model_name
        self.model_path = os.path.join(os.getcwd(), model_dir, self.model_name)
        
        # Load model and scalers
        self.model = load(os.path.join(self.model_path, 'model.joblib'))
        self.scaler = load(os.path.join(self.model_path, 'scaler.joblib'))
        
        # Load self._information
        with open(os.path.join(self.model_path, 'information.json'), 'r') as file:
            self._information = json.load(file)



        
