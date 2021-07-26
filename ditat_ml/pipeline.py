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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, confusion_matrix

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
    def X_columns(self, value: list or True):
        '''
        Automatically sets self.X with all the proper validations.

        If self.y has been set and value is True,
            value will be set to all the columns in self.df excluding the ones
            present in self.y_columns.
        '''
        if self.y is not None:
            value = self.df.drop(self.y_columns, axis=1).columns.tolist()
        # List validation.
        elif value is not None and type(value) != list:
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
        
        # Setter
        self._y_columns = value
        
        # Write y columns(s) in self._information
        self._information['y_columns'] = self._y_columns
        
        # Setting self.y

        self._y = self.df[self._y_columns]

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
                    columns=[feature],
                    test=self.X_test,
                    mapping=mapping
                )
            self._information['boolean_mapping'] = boolean_mapping
        elif self._information.get('boolean_feature_list'):
            self.X = utility_functions.boolean_pipeline(
                data=self.X,
                columns=self._information.get('boolean_feature_list'),
                test=None,
                mapping=self._information.get('boolean_feature_mapping')
            )
        
        # IV. CONTINUOUS
        if self._deployment is False and continuous_mapping:
            for feature, imputer in continuous_mapping.items():
                if feature not in self.X.columns:
                    continue
                if imputer == 'mean':
                    imputer = self.X_train[feature].mean()

                self.X_train, self.X_test = utility_functions.fillna(
                    data=self.X_train,
                    columns=[feature],
                    test=self.X_test,
                    value=imputer
                )
            self._information['continuous_mapping'] = continuous_mapping
        elif self._information.get('continuous_mapping'):
            for feature, imputer in self._information.get('continuous_mapping').items():
                if feature not in self._information['final_columns']:
                    continue
                if imputer == 'mean':
                    imputer = self.X[feature].mean()
                self.X = utility_functions.fillna(
                    data=self.X,
                    columns=[feature],
                    test=None,
                    value=imputer
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
        '''
        if self._deployment is False:
            if 'random_state' in dir(value):
                value.random_state = self.random_state

            if self._calculate_class_weights:
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
        scoring='roc_auc',
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
            scoring=scoring,
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
        cat_mapping=None,
        cat_options_mapping=None,
        boolean_mapping=None,
        continuous_mapping=None,
        verbose=True,
        corr_th=0.8,
        scoring='roc_auc'
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

        Pending:
            Learning curves.

        '''
        # ALL AGGREGATE SCORING
        self.agg_ras_train = []
        self.agg_ras_test = []

        self.agg_train_score = []
        self.agg_test_score = []

        self.agg_train_cm = []
        self.agg_test_cm = []

        self.agg_fi = []

        self.agg_df_corr = []

        for k_fold in range(k_folds):
            self.random_state = k_fold

            # 1. Load data -> self.df 
            self.load_data(path_or_dataframe=path_or_dataframe)

            # 2. Setters for features and target(s)
            self.y_columns = y_columns
            self.X_columns = X_columns

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
            self.agg_ras_train.append(self.ras_train)
            self.agg_ras_test.append(self.ras_test)
            self.agg_train_score.append(self.train_score)
            self.agg_test_score.append(self.test_score)

            self.agg_train_cm.append(self.train_cm)
            self.agg_test_cm.append(self.test_cm)

            self.agg_fi.append(self.df_fi.T)

            self.agg_df_corr.append(self.df_corr)

        # ROC AUC Scores
        self.avg_ras_train = np.mean(self.agg_ras_train)
        self.avg_ras_test = np.mean(self.agg_ras_test)

        # Scores
        self.avg_train_score = np.mean(self.agg_train_score)
        self.avg_test_score = np.mean(self.agg_test_score)

        # Confusion Matrices
        self.avg_train_cm = np.mean(self.agg_train_cm, axis=0).round(0)
        self.avg_test_cm = np.mean(self.agg_test_cm, axis=0).round(0)

        # Feature Importances.
        agg_fi_features = self.agg_fi[0].columns
        agg_fi_columns = self.agg_fi[0].index
        self.agg_fi = [df[agg_fi_features].T for df in self.agg_fi]
        self.avg_fi = pd.DataFrame(index=agg_fi_features, columns=agg_fi_columns, data=np.mean(self.agg_fi, axis=0))
        self.avg_fi.sort_values(by='feature_importance', ascending=False, inplace=True)

        # Correlation Matrix.
        self.avg_df_corr = pd.concat(self.agg_df_corr, axis=0).groupby(['column_1', 'column_2']).mean().reset_index()
        self.avg_df_corr['temp'] = self.avg_df_corr['corr'].abs()
        self.avg_df_corr.sort_values('temp', ascending=False, inplace=True)
        self.avg_df_corr = self.avg_df_corr[self.avg_df_corr['temp'] >= corr_th]
        self.avg_df_corr.drop('temp', axis=1, inplace=True)
        self.avg_df_corr.reset_index(inplace=True, drop=True)

        if verbose:
            spaces = 5
            model_str = ' '.join(str(model).replace("\n", "").split())
            analysis = f'''
########################################

PIPELINE ANALYSIS - DITAT_ML - ditat.io

Description:

- Model: {model_str}
- KFold: n = {k_folds}
- Data shape: {self.df.shape}
- Test Size %: {test_size}

Indicators:

    - Class False % : {self.y[self.y == 0].shape[0] / self.df.shape[0] :0.2f}
    - Class True  % : {self.y[self.y == 1].shape[0] / self.df.shape[0] :0.2f}

    - Train Score   : {self.avg_train_score.round(4)}
    - Test Score    : {self.avg_test_score.round(4)}

    - Auc Train     : {self.avg_ras_train.round(4)}
    - Auc  Test     : {self.avg_ras_test.round(4)}

    - Avg. Confusion Matrix - Training
              {' ' * (spaces - 2)}PN | PP
           TN {' ' * (spaces - len(str(self.avg_train_cm[0][0]).split('.')[0]))}{self.avg_train_cm[0][0] :0.0f} | {self.avg_train_cm[0][1] :0.0f}
           TP {' ' * (spaces - len(str(self.avg_train_cm[1][0]).split('.')[0]))}{self.avg_train_cm[1][0] :0.0f} | {self.avg_train_cm[1][1] :0.0f}

    - Avg. Confusion Matrix - Testing
              {' ' * (spaces - 2)}PN | PP
           TN {' ' * (spaces - len(str(self.avg_test_cm[0][0]).split('.')[0]))}{self.avg_test_cm[0][0] :0.0f} | {self.avg_test_cm[0][1] :0.0f}
           TP {' ' * (spaces - len(str(self.avg_test_cm[1][0]).split('.')[0]))}{self.avg_test_cm[1][0] :0.0f} | {self.avg_test_cm[1][1] :0.0f}

    - Feature Importance (Displaying max. 60 rows)
{self.avg_fi.head(60)}

    - Feature Correlation (th >= {corr_th}) (Displaying max. 60 rows)
{self.avg_df_corr.head(60)}

########################################
'''
            print(analysis)

    def _results(
        self,
        show_plots=True,
        save=False,
        save_path=None,
        corr_th=0.8,
        verbose=True,
        scoring='roc_auc',
        cv=5,
        other_cols=None
        ):
        '''
        '''
        # Mapping variables according to self._deployment
        if self._deployment:
            verbose = False
            corr_th = 0
            save = True
            save_path = self.model_path
        
        # Create 3 dataframes for results.
        train_results = pd.DataFrame()
        test_results = pd.DataFrame()
        full_results = pd.DataFrame()
        
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
            train_results['train_predict'] = self.model.predict(self.X_train_scaled)
            train_results['y_train'] = self.y_train
            
            # Test dataframes
            test_results['test_predict'] = self.model.predict(self.X_test_scaled)
            test_results['y_test'] = self.y_test
            
            # Train and test accuracy scores.
            self.train_score = self.model.score(self.X_train_scaled, self.y_train)
            self.test_score =self.model.score(self.X_test_scaled, self.y_test)
            
            # Verbose scores
            if verbose:
                print('Score Train', round(self.train_score, 4))
                print('Score Test', round(self.test_score, 4))
            
            # Confusion matrices
            self.train_cm = confusion_matrix(self.y_train, train_results['train_predict'])
            self.test_cm = confusion_matrix(self.y_test, test_results['test_predict'])
            
            # Verbose confusion matrices
            if verbose:
                print('Train\n', self.train_cm)
                print('Test\n', self.test_cm)
            
            # If predict_proba is a method of self.model
            if 'predict_proba' in dir(self.model):
                # Adding predict_proba to both dataframes
                train_results['train_predict_proba'] = self.model.predict_proba(self.X_train_scaled)[:, 1]
                test_results['test_predict_proba'] = self.model.predict_proba(self.X_test_scaled)[:, 1]
                
                # Creating Roc Auc Scores as attributes for both sets.
                self.ras_train = roc_auc_score(self.y_train, train_results['train_predict_proba'])
                self.ras_test = roc_auc_score(self.y_test, test_results['test_predict_proba'])
                
                # Verbose AUC
                if verbose:
                    print(f"Roc Auc score Training: {round(self.ras_train, 4)}")
                    print(f"Roc Auc score Testing: {round(self.ras_test, 4)}")
        else:
            # Similar to the previous if, but for self._deployment == True
            full_results['y'] = self.y
            full_results['predict'] = self.model.predict(self.X_scaled)
            
            # Accuracy score as attribute.
            self.full_score = self.model.score(self.X_scaled, self.y)
            if verbose:
                print('Score Full', round(self.full_score, 4))
            
            # If predict_proba is a method of 
            if 'predict_proba' in dir(self.model):
                full_results['predict_proba'] = self.model.predict_proba(self.X_scaled)[:, 1]
                self.ras_full = roc_auc_score(self.y, full_results['predict_proba'])
                if verbose:
                    print(f"Roc Auc score Full: {round(self.ras_full, 4)}")
            
            # Add extra columns if needed
            if other_cols and all(item in self.df.columns for item in other_cols):
                for col in other_cols:
                    full_results[col] = self.df[col]
            
            # Saving results
            full_results.to_csv(os.path.join(self.model_path, 'full_results.csv'), index=False)

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
                the learning curves.
            - add_date_to_save_path (bool, default=False): Add date to model name for self.path

        '''
        # Set environment
        self._deployment = True

        # Set model timestamp
        self._information['date'] = str(date.today())
        
        # Save attributes.
        self.model_name = name
        self._information['model_name'] = self.model_name
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
        self._results(other_cols=other_cols, show_plots=save_plots)
        
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
        path=None,
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
            - path (str, default=None): Where the data for the predicting is.
                includes the _YYYYMMDD.
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
            if path is None:
                raise ValueError('You have to provide a path or self.load_data() to predict.')
            self.load_data(path)
        
        # Loading model
        self.load_model(model_name)
        
        # Setting X_columns. It will look in self._information['X_columns']
        self.X_columns = None
        
        # Apply custom transformation to data if present in self._information['apply_X']
        for info in self._information['apply_X']:           
            custom_module = importlib.import_module(f'models.{self.model_name}.custom')
            custom_function = getattr(custom_module, info['function_name'])
            self.X = custom_function(self.X, info['columns'])   
        
        # Preprocess self.X
        self.preprocessing()
        
        # Scale self.X using the loaded scaler.                        
        self.scale(scaler=self.scaler)
        
        # Dataframe with predictions (and predict_proba if available).
        results = pd.DataFrame()
        results['predict'] = self.model.predict(self.X_scaled)
        if 'predict_proba' in dir(self.model):
            results['predict_proba'] = self.model.predict_proba(self.X_scaled)[:, 1]
        
        # Add column(s) to predictions if needed. Used for indexing mostly.
        if other_cols and all(item in self.df.columns for item in other_cols):
            for col in other_cols:
                results[col] = self.df[col]
        
        # Choose to save results if needed.
        if save is True:
            results.to_csv(os.path.join(self.model_path, 'results.csv'), index=False)
        return results

    def load_model(self, model_name: str):
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
        self.model_path = os.path.join(os.getcwd(), 'models', self.model_name)
        
        # Load model and scalers
        self.model = load(os.path.join(self.model_path, 'model.joblib'))
        self.scaler = load(os.path.join(self.model_path, 'scaler.joblib'))
        
        # Load self._information
        with open(os.path.join(self.model_path, 'information.json'), 'r') as file:
            self._information = json.load(file)



        
