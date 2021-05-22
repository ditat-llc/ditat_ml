import os
from joblib import dump, load
import inspect
import importlib
from datetime import date
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, confusion_matrix

from . import utility_functions


'''
High level class for a pipeline using the low-level tool in utility_functions.py

3 majors Areas:

I. Train and test your model.
    1. Load Data | Pipeline.load_data()
    2. Load feature(s) and target(s) | Pipeline.load_X_y
        Columns to be used.
    2.5 Apply custom transformation to column(s) in self.X
    3. Split into Train and Test | Pipeline.split()
    4. Preprocessing | Pipeline.preprocessing()
        This step is by far the most complex. You need to load the different
        mappings for different features.
    5. Scale data. | Pipeline.scale()
    6. Train your model | Pipeline.train()

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
        self.load_X_y()
        self.preprocessing()
        self.scale()

        and you can make predictions with sef.model.predict()
'''



class Pipeline:
    # Set to None to try different values (cross validation)
    RANDOM_STATE = 4

    def __init__(self, nrows=None):
        '''
        - self._model
            Internal attribute for the self.model property.

        - self._deployment (bool, default=False)
            Flag to have the class instance as a trainer/deployer or a predictor.

        - self._information (dict)
            Dictionary containing all the extra information about the deployed model.
            It is pickled and saved for predictions.
        '''
        self.nrows = nrows

        self._model = None
        self._deployment = False

        self._information = {'apply_X': []}
        
        # Other flags
        self._preprocessed = False
        self._split = False
        self._scale = False


    def load_data(self, path):
        '''
        Loads the data from csv file to pd.DataFrame
        and the columns are "cleaned".

        Note:
            This method is used for both Deployment and Prediction.

        Args:
            - path (str): Filepath for csv file with data.

        Returns:
            - None
        '''
        df = pd.read_csv(filepath_or_buffer=path, nrows=self.nrows)
        self.df = df

    def load_X_y(
        self,
        X_columns: list=None,
        y_columns: list or str=None
        ):
        '''
        Load self.X and self.y from self.df with checks in place.

        Also loads self.X_columns and self.y_columns (it could be a string) 

        Args:
            - X_columns (list)
            - y (list or column)

        Returns:
            - None
        '''
        if self._deployment is True:
            X_columns = self._information['X_columns']

        # self.X_columns = utility_functions.clean_columns(X_columns)
        self.X_columns = X_columns
        self._information['X_columns'] = self.X_columns
        X_checker = all(elem in self.df.columns  for elem in self.X_columns)
        
        if X_checker is False:
            raise ValueError('One or more values in X_columns do not exist in self.df')

        self.X = self.df[self.X_columns]

        if self._deployment is False:
            # self.y_columns = utility_functions.clean_columns(y_columns)
            self.y_columns = y_columns
            self._information['y_columns'] = self.y_columns

            y_ = self.y_columns if isinstance(self.y_columns, list) else [self.y_columns]
            y_checker = all(elem in self.df.columns for elem in y_)

            if y_checker is False:
                raise ValueError('y not contained in self.df')

            self.y = self.df[self.y_columns]

    def split(self, test_size: float=0.05, stratify=False):
        '''
        Simple splitter for train and test data.

        Args:
            - test_size (float, default=0.05)
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
            random_state=type(self).RANDOM_STATE,
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
                if feature not in self.X_columns:
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
                if feature not in self._information['X_columns']:
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

    def apply_X(self, function, columns):
        '''
        Universal transformation of the whole dataset before splitting.
        There should not be any imputers saved.

        It should be implemented after self.load_X_y()

        Args:
            - func (function): Function applied.
            - columns (list): List of columns to which the transformation applies.

        '''
        if self._preprocessed is True:
            raise ValueError('Apply X has to be used before preprocessing!')

        self.X = function(self.X, columns)
        self._information['apply_X'].append({
            'columns': columns,
            'function_name': function.__name__,
            'function': function
        })

    def scale(self, scaler=MinMaxScaler()):
        '''
        Depending on self_deployment, we scaled self.X_train and self.X_test
        OR self.X (predictions).

        Args:
            - scaler (default="sklearn.preprocessing.MinMaxScaler")

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
                value.random_state = type(self).RANDOM_STATE

            # Recheck this part.
            classes = np.unique(self.y_train)
            class_weight_array = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=self.y_train
            )
            self.class_weight_ = dict(zip(classes, class_weight_array))
            
            if 'class_weight' in dir(value):
                value.class_weight = self.class_weight_

        self._model = value

        if 'get_params' in dir(self._model):
            self._information['model_params'] = self._model.get_params()

    def train(self, show_plots=True, corr_th=0.8, scoring='roc_auc', verbose=True):
        '''
        Args:
            - show_plots
        '''
        if self._model is None:
            raise ValueError('You have not loaded a model.')

        self.model.fit(self.X_train_scaled, self.y_train)

        self._results(show_plots, corr_th=corr_th, scoring=scoring, verbose=verbose)

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
        if self._deployment:
            verbose = False
            corr_th = 0
            save = True
            save_path = self.model_path

        train_results = pd.DataFrame()
        test_results = pd.DataFrame()
        full_results = pd.DataFrame()

        X_ = self.X_scaled if self._deployment else self.X_train_scaled
        y_  = self.y if self._deployment else self.y_train

        if 'feature_importances_' in dir(self.model):
            df_fi = utility_functions.feature_importance(
                feature_importances=self.model.feature_importances_,
                data=X_,
                verbose=verbose
            )
            if save is True:
                df_fi.to_csv(os.path.join(self.model_path, 'feature_importances.csv'))

        if not self._deployment:
            train_results['train_predict'] = self.model.predict(self.X_train_scaled)
            train_results['y_train'] = self.y_train
            
            test_results['test_predict'] = self.model.predict(self.X_test_scaled)
            test_results['y_test'] = self.y_test

            self.train_score = self.model.score(self.X_train_scaled, self.y_train)
            self.test_score =self.model.score(self.X_test_scaled, self.y_test)
            
            print('Score Train', round(self.train_score, 4))
            print('Score Test', round(self.test_score, 4))

            self.train_cm = confusion_matrix(self.y_train, train_results['train_predict'])
            self.test_cm = confusion_matrix(self.y_test, test_results['test_predict'])
            print('Train\n', self.train_cm)
            print('Test\n', self.test_cm)
        
            if 'predict_proba' in dir(self.model):
                train_results['train_predict_proba'] = self.model.predict_proba(self.X_train_scaled)[:, 1]
                test_results['test_predict_proba'] = self.model.predict_proba(self.X_test_scaled)[:, 1]

                self.ras_train = roc_auc_score(self.y_train, train_results['train_predict_proba'])
                self.ras_test = roc_auc_score(self.y_test, test_results['test_predict_proba'])

                print(f"Roc Auc score Training: {round(self.ras_train, 4)}")
                print(f"Roc Auc score Testing: {round(self.ras_test, 4)}")

        else:
            full_results['y'] = self.y
            full_results['predict'] = self.model.predict(self.X_scaled)

            self.full_score = self.model.score(self.X_scaled, self.y)
            print('Score Full', round(self.full_score, 4))

            if 'predict_proba' in dir(self.model):
                full_results['predict_proba'] = self.model.predict_proba(self.X_scaled)[:, 1]
                self.ras_full = roc_auc_score(self.y, full_results['predict_proba'])
                print(f"Roc Auc score Full: {round(self.ras_full, 4)}")

            if other_cols and all(item in self.df.columns for item in other_cols):
                for col in other_cols:
                    full_results[col] = self.df[col]

            full_results.to_csv(os.path.join(self.model_path, 'full_results.csv'), index=False)

        utility_functions.find_high_corr(
            dataframe=X_,
            threshold=corr_th,
            verbose=verbose,
            save=save,
            save_path=save_path
        )

        if show_plots:
            plot1 = utility_functions.plot_learning_curve(
              estimator=self.model,
              X=X_, #.sample(frac=1, random_state=1),
              y=y_ ,#.sample(frac=1, random_state=1),
              scoring=scoring,
              estimator_name='estimator',
              cv=cv,
              save_path=save_path
            )
            if not self._deployment:
                plot1.show()

    # def kfold_train(self, n_splits=5):
    #     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)

    #     for train_index, test_index in skf.split(self.X, self.y):
    #         pass


    def deploy(self, name, directory='models', overwrite=False, other_cols=None):
        '''
        We train the model on the whole dataset once we have decided
        we are satified with the model performance. This will save the model
        and all its parameters in /directory/{name}_{date}/

        1. Create path for deployed model.
        2. Preprocess, scale and train on the full dataset.
        3. Save model in path.

        Args:
            - name (str): Name of your model. It will be save with name     # (not anymore)+ "_YYYYMMDD"
            - directory (str, default='models'): folder to use for creating the path.
            - overwrite (bool, default=False): Raise error if folder exists or overwrite.
            - other_cols (list, default=None): Other columns in the original self.df
                to include in the full_results.csv

        '''
        self._deployment = True

        self.model_name = name
        self.model_path = os.path.join(os.getcwd(), directory, self.model_name)

        if overwrite and os.path.exists(self.model_path):
            for f in os.listdir(self.model_path):
                if os.path.isdir(f):
                    os.remove(os.path.join(self.model_path, f))
        else:
            os.makedirs(self.model_path)

        with open(os.path.join(self.model_path, 'custom.py'), 'w') as file:
            for custom_function_info in self._information['apply_X']:
                f = custom_function_info['function']
                f_code = inspect.getsource(f)
                file.write(f_code)

                # Function cannot be read upon loading. Only name remains.
                del custom_function_info['function']

        self.preprocessing()
        self.scale()
        self.model.fit(self.X_scaled, self.y)

        self._results(other_cols=other_cols)

        dump(self.model, os.path.join(self.model_path, 'model.joblib'))
        dump(self.scaler, os.path.join(self.model_path, 'scaler.joblib'))

        with open(os.path.join(self.model_path, 'information.json'), 'w') as file:
            json.dump(self._information, file)

    def predict(self, path, model_name, save=True, other_cols=None):
        '''
        *** For prediction ONLY ***

        1. Load data to make predictions on. self.df
        2. Load model and information.
        3. Load self.X as a subset of self.df (self.load_X_y())
        3.5 Apply_X if there are any on self.X
        4. Preprocess data.
        5. Scale data.
        6. Predict

        Args:
            - path (str): Where the data for the predicting is.
            - model_name (str): Name of the model to load. It 
                includes the _YYYYMMDD.
            - save (bool, default=True): Whether to save the prediction dataframe as
                results.csv
            - other_cols (list, default=None): Include other columns from the dataframe to
                be present in the predictions

        '''
        self._deployment = True

        self.load_data(path)
        self.load_model(model_name)
        self.load_X_y()

        for info in self._information['apply_X']:           
            custom_module = importlib.import_module(f'models.{self.model_name}.custom')
            custom_function = getattr(custom_module, info['function_name'])
            self.X = custom_function(self.X, info['columns'])   

        self.preprocessing()                        
        self.scale(scaler=self.scaler)

        results = pd.DataFrame()

        results['predict'] = self.model.predict(self.X_scaled)

        if 'predict_proba' in dir(self.model):
            results['predict_proba'] = self.model.predict_proba(self.X_scaled)[:, 1]

        if other_cols and all(item in self.df.columns for item in other_cols):
            for col in other_cols:
                results[col] = self.df[col]

        if save is True:
            results.to_csv(os.path.join(self.model_path, 'results.csv'), index=False)

        return results

    def load_model(self, model_name):
        '''
        *** For prediction ONLY ***

        Loading of all the information of the model
        previously deployed and saved.

        '''
        if self._deployment is not True:
            raise AssertionError('self._deployment must be True.')

        self.model_name = model_name
        self.model_path = os.path.join(os.getcwd(), 'models', self.model_name)

        self.model = load(os.path.join(self.model_path, 'model.joblib'))
        self.scaler = load(os.path.join(self.model_path, 'scaler.joblib'))

        with open(os.path.join(self.model_path, 'information.json'), 'r') as file:
            self._information = json.load(file)



        
