import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


###############################################################################
##                                                                           ##
##            Helper Functions for AFSIS Soil Location Algorithms            ##
##                       written by: Zachary N. Vealey                       ##
##                                                                           ##
###############################################################################


def load_xray_geo_data():
    """A function to import the geographic and x-ray fluorescence data."""

    # Attempt to load data.
    DIR = 'afsis/2009-2013/'
    GEOREFS_FILE = DIR + 'Georeferences/georeferences.csv'
    XRAY_FLUOR = DIR + 'Dry_Chemistry/ICRAF/Bruker_TXRF/TXRF.csv'

    try:
        georefs_df = pd.read_csv(GEOREFS_FILE, index_col='SSN')
        xray_fluor_df = pd.read_csv(XRAY_FLUOR, index_col='SSN')
    except FileNotFoundError:
        print("Wrong file or file path")
        sys.exit(1)

    # Merge data into single pandas data frame.
    data_frames = [georefs_df, xray_fluor_df]
    cols_to_drop = ['Public_x', 'Cluster', 'Plot', 'Depth', 'Soil material', 
                    'Scientist', 'Site', 'Country', 'Region', 'Cultivated', 
                    'Gid', 'Public_y']

    soil_df = merge_data(data_frames, cols_to_drop)

    return soil_df

def load_wet_geo_data():
    """A function to import the geographic and wet chemistry data."""

    # Attempt to load data.
    DIR = 'afsis/2009-2013/'
    GEOREFS_FILE = DIR + 'Georeferences/georeferences.csv'
    WET_CHEM_PATH_1 = DIR + 'Wet_Chemistry/CROPNUTS/Wet_Chemistry_CROPNUTS.csv'
    WET_CHEM_PATH_2 = DIR + 'Wet_Chemistry/ICRAF/Wet_Chemistry_ICRAF.csv'

    try:
        georefs_df = pd.read_csv(GEOREFS_FILE, index_col='SSN')
        wet_chem_1_df = pd.read_csv(WET_CHEM_PATH_1, index_col='SSN')
        wet_chem_2_df = pd.read_csv(WET_CHEM_PATH_2, index_col='SSN')
    except FileNotFoundError:
        print("Wrong file or file path")
        sys.exit(1)

    # Merge data into single pandas data frame.
    data_frames = [georefs_df, wet_chem_1_df, wet_chem_2_df]
    cols_to_drop = ['Public_x', 'Cluster', 'Plot', 'Depth', 'Soil material', 
                    'Scientist', 'Site', 'Country', 'Region', 'Cultivated', 'Gid', 
                    'Public_y', 'A brooks', 'B brooks', 'Alpha brooks', 'Wcvfri',  
                    'Sucjkgi', 'Wcvfrsat', 'Wcvfrfc10', 'Wcvfrfc33', 'Wcvfrwp1500', 
                    'Wcvfrairdry', 'Awc1', 'Awc2', 'Ksat', 'Volfr', 'Lshrinkpct', 
                    'Plwcgpct', 'Llwcgpct', 'Piwcgpct']

    soil_df = merge_data(data_frames, cols_to_drop)

    return soil_df

def load_all_data():
    """A function to import geographic, x-ray, and wet chemicstry data."""
    # Attempt to load data.

    DIR = 'afsis/2009-2013/'
    GEOREFS_FILE = DIR + 'Georeferences/georeferences.csv'
    XRAY_FLUOR = DIR + 'Dry_Chemistry/ICRAF/Bruker_TXRF/TXRF.csv'
    WET_CHEM_PATH_1 = DIR + 'Wet_Chemistry/CROPNUTS/Wet_Chemistry_CROPNUTS.csv'
    WET_CHEM_PATH_2 = DIR + 'Wet_Chemistry/ICRAF/Wet_Chemistry_ICRAF.csv'

    try:
        georefs_df = pd.read_csv(GEOREFS_FILE, index_col='SSN')
        xray_fluor_df = pd.read_csv(XRAY_FLUOR, index_col='SSN')
        wet_chem_1_df = pd.read_csv(WET_CHEM_PATH_1, index_col='SSN')
        wet_chem_2_df = pd.read_csv(WET_CHEM_PATH_2, index_col='SSN')
    except FileNotFoundError:
        print("Wrong file or file path")
        sys.exit(1)

    # Merge data into single pandas data frame.
    data_frames = [georefs_df, xray_fluor_df, wet_chem_1_df, wet_chem_2_df]
    cols_to_drop = ['Public_x', 'Cluster', 'Plot', 'Depth', 'Soil material', 
                'Scientist', 'Site', 'Country', 'Region', 'Cultivated', 'Gid', 
                'Public_y', 'A brooks', 'B brooks', 'Alpha brooks', 'Wcvfri',  
                'Sucjkgi', 'Wcvfrsat', 'Wcvfrfc10', 'Wcvfrfc33', 'Wcvfrwp1500', 
                'Wcvfrairdry', 'Awc1', 'Awc2', 'Ksat', 'Volfr', 'Lshrinkpct', 
                'Plwcgpct', 'Llwcgpct', 'Piwcgpct', 'Public']

    soil_df = merge_data(data_frames, cols_to_drop)

    return soil_df

def merge_data(data_frames, cols_to_drop=None):
    """
    Merge the data from the different data frames and drop the 
    unneccessary columns.
    """

    soil_df = reduce(lambda left, right: pd.merge(left, right, on=['SSN']), 
                                                                    data_frames)

    if cols_to_drop:
        soil_df = soil_df.drop(cols_to_drop, axis = 1)

    return soil_df

def ml_model_test(X_train_prepared, y_train, reg_models):
    """A function to test some machine learning algorithms for performance."""

    msg="Performing initial test of some machine-learning regressors on training"
    msg+="\ndata using 5-fold cross validation. Scoring is being assessed with an"
    msg+="\nR^2 metric."
    print(msg, "\n")

    models, scores = [], []

    # Test and report scores from cross-validation for each
    for model, regressor in reg_models:
        cv_scores = cross_val_score(regressor, X_train_prepared, y_train, cv=5,
                                                    scoring='r2')
        models.append(model)
        scores.append(cv_scores)
        print('\t%s: %f (%f)' % (model, cv_scores.mean(), cv_scores.std()))

class AddElementRatios(BaseEstimator, TransformerMixin):
    """
    A class to add features corresponding to the relative ratio between all
    elements.
    """
    def __init__(self, elem_indices, add_elems=True): 
        self.elem_indices = elem_indices
        self.add_elems = add_elems
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.add_elems:
            for elem1 in self.elem_indices:
                for elem2 in self.elem_indices:
                    ratio = X[:, elem1] / X[:, elem2]
                    X = np.c_[X, ratio]          
        return X

def indices_of_top_k(arr, k):
    """
    A function to return the indices of the top k features determined from 
    analysis of feature importances (e.g. random forest, grid search, etc.).
    """
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    """A class to include only the top k features in the data set."""
    
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
        
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, 
                                                 self.k)
        return self
    
    def transform(self, X):
        return X.iloc[:, self.feature_indices_]

def report_feat_import(feature_importances_lat, feature_importances_long,
                                                            X, grid_search_fs):
    """A function to report the results of the feature selection analysis."""

    print("\nImportance of Different Features in Prediction:", "\n")

    fi_lat_sorted = sorted(zip(feature_importances_lat, list(X)), reverse=True)
    fi_long_sorted = sorted(zip(feature_importances_long, list(X)), reverse=True)

    print("Latitude Feature Importances\tLongitude Feature Importances")
    for i in range(len(fi_lat_sorted)):
        print("\t{:.5f}  <-  {:}\t\t\t{:.5f}  <-  {:}".format(
                fi_lat_sorted[i][0], fi_lat_sorted[i][1], 
                fi_long_sorted[i][0], fi_long_sorted[i][1]))
        
    print("\nThe top {:} features of importance to latitude were kept.\n".format(
                        grid_search_fs.best_params_['feature_selection__k']))

###############################################################################