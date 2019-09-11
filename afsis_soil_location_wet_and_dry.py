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
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import afsis_helper_functions as af


###############################################################################
##                                                                           ##
##     Determination of GPS Coordinates in Africa Based on Soil Samples      ##
##             (Analysis Performed Using Both Wet and Dry Data)              ##
##                                                                           ##
##                       written by: Zachary N. Vealey                       ##
##                                                                           ##
###############################################################################


    
###############################################################################
##########  Load and Format Data Statistics and ML Pipeline Creation  #########
###############################################################################


# Check that the script is being run with Python 3.x 
if sys.version_info[0] < 3:
    print("This script needs to be run in a Python 3.x environment")
    sys.exit(1)
    
# Load geographical, x-ray fluorescence, and wet-chemistry data.
soil_df = af.load_xray_geo_data()


###############################################################################
##############  Brief Data Statistics and ML Pipeline Creation  ###############
###############################################################################


# Split data into a feature matrix and label vector.
X = soil_df.drop(['Latitude', 'Longitude'], axis=1)
y = soil_df[['Latitude', 'Longitude']]

# Separate data into training and test sets.
print("Creating training and test sets by an 80/20 split.", "\n")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                           random_state=42)

# Create pipeline to prepare the data which includes imputing missing values,
# adding ratios of the elements, standardization, and pca.
prep_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])


###############################################################################
##################  Select and Train a Model For the Data  ####################
###############################################################################


# Prepare training data using above pipeline.
X_train_prepared = prep_pipeline.fit_transform(X_train)

# Try out some models, and then progress based on cross-validation performance.
reg_models = [('Linear Regression', MultiOutputRegressor(
                                        LinearRegression())),
             ('Lasso', MultiOutputRegressor(
                                        Lasso())),
             ('Elastic Net', MultiOutputRegressor(
                                        ElasticNet())),
             ('k-Nearest Neighbors', MultiOutputRegressor(
                                        KNeighborsRegressor())),
             ('Decision Tree', MultiOutputRegressor(
                                        DecisionTreeRegressor())),
             ('Random Forest', MultiOutputRegressor(
                                        RandomForestRegressor(n_estimators=100,
                                                n_jobs=-1, random_state=42))),
             ('Gradient Boosting', MultiOutputRegressor(
                                        GradientBoostingRegressor(
                                                random_state=42))),
             ('Support Vector Machine', MultiOutputRegressor(
                                        SVR(kernel='rbf', gamma='scale')))]

af.ml_model_test(X_train_prepared, y_train, reg_models)
    
msg="\nRandom Forest appears to perform the best according to \n"
msg+="training with R^2.\n"
print(msg)
print("#" * 79 + "\n")
    

###############################################################################
###########  Tune Hyperparameters to Increase Model Performance  ##############
###############################################################################


# The selected model to be trained is a k-Nearest Neighbors regression, create
# a pipeline which prepares data and then fits with kNN.
prepare_and_predict_pipeline = Pipeline([
    ('prep', prep_pipeline),
    ('forest', MultiOutputRegressor(RandomForestRegressor(n_jobs=-1, 
                                                          random_state=42)))
])

# Specify parameters for random search and tune model hyperparameters.
param_distrib = {'prep__imputer__strategy': ['mean','median','most_frequent'],
                 'forest__estimator__n_estimators': randint(3, 500),
                 'forest__estimator__max_features': randint(2, X_train.shape[1]),
                 'forest__estimator__bootstrap': [True, False],
                }

print("Performing randomized search of hyperparameters on random forest.\n")
rnd_search = RandomizedSearchCV(prepare_and_predict_pipeline, 
                                param_distributions=param_distrib,
                                n_iter=50, cv=3, scoring='r2',
                                verbose=1, n_jobs=-1, random_state=42)
rnd_search.fit(X_train, y_train)

print('\nBest R^2 Value From Random Search: ', rnd_search.best_score_, '\n')


###############################################################################
#######################  Evaluate Model on Test Set  ##########################
###############################################################################


final_model = rnd_search.best_estimator_
final_predictions = final_model.predict(X_test)

# Report metrics for final model's success on test set.
print("Running analysis of final model's performance.\n")
final_r2 = r2_score(y_test, final_predictions)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("\tFinal R^2: ", final_r2)
print("\tFinal RMSE: ", final_rmse, "\n")
print("#" * 79 + "\n")

###############################################################################