#For building, tuning, and evaluating regression models
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from math import sqrt
from sklearn.linear_model import  Ridge

def r_squared(y_true, y_predict):
    '''
    This function returns the R squared scored based on the true and predicted values
    :param y_true: true values in numpy array fromat
    :param y_predict: predicted values in numpy array format
    :return: r squared score or coefficient of determination
    '''

    cd = r2_score(y_true, y_predict)
    return cd

def rss (y_true, y_predict):
    '''
    This function calculates sum of residual squared errors
    :param y_true: true values in numpy array fromat
    :param y_predict: predicted values in numpy array format
    :return: sum of residual squared errors
    '''
    resids = y_true - y_predict
    resids_squared = resids * resids
    RSS = resids_squared.sum()
    return RSS


def fit_model(X, y,metric, model):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y].
        metric: type of metric, r^2 or rss
        model: type of model
    """
    cv_sets = ShuffleSplit(n_splits=10, test_size= 0.2, train_size= 0.8, random_state=42)
    

    if model == 'regression_tree':

        clf = DecisionTreeRegressor(random_state=42)

        # Creating a dictionary for the parameter 'max_depth' with a range from 1 to 10
        param = {
                    'max_depth': [1,2,3,4,5,6,7,8,9,10]
        }


    elif model == 'ridge':
        clf = Ridge(random_state=42, fit_intercept=False)
        param = {
            'alpha': [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }


    if metric == 'r2':
            scoring_fnc = make_scorer(r_squared,greater_is_better=True)

    elif metric == 'rss':
            scoring_fnc = make_scorer(rss, greater_is_better=False)

    # Creating the grid search cv object --> GridSearchCV()
    grid = GridSearchCV(estimator=clf, param_grid=param, cv=cv_sets,scoring= scoring_fnc)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

