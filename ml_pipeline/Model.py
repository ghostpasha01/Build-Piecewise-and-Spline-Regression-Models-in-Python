from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer,  ColumnTransformer
from sklearn.metrics import *
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler, PolynomialFeatures
import numpy as np
from ml_pipeline.kuma_utils.preprocessing.imputer import LGBMImputer
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from ml_pipeline.piecewise.piecewise.regressor import piecewise
from patsy import dmatrix
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedKFold,cross_val_score
from pyearth import Earth



def linear_regression(X_train, y_train, X_test, y_test):
    '''
    The function creates a linear regression model
    -------
    Parameters:
    X_train: training input variables
    y_train: training target variable
    X_test: testing input variables
    y_test: testing target variable

    Returns:
    result: Linear Regression Model Object
    res: predictions
    '''
    try:
        X_const=sm.add_constant(X_train)

        result=sm.OLS(y_train,X_const).fit()

        res=result.predict(sm.add_constant(X_test[['WL', 'Yoga', 'Laps', 'WI', 'PAFS', 'Team_Clippers', 'Team_Lakers',
        'Team_Porcupines', 'Team_Trailblazers', 'Team_Warriors']]))
        print("Linear Regression Results:")
        print("MAE of the model is",mean_absolute_error(y_test,res))
        print("MSE of the model is",mean_squared_error(y_test,res))
        print("RMSE of the model is",np.sqrt(mean_squared_error(y_test,res)))
        print("R2 score for test is",r2_score(y_test,res))

    except Exception as e:
        print(e)

    else:
        return result, res



def polynomial_regression(X_train, y_train, X_test, y_test, deg):
    '''
    The function creates a polynomial regression model
    -------
    Parameters:
    X_train: training input variables
    y_train: training target variable
    X_test: testing input variables
    y_test: testing target variable
    deg: degree of the polynomial

    Returns:
    lm: Model Object
    predictions: predictions
    '''

    try:
        poly=PolynomialFeatures(degree=deg)
        X_poly=poly.fit_transform(X_train)

        poly.fit(X_poly,y_train)
        lm=linear_model.LinearRegression()

        lm.fit(X_poly,y_train)
        #Doing predictions on test data
        predictions=lm.predict(poly.fit_transform(X_test))
        print("Polynomial Regression Results:")
        print("MAE of the model is",mean_absolute_error(y_test,predictions))
        print("MSE of the model is",mean_squared_error(y_test,predictions))
        print("RMSE of the model is",np.sqrt(mean_squared_error(y_test,predictions)))
        print("R2 score for test is",r2_score(y_test,predictions))

    except Exception as e:
        print(e)

    else:
        return lm, predictions



def piecewise_regressor(X_train, y_train, X_test, y_test):
    '''
    The function creates a piecewise regression model on WL feature
    -------
    Parameters:
    X_train: training input variables
    y_train: training target variable
    X_test: testing input variables
    y_test: testing target variable

    Returns:
    model_piecewise: Piecewise Regression Model Object
    predictions: predictions on test data
    '''
    try:
        model_piecewise=piecewise(X_train.WL.ravel(),y_train.ravel())
        predictions=model_piecewise.predict(X_test.WL.ravel())
        print("Piecewise Regression Results:")
        print("MAE of the model is",mean_absolute_error(y_test,predictions))
        print("MSE of the model is",mean_squared_error(y_test,predictions))
        print("RMSE of the model is",np.sqrt(mean_squared_error(y_test,predictions)))
        print("R2 score for test is",r2_score(y_test,predictions))

    except Exception as e:
        print(e)

    else:
        return model_piecewise, predictions



def spline_regressor(X_train, y_train, X_test, y_test):
    '''
    The function creates a spline regression model on WL feature
    -------
    Parameters:
    X_train: training input variables
    y_train: training target variable
    X_test: testing input variables
    y_test: testing target variable

    Returns:
    spline_fit: Spline Regression Model Object
    predictions: predictions on test data
    '''
    try:
        X_spline=dmatrix('bs(x,df=5,degree=3,include_intercept=False)',{'x':X_train.WL},return_type='dataframe')
        spline_fit=sm.GLM(y_train,X_spline).fit()
        predictions=spline_fit.predict(dmatrix('bs(x,df=5,degree=3,include_intercept=False)',{'x':X_test.WL},return_type='dataframe'))

        print("Spline Regression Results:")
        print("MAE of the model is",mean_absolute_error(y_test,predictions))
        print("MSE of the model is",mean_squared_error(y_test,predictions))
        print("RMSE of the model is",np.sqrt(mean_squared_error(y_test,predictions)))
        print("R2 score for test is",r2_score(y_test,predictions))

    except Exception as e:
        print(e)

    else:
        return spline_fit, predictions



def spline_smoothing(X_train, y_train, X_test, y_test):
    '''
    Spline surface smoothing with WL and Yoga
    -------
    Parameters:
    X_train: training input variables
    y_train: training target variable
    X_test: testing input variables
    y_test: testing target variable

    Returns:
    spline_model: Spline Regression Model Object
    predictions: predictions on test data

    '''
    try:
        spline_model=make_pipeline(SplineTransformer(n_knots=5,degree=3),linear_model.LinearRegression())
        spline_model.fit(X_train[['WL','Yoga']],y_train)
        predictions=spline_model.predict(X_test[['WL','Yoga']])
        print("Multiple Spline Regression Results:")
        print("MAE of the model is",mean_absolute_error(y_test,predictions))
        print("MSE of the model is",mean_squared_error(y_test,predictions))
        print("RMSE of the model is",np.sqrt(mean_squared_error(y_test,predictions)))
        print("R2 score for test is",r2_score(y_test,predictions))

    except Exception as e:
        print(e)

    else:
        return spline_model, predictions



def mars_model(X_train, y_train, X_test, y_test):
    '''
    MARS regression
    -------
    Parameters:
    X_train: training input variables
    y_train: training target variable
    X_test: testing input variables
    y_test: testing target variable

    Returns:
    model: MARS Model Object
    predictions: predictions on test data

    '''
    try:
        model=Earth()
        cv=RepeatedKFold(n_splits=5,n_repeats=3,random_state=12)
        scores=cross_val_score(model, X_train,y_train,scoring='neg_root_mean_squared_error',cv=cv,n_jobs=-1)
        model=Earth(max_degree=8,allow_linear ='True',endspan=20)
        model.fit(X_train,y_train)

        predictions=model.predict(X_test)

        print("MARS Results:")
        print("MAE of the model is",mean_absolute_error(y_test,predictions))
        print("MSE of the model is",mean_squared_error(y_test,predictions))
        print("RMSE of the model is",np.sqrt(mean_squared_error(y_test,predictions)))
        print("R2 score for test is",r2_score(y_test,predictions))

    except Exception as e:
        print(e)

    else:
        return model, predictions

