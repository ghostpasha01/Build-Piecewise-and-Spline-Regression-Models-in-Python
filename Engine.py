import pandas as pd
from ml_pipeline import Processing
from ml_pipeline.Processing import Preprocessing
from ml_pipeline.kuma_utils.preprocessing.imputer import LGBMImputer
import warnings
from ml_pipeline import Model
warnings.filterwarnings("ignore")

#reading dataset
data=pd.read_csv('input/NBA_Dataset_csv.csv')
data = Processing.rename_columns(data)

#removing outliers
target_col='Points' #target column name
df=Preprocessing(data).remove_outlier(target_col)

# one hot encoding
df = Processing.one_hot_encode(df)

#train test split of data
target_col='Points' #target column name
X_train,X_test,y_train,y_test=Preprocessing(df).split_data(target_col)


#lgbm imputer for missing value imputation
X_train,X_test=Preprocessing(df).lgbm_imputer(X_train,X_test)

print(" ")
print("Modelling Started...")
print(" ")
# Modelling starts here
# 1. Linear Regression
lr, res = Model.linear_regression(X_train, y_train, X_test, y_test)
# ml foundry log, uncomment to run
# ML_foundry.truefoundry_model_tracker("SplinesProject","SplinesRegressionProject",lr,"LinearRegression","statsmodels","Doing the first run for LR",y_test,res)

print(" ")
# 2. Polynomial Regression 
lm, predictions = Model.polynomial_regression(X_train, y_train, X_test, y_test, 4)
# ml foundry log, uncomment to run
# ML_foundry.truefoundry_model_tracker("SplinesProject","SplinesRegressionProject",lm,"PolynomialRegression","sklearn","Doing the first run for PR",y_test,predictions)

print(" ")
# 3. Piecewise Regression
model_piecewise, y_pred = Model.piecewise_regressor(X_train, y_train, X_test, y_test)
# ml foundry log, uncomment to run
# ML_foundry.truefoundry_model_tracker("SplinesProject","SplinesRegressionProject",model_piecewise,"PiecewiseRegression","sklearn","Doing the first run for Piecewise",y_test,y_pred)


print(" ")
# 4. Spline Regression
spline_fit, y_pred_test = Model.spline_regressor(X_train, y_train, X_test, y_test)
# ml foundry log, uncomment to run
# ML_foundry.truefoundry_model_tracker("SplinesProject","SplinesRegressionProject",spline_fit,"SplineRegression","sklearn","Doing the first run for Splines",y_test,y_pred_test)


print(" ")
# 5. Multiple Spline Regression
spline_model, mspline_preds = Model.spline_smoothing(X_train, y_train, X_test, y_test)
# ml foundry log, uncomment to run
# ML_foundry.truefoundry_model_tracker("SplinesProject","SplinesRegressionProject",spline_model,"MultipleSplineRegression","sklearn","Doing the first run for SplineMultiple",y_test,mspline_preds)


print(" ")
# 6. MARS
model, mars_preds = Model.mars_model(X_train, y_train, X_test, y_test)
# ml foundry log, uncomment to run
# ML_foundry.truefoundry_model_tracker("SplinesProject","SplinesRegressionProject",model,"MARS","sklearn","Doing the first run for MARS",y_test,mars_preds)
