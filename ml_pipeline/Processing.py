from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from category_encoders import OneHotEncoder
from ml_pipeline.kuma_utils.preprocessing.imputer import LGBMImputer
import pandas as pd


class Preprocessing:
    def __init__(self,data):
        self.data=data

    #function to remove outliers
    def remove_outlier(self,target):
        cols=self.data.drop(target,axis=1).select_dtypes(exclude=['object']).columns.tolist()
        Q1 = self.data[cols].quantile(0.25)
        Q3 = self.data[cols].quantile(0.75)
        IQR = Q3 - Q1
        self.data =self.data[~((self.data[cols] < (Q1 - 1.5 * IQR)) | (self.data[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
        print('Outliers removed..')
        return self.data

    #function for lgbm imputation
    def lgbm_imputer(self,x_train,x_test):
        imputer=LGBMImputer(n_iter=15,verbose=True)
        x_train=imputer.fit_transform(x_train)
        x_test=imputer.transform(x_test)
        x_train=pd.DataFrame(x_train,columns=x_train.columns)
        x_test=pd.DataFrame(x_test,columns=x_test.columns)
        print('Missing values imputed..')
        return x_train,x_test

    # splitting data.
    def split_data(self, target_col):
        X=self.data.drop(target_col,axis=1)
        Y=self.data[target_col]
        # split a dataset into train and test sets
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size=0.2)
        return X_train, X_test,y_train,y_test


def rename_columns(data):
    '''
    rename columns
    '''
    #renaming required columns
    df=data.rename(columns={'Points_Scored':'Points','Weightlifting_Sessions_Average':'WL','Yoga_Sessions_Average':'Yoga',
                        'Laps_Run_Per_Practice_Average':'Laps','Water_Intake':'WI',
                        'Players_Absent_For_Sessions':'PAFS'})
    
    return df



def one_hot_encode(data):
    '''
    one hot encode team variable
    '''
    #one hot encoding
    one_hot_df=pd.get_dummies(data,columns=['Team'],drop_first=True)
    return one_hot_df