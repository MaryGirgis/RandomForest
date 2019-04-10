"""
1- Import Libraries
"""
# Data Manupulation
import pandas as pd  
import numpy as np 

# Plotting graphs
import matplotlib.pyplot as plt
import seaborn as sns


# Machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.metrics import  roc_curve,auc

class RandomForestAlgorithm:
    

    """
    2- Preparing Data For Training
    2.1- split the data into training and testing sets 30 % for testing 70 % for training (parmater percentage)
    2.2 -Create a Gaussian Classifier
    2.3-Train the model using the training sets 
    
    """
    @classmethod
    def generate_x_y_params(self,df):
        """
        Param X= input columns of indicators 
        Param y =input column of y output 
        return X and y paramters 
        """
        y=df.iloc[:,-1] 
        X=df.loc[:, df.columns != y.name]
        return X,y
    @classmethod
    def train_model(self,X,y,test_size,n_estimators,random_state=0):
        """
        Param X= input columns numpy array
        Param y =Output Column numpy array
        test_size = percentage of test data set (integer)
        n_estimators : number of trees in the random forest 
        return regressor 
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0) 
        regressor = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state) 
        regressor.fit(X_train, y_train)  
        y_pred=regressor.predict(X_test)
        self.classification_metrics_results(y_test,y_pred)
        self.regression_metrics_results(y_test,y_pred)
        return regressor
    """
    3-Classification Matrics :Model Accuracy score  - classification report 
    """
    @classmethod
    def classification_metrics_results(self,y_test,y_pred):
        """
        param y_test input column numpy array
        param y_pred input column numpy array 
        """
        acuracyscore= accuracy_score(y_test, y_pred) 
        classificationreport=classification_report(y_test,y_pred)
        confusionmatrix=confusion_matrix(y_test, y_pred)
        print(acuracyscore)
        print(classificationreport)
        print(confusionmatrix)
    
    """
    #4 Regression Metrics :Mean Absolute Error  -(RMSE)Mean Squared Error -R2Square
    
    """
    @classmethod
    def regression_metrics_results(self,y_test,y_pred):
        
        MAR=mean_absolute_error(y_test, y_pred)
        RMSE=mean_squared_error(y_test, y_pred)
        R2=r2_score(y_test, y_pred)
        print(MAR)
        print(RMSE)
        print(R2)
        
    """
    #5 -Finding important Features 
    """
    @classmethod
    def important_features(self,regressor,featureslist):
        """
        param features list : list of features columns numpy array ['Open','High','Low','Close']
        Plot important features 
        return list of important features sorted decending 
        """
        feature_imp = pd.Series(regressor.feature_importances_,index=featureslist).sort_values(ascending=False)
        
        sns.barplot(x=feature_imp, y=feature_imp.index)
    
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Visualizing Important Features")
        plt.legend()
        plt.show()
        
        return feature_imp
    """
    6- Draw Roc Curve 
    """
    @classmethod
    def roc_curve_results(self,X_test,y_test,y_pred,regressor):
        """
        param X test , y test , y_pred
        """
        # Generating the ROC curve
        
        y_pred_proba_pca = regressor.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve( np.array(y_test, dtype=float),np.array(y_pred, dtype=float) )
        roc_auc = auc(fpr, tpr)
        print('y prediction probabilty '+ str(y_pred_proba_pca))
        print("AUC score is " + str(roc_auc))
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()
        
    @classmethod
    def predict_fun(self,regressor,predictval):
        """
        input regressor and one row  of indicators values 
        return results of value prediction  
        """
        return regressor.predict(predictval)

