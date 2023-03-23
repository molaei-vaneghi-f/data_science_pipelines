"""build and train a regression model on the Capital Bike Share (Washington, D.C.) Kaggle data set, 
in order to predict demand for bicycle rentals at any given hour, based on time and weather"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
# pip install statsmodels
from statsmodels.api import OLS, add_constant
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
sns.set()
sns.set_style('white')

#%%
class feature_eng:
    
    def __init__(self):
        
        self.trans = ColumnTransformer([
            ('encode', OneHotEncoder(drop = 'first', handle_unknown='ignore'), ['month','weekday','hour']), #,'weather'
            ('encode_binary', OneHotEncoder(drop='if_binary'), ['holiday']),
            ('st_scale', RobustScaler(), ['atemp','windspeed','humidity'])    # use robustscaler when having outliars
        ])

        self.poly = PolynomialFeatures(include_bias=False, interaction_only=False, degree=2)
        self.inter = PolynomialFeatures(include_bias = False, interaction_only = True)
        
    def col_trans (self, X):
        # apply column transformer
        Xc = X.copy(deep = True) 
        Xc = self.trans.fit_transform(Xc)
        Xc = pd.DataFrame(Xc.toarray(), columns=self.trans.get_feature_names_out().tolist())
        return Xc

    # in the context of feature engineeridng we use fit_transform (poly, inter, etc.) not just fit (lr)
    def fit_trans (self, Xtrain, model):
        Xtrain_fe = self.col_trans(Xtrain)
        Xtrain_fe = model.fit_transform (Xtrain_fe) 
        Xtrain_fe = pd.DataFrame(Xtrain_fe, columns = model.get_feature_names_out())
        return Xtrain_fe
    
    # trans on Xtrain and Xtest
    def trans (self, X, model):
        X_fe = self.col_trans(X)
        X_fe = model.transform(X_fe)
        X_fe = pd.DataFrame(X_fe, columns = model.get_feature_names_out())
        return X_fe


def reg_funcs (model_name, Xtrain, Xtest, ytrain, ytest):
    
    # incorporate pretty tables: mytable = pretty_table(df), mytable.add_row()
    # model_name is case-sensitive
    m = model_name(alpha=1, random_state=31)
    m.fit(Xtrain, ytrain)
    ypred_Xtest = m.predict(Xtest)
    
    # scores
    # in case of lasso, ridge, and elastic net score returns r2 (coefficient of determination)
    Xtrain_score = m.score(Xtrain, ytrain)
    Xtest_score = m.score(Xtest, ytest)
    Xtest_rmse = np.sqrt(mean_squared_error(ytest, ypred_Xtest))
    print(f'''{str(model_name)}: 
          train score (r2 for reg): {round(Xtrain_score, 3)}, 
          test score (r2 for reg): {round(Xtest_score, 3)}, 
          rmse (on test data): {round(Xtest_rmse, 3)}''')
          
#%% DATA LOADING

db = './data'
bike = pd.read_csv(os.path.join(db, 'bike_train.csv'), parse_dates=True) #index_col=0)
# bike.head()

#%% PRE-PROCESSING

# converting datetime column
date = pd.to_datetime(bike['datetime'])
bike ['year'] = date.dt.year
bike ['month'] = date.dt.month
bike['weekday'] = date.dt.weekday
bike ['day'] = date.dt.day
bike ['hour'] = date.dt.hour
bike ['time'] = date.dt.time 

bike.drop(['casual', 'registered', 'datetime', 'time', 'temp', 'weather'], axis = 1, inplace = True)

#%% EDA 

# scatter-plots
g = sns.FacetGrid(bike, hue="year", col="season", margin_titles=True,
                palette="pastel",hue_kws=dict(marker=["^", "v"]))
g.map(plt.scatter, "hour", "count", edgecolor="w").add_legend()
plt.subplots_adjust(top=0.8)
g.fig.suptitle('count by year, season, and hour');


g = sns.FacetGrid(bike, hue="holiday", col="season", margin_titles=True,
                palette="pastel",hue_kws=dict(marker=["^", "v"]))
g.map(plt.scatter, "day", "count", edgecolor="w").add_legend()
plt.subplots_adjust(top=0.8)
g.fig.suptitle('count by holiday, season, and day');

# violin-plots
fig, ax = plt.subplots(figsize=(15, 6))
sns.violinplot(data=bike, x='season', y='count', hue='year', palette='Set2', ax=ax)
ax.set(xlabel='', ylabel='')  
sns.despine()
plt.show()

# bar-plots
sns.barplot(x='hour', y='count', data=bike)

#%% TRAIN AND TEST SPLIT

X = bike.drop(['count'], axis = 1, inplace = False)
y = bike['count']

Xtrain, Xtest, ytrain, ytest = train_test_split(X,
                                                y, 
                                                test_size = 0.20, 
                                                random_state = 77 # random_state is seed value for the random number generator
)

print(f'Training shapes: \n\tXtrain: {Xtrain.shape}\n\tytrain: {ytrain.shape}')
print(f'Test shapes: \n\tXtest: {Xtest.shape}\n\tytest: {ytest.shape}')

#%% TRYING DIFFERENT FEATURE ENGINEERING METHODS

# 1: Remove highly correlated variables:
corr = bike.corr()
plt.figure(figsize=(9, 9))
sns.heatmap(corr, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');

# 2.1: Select significant features using pvalues (drop the features with high pvalue)
plt.rcParams['figure.figsize'] = (12,6)
m = OLS(ytrain, add_constant(Xtrain)) 
m_result = m.fit()
print(m_result.summary())

# 2.2: Select significant features using random forest feature importance: drop features with the smallest importance
random_forest = RandomForestRegressor(max_depth=7, random_state=0)
random_forest.fit(Xtrain, ytrain)
feature_importance_dict = {'importance': random_forest.feature_importances_, 'feature': Xtrain.columns}
pd.DataFrame(data=feature_importance_dict).sort_values('importance', ascending=False)

# 2.3: Select significant features recursive feature elimination  
rfe = RFE(estimator=LinearRegression(), verbose=1, n_features_to_select=4)  
rfe.fit(Xtrain, ytrain)

# feature_names_in_ are all features
rfe.feature_names_in_
rfe.support_
rfe.ranking_
# display selected/important features with their names:
rfe.feature_names_in_[rfe.support_] 
# display dropped/unimportant features with their names:
# rfe.feature_names_in_[~rfe.support_] 
print(f'drop the following features: {rfe.feature_names_in_[~rfe.support_]}')

# 3: Engineering New Fatures in case of Simpson's Paradox

# 4: building feature vector| interaction and polynomial feature expansion
feature_eng = feature_eng()
poly = PolynomialFeatures(include_bias=False, interaction_only=False, degree=2)   

# poly on training data
Xtrain_fe = feature_eng.fit_trans(Xtrain, poly)

# poly on test data
Xtest_fe = poly.transform(feature_eng.col_trans(Xtest))
Xtest_fe  = pd.DataFrame(Xtest_fe, columns = poly.get_feature_names_out())

#%% LR

lr = LinearRegression().fit(Xtrain_fe, ytrain)
ypred_Xtest_fe = lr.predict(Xtest_fe)

print(f'''linear regression:
      lr train score: {round(lr.score(Xtrain_fe, ytrain), 3)}
      lr test score: {round(lr.score(Xtest_fe, ytest), 3)}
      lr rmse (on test data): {round(np.sqrt(mean_squared_error(ytest, ypred_Xtest_fe)), 2)}
      lr r2 (on test data): {round(r2_score(ytest, ypred_Xtest_fe), 2)}''') # 1 is perfect predic  
      
#%% REGULARIZATION
  
reg_funcs(Ridge, Xtrain_fe, Xtest_fe, ytrain, ytest)
reg_funcs(Lasso, Xtrain_fe, Xtest_fe, ytrain, ytest)
reg_funcs(ElasticNet, Xtrain_fe, Xtest_fe, ytrain, ytest)

    
    
    
    
    
    
    
    