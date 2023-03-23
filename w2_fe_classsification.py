import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder                              
from sklearn.preprocessing import KBinsDiscretizer 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler                          
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

# Models:
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve, roc_auc_score, roc_curve, auc 
from sklearn import metrics

# load the data
db = './data'
titanic_train = pd.read_csv(os.path.join(db, 'titanic_train.csv'), index_col=0)
titanic_train.head()
titanic_train.info()

#%% Feature Engineering using columnTransformer

X = titanic_train[['Age','Pclass','Sex']]  
y = titanic_train['Survived']
X.shape, y.shape

# split
Xtrain, Xtest, ytrain, ytest = train_test_split(X,
                                                y, 
                                                test_size = 0.20, 
                                                random_state = 77 # random_state is seed value for the random number generator
)

print(f'Training shapes: \n\tXtrain: {Xtrain.shape}\n\tytrain: {ytrain.shape}')
print(f'Test shapes: \n\tXtest: {Xtest.shape}\n\tytest: {ytest.shape}')


#%% pipelines
    
impute_and_encode = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(drop='first', handle_unknown = "ignore") 
)


impute_and_scale = make_pipeline(
    SimpleImputer(strategy='median'), #here we first impute using the median 
    StandardScaler()
    
)

impute_and_bin = make_pipeline(
    SimpleImputer(strategy='median'), #here we first impute using the median 
    KBinsDiscretizer(n_bins=4, encode='onehot', strategy='quantile') #then we bin the data and OHE it
)

# # Pipeline with Custom Transformer Function

# # custom function
# def log_transform(x):
#     return np.log(x+1)

# log_transform_pipeline = Pipeline(steps=[
#     ('impute', SimpleImputer(strategy='median')),
#     ('logtransform',FunctionTransformer(log_transform)),
#     ('scaler',RobustScaler()) 
# ])


#%% Combine Multiple Pipelines Using ColumnTransformer: syntax: (name, transformer, column(s))

pipe_transformer = [
    ('impute_and_scale', impute_and_scale, ['Age']),
    ('encode_multi', OneHotEncoder(drop='first', handle_unknown='ignore'),['Pclass']),
    ('encode_binary', OneHotEncoder(drop='if_binary'), ['Sex']),
    # ('impute_and_encode', impute_and_encode, ['Cabin']), # s.th goes wrong when this line is included
]

trans = ColumnTransformer(pipe_transformer)

# Now we can feature engineer our Xtrain and Xtest using one-liners:

Xtrain_fe = trans.fit_transform(Xtrain)
Xtrain_fe_df = pd.DataFrame(Xtrain_fe, columns = trans.get_feature_names_out())
Xtrain_fe_df.head()   

Xtest_fe = trans.fit_transform(Xtest)
Xtest_fe_df = pd.DataFrame(Xtest_fe, columns = trans.get_feature_names_out())
Xtest_fe_df.head()  

#%% Logistic Regression

m_logreg = LogisticRegression(class_weight='balanced', max_iter = 10000)
m_logreg.fit(Xtrain_fe, ytrain)

# look at the model parameters
w_0 = m_logreg.intercept_[0]
w_1 = m_logreg.coef_[0][0]
w_2 = m_logreg.coef_[0][1]
print(f'LogReg model coefficients :{w_1, w_2}\nLogReg model intercept: {w_0}')
# identified classes (for now just 0: drowned, and 1:survived)
m_logreg.classes_
m_logreg.coef_.shape

# predictions (0 and 1 for now)
target_pred_train = m_logreg.predict(Xtrain_fe)
target_pred_test = m_logreg.predict(Xtest_fe)
# predicted probabilities
pred_probs_train = m_logreg.predict_proba(Xtrain_fe)
pred_probs_test = m_logreg.predict_proba(Xtest_fe)
# let's put this in a dataframe for easier reading
# pred_probs_test_df = pd.DataFrame(data = pred_probs_test, columns = m_logreg.classes_)
# pred_probs_test_df = pred_probs_test_df.rename(columns = {0: 'p_drowned', 1: 'p_survived'})

# evaluate
acc_train = m_logreg.score(Xtrain_fe, ytrain) 
acc_test = m_logreg.score(Xtest_fe, ytest) 
print(f'''training score = {m_logreg.score(Xtrain_fe, ytrain)}, 
test score = {round(m_logreg.score(Xtest_fe, ytest), 2)}''')

# Cross-Validation 

cross_validation_m_logreg = cross_val_score(estimator=m_logreg, # the model to evaluate
                                             X=Xtrain_fe,
                                             y=ytrain,
                                             scoring='accuracy', #'neg_mean_squared_log_error' # evaluation metrics
                                             cv=5, # cross validation splitting
                                             verbose = 3
) 

print(f'\nvalidation scores (accuracy): {np.round(cross_validation_m_logreg,2)}')
print(f'\nmean: {cross_validation_m_logreg.mean():.2}')
print(f'std: {cross_validation_m_logreg.std():.2}')
print(f'\ntest score (accuracy): {round(m_logreg.score(Xtest_fe,ytest),3)}')


#%% precision, Recall, and F1
print(f"""precision_test = {round(precision_score(ytest, target_pred_test),2)} \n 
recall_test = {round(recall_score(ytest, target_pred_test),2)}\n
f1_test = {round(f1_score(ytest, target_pred_test),2)}""")


#%% Confusion Matrix for test data
# what we would want to see in conf plot is 0:0 (TN) and 1:1 (TP) values to be high and 0:1 (FP) and 1:0 (FN) to be low.
conf = confusion_matrix(ytest, target_pred_test)

def plot_heatmap(confusion):
    
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion,
                xticklabels = np.unique(y),
                yticklabels = np.unique(y),
                cmap = 'BuPu',
                annot=True,
                fmt = 'g'
               )

    # fmt is used to switch off scientific notation
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize = 14)

plot_heatmap(conf)

#%% random forest

# Instanciate a model
m_rf = RandomForestClassifier(n_estimators=100, 
                              max_depth=4, 
                              max_features=2)
# Fit it to the training data
m_rf.fit(Xtrain_fe, ytrain)
# Look at the training score
m_rf.score(Xtrain_fe, ytrain)
# Look at the test score
m_rf.score(Xtest_fe, ytest) 
# Get predictions:
rf_pred_test = m_rf.predict(Xtest_fe)
rf_pred_probs_test = m_rf.predict_proba(Xtest_fe)

#%% ROC for test data comparing random forest to logreg

# set up plotting area
plt.figure(0).clf()

# ROC for LogReg
fpr, tpr, threshold = roc_curve(ytest, pred_probs_test [:,1]);
auc_lr = round(metrics.roc_auc_score(ytest, pred_probs_test [:,1]), 4)
# plot_roc_curve(m_logreg, Xtest_fe, ytest)
plt.plot(fpr,tpr,label="Logistic Regression, AUC="+str(auc_lr))

# ROc for Random Forest
fpr, tpr, threshold = roc_curve(ytest, rf_pred_probs_test [:,1]);
auc_rf = round(metrics.roc_auc_score(ytest, rf_pred_probs_test [:,1]), 4)
plt.plot(fpr,tpr,label="Random Forest, AUC="+str(auc_rf))

plt.legend()

