import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load the data
db = './data'
titanic_train = pd.read_csv(os.path.join(db, 'titanic_train.csv'), index_col=0)

#%% EDA

sns.set_theme(style="whitegrid")
# layering multiple hist on top of each other (default method)
# sns.histplot(data=titanic_train, x="Age", hue ='Survived', kde=True, color = 'aquamarine') #, binwidth=3)
# stacking multiple hist on top of each other 
sns.histplot(data=titanic_train, x="Age", hue ='Sex', kde=True, color = 'aquamarine', multiple="stack") #, binwidth=3)
plt.title(' ')
# plt.axis((xmin, xmax, ymin, ymax)) 
# plt.axis((20, 100, 0, 10))
plt.xlabel('age')
plt.ylabel('number of passengers')
plt.legend(loc='upper right')
plt.legend(['female', 'male']) # titanic_train['Sex'].value_counts()
plt.gcf().set_size_inches(8, 5)
plt.ion(); plt.show()

#
sns.set_theme(style="whitegrid")
# layering multiple hist on top of each other (default method)
# sns.histplot(data=titanic_train, x="Age", hue ='Survived', kde=True, color = 'aquamarine') #, binwidth=3)
# stacking multiple hist on top of each other 
sns.histplot(data=titanic_train, x="Age", hue ='Pclass', kde=True, color = 'aquamarine', multiple="stack") #element="poly" )  #, binwidth=3)
plt.title(' ')
# plt.axis((xmin, xmax, ymin, ymax)) 
# plt.axis((20, 100, 0, 10))
plt.xlabel('age')
plt.ylabel('number of passengers')
plt.legend(loc='upper right')
plt.legend(['class 3', 'class 2', 'class 3'])
plt.gcf().set_size_inches(8, 5)
plt.ion(); plt.show()

# quit look at the data
# titanic_train.hist(bins=10,figsize=(9,7),grid=False);

# Survival by Class, Age and Fare
g = sns.FacetGrid(titanic_train, hue="Survived", col="Pclass", margin_titles=True,
                palette="magma",hue_kws=dict(marker=["^", "v"]))
g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Survival by Class, Age and Fare');

# violin plot
# 63% of the first class people survived while 24% of the lower class survived
titanic_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(
    'Survived', ascending = False)
plt.figure(figsize = (15, 6))
sns.violinplot(data = titanic_train, x = 'Pclass', y = 'Survived')

# correlation
# Pclass has got highest negative correlation with "Survived" followed by Fare, Parch and Age
# Fare has got highest positive correlation with "Survived" followed by Parch

corr=titanic_train.corr()
plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01, square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');
titanic_train.corr()["Survived"]