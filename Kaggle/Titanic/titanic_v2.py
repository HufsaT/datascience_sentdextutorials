# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 21:23:26 2018

@author: Hufsa
"""
# https://www.kaggle.com/helgejo/an-interactive-data-science-tutorial

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split , StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations

mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=True )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_correlation_map( df ):
    corr = titanic.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=False, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )
    
def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    print (model.score( X , y ))
    
train = pd.read_csv("train_titanic.csv")
test = pd.read_csv("test_titanic.csv")
full = train.append(test, ignore_index=True)
titanic = full[:891]
del train, test
#print("full: ", full.shape, " titanic: ", titanic.shape)
#print(full.info()) 
#print(full.head())

#print(titanic.describe())
#plot_correlation_map( titanic ) # finds correlation between all numeric values in array

# below func gives probability DENSITY (prob. of a single unit of x) for survival based on fare
#plot_distribution( titanic , var = 'Fare' , target = 'Survived' )

# now we get probs of survival by category class and breakdown Sex 
# (row/col determins order of charts)
#plot_categories(titanic,"Pclass","Survived", row="Sex")



def cleanTicket(ticket):
    ticket = ticket.replace('.','')
    ticket = ticket.replace('/','')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0: # filtered out numbers
        return ticket[0]
    else: 
        return 'XXX'
    
full['Ticket'] = full['Ticket'].map(cleanTicket)
tickets_dummies = pd.get_dummies(full['Ticket'], prefix='Ticket')
full = pd.concat([full, tickets_dummies], axis=1)
full.drop('Ticket', inplace=True, axis=1)

# creating new col for sex with binary values
full["Sex"] = full.Sex.map({'male':1,'female':0})

# get dummies for other values with more than 2 classes:
# first fill missing vals, then get dummies and add to full, dropping original col
full["Embarked"].fillna("S",inplace=True)
embarked = pd.get_dummies(full["Embarked"], prefix = "Embarked")
full = pd.concat([full, embarked],axis=1)
full.drop("Embarked",axis=1,inplace=True)

pclass = pd.get_dummies(full["Pclass"],prefix = "Pclass")
full = pd.concat([full,pclass],axis=1)
full.drop("Pclass",axis=1,inplace=True)

#now we ensure any missing values are filled with the mean of their column.
#full["Age"].fillna(full.Age.mean(),inplace=True)

#print(imputed.head())
#plot_distribution(titanic, "Fare" ,"Survived",row = "Sex")

# now we get class from people's titles.
full["Title"] = full.Name.map(lambda x: x.split(",")[1].split(".")[0].strip())
# for weight calc:
titanic["Title"] = full.Name.map(lambda x: x.split(",")[1].split(".")[0].strip())

Title_dict = {
        "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
        }

full["Title"] = full.Title.map(Title_dict)
titanic["Title"] = titanic.Title.map(Title_dict) # for weight calc

titles = pd.get_dummies(full.Title)
full = pd.concat([full,titles], axis=1)

# get pie chart of various titles vs survival
#fig = plt.figure(figsize=(10,10))
#i=1
#for t in titanic["Title"].unique():
#    fig.add_subplot(3,3,i)
#    plt.title("Title: {}".format(t))
#    titanic.Survived[titanic.Title==t].value_counts().plot(kind="pie")
#    plt.show()
#    i+=1

# get median age by title
full["Age"].fillna(-1,inplace=True)

medians = {}
for key, title in Title_dict.items():
    median = full["Age"][(full.Age != -1) & (full.Title == title)].median()
    medians[title] = median
    
for index,row in full.iterrows():
    if row["Age"] == -1:
        # can't use replace because loc returns a single value and replace
        # works on dfs. Can't return a df on loc because then we can't replace specific value.
#        print(full.loc[index,"Age"])
#        full.loc[[index,"Age"]].replace(-1,medians[row["Title"]],inplace=True)
        full.loc[index,"Age"] = medians[row["Title"]] # we can't use row[age] 
        # b/c can't edit a dataframe without .iloc/loc
#        print(row["Age"], row["Title"])

# repeat for fares
full["Fare"].fillna(-1,inplace=True)
medians = {}
for key, title in Title_dict.items():
    median = full.Fare[(full.Fare != -1) & (full.Title == title)].median()
    medians[title] = median

for index, row in full.iterrows():
    if row["Fare"] ==-1:
        full.loc[index,"Fare"] = medians[row["Title"]]

full.drop("Name",axis=1,inplace=True)    

# now extract cabin category/class by cabin num
full["Cabin"].fillna("U",inplace=True) # unknown
# map existing data to new data column, extracting first char
full["Cabin"] = full["Cabin"].map(lambda x: x[0])
# python lets you iterate in a string
cabin_dummies = pd.get_dummies(full["Cabin"], prefix = "Cabin")
full = pd.concat([full,cabin_dummies],axis=1)

# create a graph to see which cabins did better, using TRAINing data b/c we don't have survival values for full.
titanic["Cabin"].fillna("U",inplace=True)
titanic["Cabin"] = titanic.Cabin.map(lambda x:x[0])

#fig = plt.figure(figsize=(10,10))
#i = 1
#for cabin in titanic["Cabin"].unique():
#    fig.add_subplot(3,3,i)
#    plt.title("Cabin: {}".format(cabin))
#    titanic["Survived"][titanic.Cabin==cabin].value_counts().plot.pie(colors=["r","b"])
#    plt.show()
#    i+=1
# assign a weight to cabin data instead of deleting column
cabin_weight = {
        "T":0,
        "U":1,
        "A":2,
        "G":3,
        "F":4,
        "C":4,
        "E":5,
        "B":5,
        "D":6
        }
full["Cabin"]= full.Cabin.map(cabin_weight)

title_weight = {
        "Mrs": 6,
        "Miss":5,
        "Royalty":4,
        "Master":3,
        "Officer": 2,
        "Mr": 1
        }
full["Title"] = full.Title.map(title_weight)
#print(full.head(5))
#print(full.info()) 

# create new familysize var to see how total family correlates to survival
# initially created this col in titnaic df to check correlation, then added back to full
full["FamilySize"] = full.Parch + full.SibSp + 1
#plot_categories(titanic,"FamilySize","Survived",col="Sex")
full["Single"] = full.FamilySize.map(lambda x: 1 if x==1 else 0)
full["SmallFamily"] = full.FamilySize.map(lambda x: 1 if 2<=x<5 else 0)
full["LargeFamily"] = full.FamilySize.map(lambda x: 1 if 5<=x else 0)

#print(full.info()) 

full.drop("PassengerId",axis=1,inplace=True)
full.drop("Survived",axis=1,inplace=True)

# divide bak into train and test
train_x = full[:891]
train_y = titanic.Survived 
test_x = full[891:] # ignoring test data for now

clf = RandomForestClassifier(n_estimators=50,max_features="sqrt")
clf.fit(train_x,train_y)
#print(test_x.info())

# creating feature comp chart
feature = pd.DataFrame()
feature["Feature"] = train_x.columns
feature["Importance"] = clf.feature_importances_
feature.set_index("Feature",inplace=True)
feature.sort_values(by=["Importance"],ascending=True,inplace=True)
#feature.plot(kind="barh",figsize=(30,200))
model = SelectFromModel(clf, prefit=True ) # we ahve already fit our data
train_x_reduced = model.transform(train_x) # removes poor correlation features

test_x_reduced = model.transform(test_x) # also doing this to actual test data (not to the training test data)
#print(test_x_reduced.shape) 

# now create a quick program to calculate accuracy:
# because we are using cross_val, we do not need to split the original
# 891 data lines into their own train/test.
def compute_score(clf,x,y,scoring='accuracy'):
    xval = cross_val_score(clf,x,y,cv=5,scoring=scoring)
    return xval.mean()

# to find best model, we try 3 diff ones
svml = SVC()
gboost = GradientBoostingClassifier()
rf = RandomForestClassifier(n_estimators=100)
logreg = LogisticRegressionCV()
gaus = GaussianNB()
knear = KNeighborsClassifier()


models = [logreg, svml, rf, gboost,knear,gaus]

#for model in models:
#    print("Cross-validating: {0}".format(model.__class__))
#    score = compute_score(clf=model,x=train_x_reduced, y=train_y)
#    print("Accuracy of model: {0}".format(score))
#    print("*************")
    
model = GradientBoostingClassifier()
model.fit(train_x,train_y)
output = model.predict(test_x).astype(int) # so we don't get floats

passIDs = pd.read_csv("test_titanic.csv")
results = pd.DataFrame()
results["PassengerId"] = passIDs["PassengerId"]
results["Survived"] = output
print(results.shape)
results.to_csv("titanicsubmission.csv",index=False)


