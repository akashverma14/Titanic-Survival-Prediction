# Titanic Survival Prediction

#1. Import Necessary Libraries
#data analysis libraries
import numpy as np
import pandas as pd
pd.set_option('precision', 2)

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sbn

#ignore warning
import warnings
warnings.filterwarnings('ignore')

#2. Read in and explore the data
#import train and test csv files
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#take a look at the training data
print train.describe()
print "\n"

print train.describe(include='all')
print "\n"
#Step-3 Data Analysis
#get a list of the features within the dataset

print  "\n\n",train.columns
#see a sample of the dataset
print
print train.head()
print
print train.sample()

#check datatypes for each column(only numerical data types can be used directly)
print
print"Data Types for each feature :-"
print train.dtypes

#check for any other unusable values
print
print pd.isnull(train).sum()
print

#Step-4 Data Visualisation

#4.A Sex Feature
# draw a bar plot of survival by sex
sbn.barplot(x="Sex",y="Survived",data=train)
plt.show()

print"...........................................\n\n"
print train

print"...........................................\n\n"
print train["Survived"]

print"...........................................\n\n"
print train["Sex"]=='female'

print"*********************************************************\n\n"
print train["Survived"][train["Sex"]== 'female']
#tells about only index of female and their survival status

print"*********************************************************\n\n"
print train["Survived"][train["Sex"]== 'female'].value_counts()

print"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n"
print train["Survived"][train["Sex"]== 'female'].value_counts(normalize=True)

print"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n"
print train["Survived"][train["Sex"]== 'female'].value_counts(normalize=True)[1]

print"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n"
print "Percentage of female who survived",train["Survived"][train["Sex"]== 'female'].value_counts(normalize=True)[1]*100
print "Percentage of female who survived",train["Survived"][train["Sex"]== 'male'].value_counts(normalize=True)[1]*100

#4.B PClass Feature
# draw a bar plot of survival by pclass
sbn.barplot(x="Pclass",y="Survived",data=train)
plt.show()

print"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n"
print "Percentage of Pclass =1 who survived",train["Survived"][train["Pclass"]== 1].value_counts(normalize=True)[1]*100
print "Percentage of Pclass =2 who survived",train["Survived"][train["Pclass"]== 2].value_counts(normalize=True)[1]*100
print "Percentage of Pclass =3 who survived",train["Survived"][train["Pclass"]== 3].value_counts(normalize=True)[1]*100

#4.C SibSp Feature
# draw a bar plot of survival by SibSp
sbn.barplot(x="SibSp",y="Survived",data=train)
plt.show()

print"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n"
print "Percentage of SibSp =0 who survived",train["Survived"][train["SibSp"]== 0].value_counts(normalize=True)[1]*100
print "Percentage of SibSp =1 who survived",train["Survived"][train["SibSp"]== 1].value_counts(normalize=True)[1]*100
print "Percentage of SibSp =2 who survived",train["Survived"][train["SibSp"]== 2].value_counts(normalize=True)[1]*100

#4.D Parch Feature
# draw a bar plot of survival by Parch vs Survival
sbn.barplot(x="Parch",y="Survived",data=train)
plt.show()

#4.E Age Feature

#sort the ages into loical categories
train["Age"] = train["Age"].fillna(-0.5)#fill the age whose value is missing
test["Age"] = test["Age"].fillna(-0.5)

bins= [-1,0,5,12,18,24,35,60,np.inf]#np.inf= positive infinity
labels= ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']#-1-0=unknown,0-5=baby
train['AgeGroup']=pd.cut(train["Age"],bins,labels=labels)#to check every data of age that it lies in which category
test['AgeGroup']=pd.cut(test["Age"],bins,labels=labels)
print train
#draw a bar plot of age vs survival
sbn.barplot(x="AgeGroup",y="Survived",data=train)
plt.show()

#4.F Cabin Feature
#I think the idea here is that with recorded cabin number are of high class and more likely to survive
train["CabinBool"]=(train["Cabin"].notnull().astype('int'))
test["CabinBool"]=(test["Cabin"].notnull().astype('int'))
print "################################"
print train

# calculate percentage of CabinBool vs survived
print "Percentage of CabinBool =0 who survived",train["Survived"][train["CabinBool"]== 0].value_counts(normalize=True)[1]*100
print "Percentage of CabinBool =1 who survived",train["Survived"][train["CabinBool"]== 1].value_counts(normalize=True)[1]*100
#draw a bar plot of CabinBool vs survival
sbn.barplot(x="CabinBool",y="Survived",data=train)
plt.show()

#5.Cleaning Data

#time to clean our data to account for missing values and unnecssary information
print
print test.describe(include="all")
# function to know how many datas are missing in each column
print
print pd.isnull(test).sum()
print


#Some observation from above o/p of test.csv file
#1.We have a total of 418 passengers

#Cabin feature
#we'll start off by droping Cabin feature since it is not useful more
train=train.drop(['Cabin'],axis=1)
test=test.drop(['Cabin'],axis=1)

#Ticket feature
#we'll be also droping Cabin feature since not a lot more useful
train=train.drop(['Ticket'],axis=1)
test=test.drop(['Ticket'],axis=1)

#Embarked Feature
#now we need to fill in the missing values in Embarked feature
print "Number of people embarking in South Hampton(S):"
print "\n\nSHAPE= ",train[train["Embarked"] == "S"].shape
print "\nSHAPE[0]= ",train[train["Embarked"] == "S"].shape[0]

print "\nNumber of people embarking in Chernboyl(C):"
print "SHAPE[0]= ",train[train["Embarked"] == "C"].shape[0]

print "\nNumber of people embarking in Quuenstown(Q):"
print "SHAPE[0]= ",train[train["Embarked"] == "Q"].shape[0]

#replacing the missing values in the embarked feature with S
train=train.fillna({"Embarked":"S"})

# function to know how many datas are missing in each column
#print pd.isnull(train).sum()

#create a combined group of both dataset
combine = [train,test]
print combine[0]

#extract a title for each name in the train and test datasets
for dataset in combine:
    dataset['Title']= dataset['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)
    #here + denotes wild character that means 1 or more alphabet cann be read
    #expand =false means if title is found then don't search any further

print "\n\n"
print train
print "\n\n"

print pd.crosstab(train['Title'],train['Sex'])

#replace various title with more common name
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess','Sir'],'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

print "\nAfter grouping rare title: ",train
print pd.crosstab(train['Title'],train['Sex'])
print

print train[['Title','Survived']].groupby(['Title'],as_index=False).count()

print "\nMap each of tht title groups to a numerical value"
title_mapping={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Royal":5,"Rare":6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print "\nAfter replacing title with numerical value\n"
print train
print

#fill missing age with mode age group for each title
mr_age = train[train["Title"] == 1]["AgeGroup"].mode()
print "mode() of mr_age: ",mr_age
print "\n\n"

miss_age = train[train["Title"] == 2]["AgeGroup"].mode()
print "mode() of miss_age: ",miss_age
print "\n\n"

mrs_age = train[train["Title"] == 3]["AgeGroup"].mode()
print "mode() of mrs_age: ",mrs_age
print "\n\n"

master_age = train[train["Title"] == 4]["AgeGroup"].mode()
print "mode() of master_age: ",master_age
print "\n\n"

royal_age = train[train["Title"] == 5]["AgeGroup"].mode()
print "mode() of royal_age: ",royal_age
print "\n\n"

rare_age = train[train["Title"] == 6]["AgeGroup"].mode()
print "mode() of royal_age: ",royal_age
print "\n\n"

print"*****************************************************************************"
print train.describe(include="all")
print train

print "\n********** train[AgeGroup][0]: \n"
for x in range(10):
    print train["AgeGroup"][x]

age_title_mapping={1:"Young Adult",2:"Student",3:"Adult",4:"Baby",5:"Adult",6:"Adult"}

for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[ train["Title"][x] ]

for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[ train["Title"][x] ]

print train

#map each Age value to a numerical value
age_mapping={'Baby':1,'Child':2,'Teenager':3,'Student':4,'Young Adult':5,'Adult':6,'Senior':7}

train['AgeGroup']=train['AgeGroup'].map(age_mapping)
test['AgeGroup']=test['AgeGroup'].map(age_mapping)
print
print train

#dropping the age feature for now
train=train.drop(['Age'],axis=1)
test=test.drop(['Age'],axis=1)

print "\n\nAge column droppped"
print train
print"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
#name feature
#we can drop the name feature now that we've extracted the titles

#drop the name feature since it contain no more useful info
train=train.drop(['Name'],axis=1)
test=test.drop(['Name'],axis=1)

#sex feature
#map each sex value to a numerical value
sex_mapping= {"male":0,"female":1}
train['Sex']=train["Sex"].map(sex_mapping)
test['Sex']=test["Sex"].map(sex_mapping)
print train

#Embarked Feature
#map each embarked value to a numerical value
embarked_mapping={"S":1,"C":2,"Q":3}
train['Embarked']=train["Embarked"].map(embarked_mapping)
test['Embarked']=test["Embarked"].map(embarked_mapping)
print
print test.head()

#fare feature
#it is time seperate the fare values into some logical groups as well as filling in the single missing values in the TEST dataset

#fill in missing Fare values in test set based on mean fare for the Pclass
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass=test["Pclass"][x]
        test["Fare"][x]=round(train[train["Pclass"]==pclass]["Fare"].mean(),2)

#map fare values into groups of numerical values
train['FareBand']=pd.qcut(train['Fare'], 4, labels=[1,2,3,4])
test['FareBand']=pd.qcut(test['Fare'], 4, labels=[1,2,3,4])
#qcut => used to divide the data into diff group
# 4 means fare is divided into 4 groups
#labels=[1,2,3,4] => assigning name to each group

#drop fare values
train=train.drop(['Fare'],axis=1)
test=test.drop(['Fare'],axis=1)
#check train data
print "\n\nfare column dropped"
print train

#**********************************
#6.Choosing the best model
#**********************************

#splitting hr training data
#We will use part of our training data(20% in this case) to test the accuracy of algo

from sklearn.model_selection import train_test_split
input_predictors=train.drop(['Survived','PassengerId'], axis=1)
output_target=train["Survived"]
x_train,x_val,y_train,y_val=train_test_split(input_predictors,output_target,test_size=0.20,random_state=7)


#Testing Different Models
# testing data will be tested with diff models to find accuracy


from sklearn.metrics import accuracy_score

# Model-1. LogisticRegression
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_val)
acc_logreg= round(accuracy_score(y_pred,y_val)*100 , 2)
print "MODEL-1: Accuracy of LogisticRegression: ",acc_logreg

# Model-2. Gaussian Naive Byes
from sklearn.naive_bayes import GaussianNB
gaussian=GaussianNB()
gaussian.fit(x_train,y_train)
y_pred=gaussian.predict(x_val)
acc_gaussian= round(accuracy_score(y_pred,y_val)*100 , 2)
print "MODEL-2: Accuracy of Gaussian Naive Byes: ",acc_gaussian

# Model-3. Support Vector Machines
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_val)
acc_svc= round(accuracy_score(y_pred,y_val)*100 , 2)
print "MODEL-3: Accuracy of Support Vector Machines: ",acc_svc

# Model-4. Linear SVC
from sklearn.svm import LinearSVC
linear_svc=LinearSVC()
linear_svc.fit(x_train,y_train)
y_pred=linear_svc.predict(x_val)
acc_linear_svc= round(accuracy_score(y_pred,y_val)*100 , 2)
print "MODEL-4: Accuracy of Linear SVC: ",acc_linear_svc


# Model-5. Perceptron
from sklearn.linear_model import Perceptron
perceptron=Perceptron()
perceptron.fit(x_train,y_train)
y_pred=perceptron.predict(x_val)
acc_perceptron= round(accuracy_score(y_pred,y_val)*100 , 2)
print "MODEL-5: Accuracy of Perceptron: ",acc_perceptron


# Model-6. Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
decisiontree=DecisionTreeClassifier()
decisiontree.fit(x_train,y_train)
y_pred=decisiontree.predict(x_val)
acc_decisiontree =round(accuracy_score(y_pred,y_val)*100 , 2)
print "MODEL-6: Accuracy of Decision Tree Classifier: ",acc_decisiontree

# Model-7. Random Forest
from sklearn.ensemble import RandomForestClassifier
randomforest=RandomForestClassifier()
randomforest.fit(x_train,y_train)
y_pred=randomforest.predict(x_val)
acc_randomforest =round(accuracy_score(y_pred,y_val)*100 , 2)
print "MODEL-7: Accuracy of Random Forest: ",acc_randomforest

# Model-8. KNN or k-Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred=knn.predict(x_val)
acc_knn =round(accuracy_score(y_pred,y_val)*100 , 2)
print "MODEL-8: Accuracy of k-Nearest Neighbour: ",acc_knn


# Model-9. Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier()
sgd.fit(x_train,y_train)
y_pred=sgd.predict(x_val)
acc_sgd =round(accuracy_score(y_pred,y_val)*100 , 2)
print "MODEL-9: Accuracy of Stochastic Gradient Descent: ",acc_sgd


# Model-10. Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbk=GradientBoostingClassifier()
gbk.fit(x_train,y_train)
y_pred=gbk.predict(x_val)
acc_gbk =round(accuracy_score(y_pred,y_val)*100 , 2)
print "MODEL-10: Accuracy of Gradient Boosting Classifier: ",acc_gbk

#Let's compare the accuracies of each model

models=pd.DataFrame({'Model':['LogisticRegression','Gaussian Naive Byes','Support Vector Machines',
                              'Linear SVC','Perceptron','Decision Tree Classifier','Random Forest',
                              'k-Nearest Neighbour','Stochastic Gradient Descent',
                              'Gradient Boosting Classifier'],
                     'Score':[acc_logreg,acc_gaussian,acc_svc,acc_linear_svc,acc_perceptron,
                              acc_decisiontree,acc_randomforest,acc_knn,acc_sgd,acc_gbk]
                     })
print
print models.sort_values(by='Score',ascending=False)

#i decided  to use the random forest model for the testing data

#7. Creating Submission File

#set ids as PassengerId and predict survival
ids=test['PassengerId']
predictions=randomforest.predict(test.drop('PassengerId',axis=1))

#set the output as a dataframe and convert to csv file named submimssion.csv
output = pd.DataFrame({'PassengerId': ids ,'Survived': predictions})
output.to_csv('Submission.csv',index=False)

print "All survival predictions done"
print "All predictions exportd to Submission.csv file"

print output