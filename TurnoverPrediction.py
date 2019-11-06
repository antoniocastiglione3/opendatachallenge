# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:05:28 2019

@author: antonio.castiglione
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn import model_selection
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

#read data
pathScripts="C:\\Users\\antonio.castiglione\\codiciPy\\"
df2=pd.read_csv(pathScripts+'\\dfTurnover1.csv',sep=',')
df2=pd.DataFrame(df2)
df2=df2.drop(df2.columns[[0]], axis=1)


# Create dummy variables for the 'department' and 'position' features, since they are categorical 
department = pd.get_dummies(data=df2['department'],drop_first=True,prefix='dep') #drop first column to avoid dummy trap
position = pd.get_dummies(data=df2['position'],drop_first=True,prefix='pos')
education = pd.get_dummies(data=df2['education_level'],drop_first=True,prefix='edu')
major = pd.get_dummies(data=df2['major'],drop_first=True,prefix='maj')
df2.drop(['department','position','education_level','major'],axis=1,inplace=True)
df2 = pd.concat([df2,department,position,education,major],axis=1)

# Create base rate model
def base_rate_model(X) :
    y = np.zeros(X.shape[0])
    return y

# Create train and test splits and scaling variables for machine learning
target_name = 'turnover'
X = df2.drop('turnover', axis=1)
robust_scaler = RobustScaler()
X = robust_scaler.fit_transform(X)
y=df2[target_name]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=123, stratify=y)



# Check accuracy of Logistic Model
logis = LogisticRegression(penalty='l2', C=1)
logis.fit(X_train, y_train)
print ("Logistic accuracy is %2.2f" % accuracy_score(y_test, logis.predict(X_test)))
logit_roc_auc = roc_auc_score(y_test, logis.predict(X_test))
print ("Logistic AUC = %2.2f" % logit_roc_auc)
######probability leaves employee
probslog = logis.predict_proba(X_test)[:, 1] # predict probabilities associated with the employee leaving
logisProb_roc_auc = roc_auc_score(y_test, probslog) # calculate AUC score using test dataset
print('AUC score: %.3f' % logisProb_roc_auc)

print(classification_report(y_test,logis.predict(X_test)))

# Random Forest Model
rf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=20, class_weight="balanced")
rf.fit(X_train, y_train)
print("Random Forest accuracy is %2.2f" % accuracy_score(y_test, rf.predict(X_test)))
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
print ("Random Forest AUC = %2.2f" % rf_roc_auc)
######probability leaves employee
probsrf = rf.predict_proba(X_test)[:, 1] # predict probabilities associated with the employee leaving
rfProb_roc_auc = roc_auc_score(y_test, probsrf) # calculate AUC score using test dataset
print('AUC score: %.3f' % rfProb_roc_auc)

print(classification_report(y_test, rf.predict(X_test)))
#print('Accuracy of Random Forest Classifier on test set: {:.2f}'.format(rf.score(X_test, y_test)*100))

# Classification report for the optimised RF Regression
rf.fit(X_train, y_train)
rfp=rf.predict(X_test)

#adaboost model
ada = AdaBoostClassifier(n_estimators=100, random_state=0)
ada.fit(X_train, y_train)
print ("AdaBoost accuracy is %2.2f" % accuracy_score(y_test, ada.predict(X_test)))
ada_roc_auc = roc_auc_score(y_test, ada.predict(X_test))
print ("AdaBoost AUC = %2.2f" % ada_roc_auc)
######probability leaves employee
probsada = ada.predict_proba(X_test)[:, 1] # predict probabilities associated with the employee leaving
adaProb_roc_auc = roc_auc_score(y_test, probsada) # calculate AUC score using test dataset
print('AUC score: %.3f' % adaProb_roc_auc)

print(classification_report(y_test, ada.predict(X_test)))

#decision tree model
dtree = tree.DecisionTreeClassifier(max_depth=3,class_weight="balanced",min_weight_fraction_leaf=0.01)
dtree.fit(X_train, y_train)
print ("Decision Tree accuracy is %2.2f" % accuracy_score(y_test, dtree.predict(X_test)))
dt_roc_auc = roc_auc_score(y_test, dtree.predict(X_test))
print ("Decision Tree AUC = %2.2f" % dt_roc_auc)
######probability leaves employee
probsdtree = dtree.predict_proba(X_test)[:, 1] # predict probabilities associated with the employee leaving
dtreeProb_roc_auc = roc_auc_score(y_test, probsdtree) # calculate AUC score using test dataset
print('AUC score: %.3f' % dtreeProb_roc_auc)

print(classification_report(y_test, dtree.predict(X_test)))

############################################################################################

# Create ROC Graph
from sklearn.metrics import roc_curve
log_fpr, log_tpr, log_thresholds = roc_curve(y_test, logis.predict_proba(X_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dtree.predict_proba(X_test)[:,1])
ada_fpr, ada_tpr, ada_thresholds = roc_curve(y_test, ada.predict_proba(X_test)[:,1])

plt.figure()

# Plot Logistic Regression ROC
plt.plot(log_fpr, log_tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)

# Plot Decision Tree ROC
plt.plot(dt_fpr, dt_tpr, label='Decision Tree (area = %0.2f)' % dt_roc_auc)

# Plot AdaBoost ROC
plt.plot(ada_fpr, ada_tpr, label='AdaBoost (area = %0.2f)' % ada_roc_auc)

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate' 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()


#features importance
#################################################################################
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12,6)

## plot the importances 2 ##
importances = rf.feature_importances_
names = df2.drop(['turnover'],axis=1).columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))
plt.title("Feature importances by rf Classifier")
plt.bar(range(len(indices)), importances[indices],   align="center")

plt.xticks(range(len(indices)), names[indices], rotation='vertical',fontsize=14)
#plt.xlim([-1, len(indices)])
plt.show()
########################################################
## Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, rf.predict(X_test))
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
########################################################à
print('Accuracy of RandomForest Regression Classifier on test set: {:.2f}'.format(rf.score(X_test, y_test)*100))

# Classification report for the optimised RF Regression
rf.fit(X_train, y_train)
print(classification_report(y_test, rf.predict(X_test)))
######################################################
