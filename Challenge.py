#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# In[1]:


import numpy as np  
import pandas as pd 

import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv("D:\\Nueva carpeta\\Train_data.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data['Target'].value_counts()


# In[5]:


data.isna().sum()


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


data.hist(bins=10, figsize=(16,12))
plt.show()


# In[8]:


plt.figure(figsize=(16, 12))
sns.heatmap(data.corr(), cmap='bwr', annot=True) # annot = True: to display the correlation value in the graph


# In[9]:


sns.pairplot(data[["GP","MIN","PTS","FG%","AST","STL","BLK","TOV","Target"]],hue="Target")


# ### New features
# 
# I'll use clusters as a feature because there are some position where a player could play and the criterium of succesful is different among positions. I expect that the clusters recognise this patterns and help the algorimths to classified 

# In[12]:


from sklearn.cluster import KMeans

clustering = KMeans(n_clusters = 4).fit(data.drop("Target", axis = 1))

data["cluster"] = clustering.predict(data.drop("Target", axis = 1))


# In[13]:


sns.pairplot(data[["GP","MIN","PTS","FG%","AST","STL","BLK","TOV","cluster"]],hue="cluster")


# In[14]:


data = pd.get_dummies(data=data, columns=['cluster'])


# # Model training
# 
# ## XGBoost

# In[15]:


from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

y = data.Target                
X = data.drop('Target', axis=1) 

seed = 8
test_size = 0.10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


# In[67]:


model = XGBClassifier(use_label_encoder=False, eval_metric='error',
                      learning_rate=0.2,gamma=0.25,max_depth=3) # Recall for ensure precision


# In[68]:


model.fit(X_train, y_train)


# In[69]:


y_pred_xg = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_xg)
accuracy


# In[66]:


metrics.confusion_matrix(y_test, y_pred_xg)


# ## Random Forest

# In[20]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)

y_pred_rf = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_rf)
accuracy


# In[21]:


metrics.confusion_matrix(y_test, y_pred_rf)


# ## SVM

# In[22]:


from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

svm = make_pipeline(StandardScaler(), SVC(kernel="poly",gamma="scale",probability=True))
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_svm)
accuracy


# In[23]:


metrics.confusion_matrix(y_test, y_pred_svm)


# ## Gradient Boosting

# In[41]:


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2,
                                 max_depth=2, random_state=0).fit(X_train, y_train)

y_pred_gbc = gbc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_gbc)
accuracy


# In[40]:


metrics.confusion_matrix(y_test, y_pred_gbc)


# # Voting

# In[103]:


from sklearn.ensemble import VotingClassifier

clf1 = XGBClassifier(use_label_encoder=False, eval_metric='error',
                      learning_rate=0.3,gamma=0.25,max_depth=6)
clf2 = RandomForestClassifier(n_estimators=100)
clf3 = make_pipeline(StandardScaler(), SVC(kernel="poly",gamma="scale",probability=True))
clf4 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.3,
                                 max_depth=2, random_state=0)

eclf1 = VotingClassifier(estimators=[('xgb', clf1), ('rf', clf2), ('svm', clf3), ('gbc', clf4)], 
                         voting='soft',weights = [2,1,1,1])
eclf1 = eclf1.fit(X_train, y_train)

y_pred_voting = eclf1.predict(X_test)


# In[104]:


accuracy = accuracy_score(y_test, y_pred_voting)
accuracy


# In[105]:


eclf_total = eclf1.fit(X, y)

y_pred_total = eclf_total.predict(X)

accuracy = accuracy_score(y, y_pred_total)
accuracy


# # Prediction

# In[106]:


test = pd.read_csv("D:\\Nueva carpeta\\Test_data.csv")

test.head()


# In[107]:


test["cluster"] = clustering.predict(test)
test = pd.get_dummies(data=test, columns=['cluster'])

prediction = eclf_total.predict(test)


# In[108]:


prediction = pd.DataFrame(prediction, columns=["prediction"])

prediction


# In[109]:


prediction.to_csv("D:\\Nueva carpeta\\Submission.csv",index=False)


# In[ ]:




