# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:40:30 2020

@author: BESIME
"""


import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

#%% import data

df = pd.read_excel("quiz1.xlsx")
df.head()
#data.drop(["id","Unnamed: 32"], axis= 1, inplace= True)

#%% hasta=1 iyi=0 
y = df.sick.values
x = df.drop(["sick"],axis = 1)
x = pd.get_dummies(x)
x.head(2)

feat_names = x.columns
targ_names = ["Sick","Not Sick"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=42,test_size=.1)


from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix,f1_score
import matplotlib.pyplot as plt
import seaborn as sns

clf = DecisionTreeClassifier(max_depth=5).fit(x_train,y_train)
print("Training:"+str(clf.score(x_train,y_train)))
print("Test:"+str(clf.score(x_test,y_test)))
pred = clf.predict(x_train)
confusion_matrix = confusion_matrix(y_true=y_train,y_pred=pred)

sns.heatmap(confusion_matrix,annot=True,annot_kws={"size":16})
plt.show()

confusion_matrix

print("precision score : "+str(precision_score(y_train,pred))) # tp/tp+fp
print("accuracy score : "+str(accuracy_score(y_train,pred))) # total correct 
print("recall score : "+str(recall_score(y_train,pred)))   # tp/tp+fn
print("f1 score : "+str(f1_score(y_train,pred))) 

from IPython.display import Image  
from sklearn.externals.six import StringIO  
import pydotplus
import graphviz
# Convert to png using system command (requires Graphviz)

#dot_data = StringIO()
data = tree.export_graphviz(clf,out_file=None,feature_names=feat_names,class_names=targ_names,   
                         filled=True, rounded=True,  
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png("tree2.png")
#Image(graph.create_df())
#Image(filename="tree2.png")

img=pltimg.imread("tree2.png")
implot = plt.imshow(img)
plt.show()










