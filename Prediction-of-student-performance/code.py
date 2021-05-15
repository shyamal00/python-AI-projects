#load dataset of students
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score


data = pd.read_csv('D:\Python Projects\Prediction-of-student-performance\data.csv', sep=';') #seperate data by ';'
a = len(data) #Number of rows in data
#print(a)

#apply pass in row and when g1,g2,g3 is > 35 apply 1(pass) else 0(fail)
data['pass'] = data.apply(lambda row: 1 if(row['G1']+row['G2']+row['G3'])>= 35 else 0,axis=1)
data = data.drop(['G1','G2','G3'],axis=1)
data.head
#print(data)

#converted data set into numbers 
data = pd.get_dummies(data,columns=['sex','school','address','famsize','Pstatus','Mjob','Fjob','reason'
                                    ,'guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'])
data.head()
#print(data)

#shuffle rows
data = data.sample(frac=1)
#split training and testing data
data_train = data[:500]
data_test = data[500:]

data_train_att = data_train.drop(['pass'],axis=1)
data_train_pass = data_train['pass']

data_test_att = data_test.drop(['pass'],axis=1)
data_test_pass = data_test['pass']

data_att = data.drop(['pass'],axis=1)
data_pass = data['pass']

#number of passing students in dataset
print("passing %d out of %d (%.2f%%)" % (np.sum(data_pass),len(data_pass),100*float(np.sum(data_pass))/len(data_pass)))

# fit a decision tree
t = tree.DecisionTreeClassifier(criterion="entropy",max_depth=5)
t = t.fit(data_train_att,data_train_pass)

## graphical interface of 5 questions ##
#tr = tree.export_graphviz(t,out_file="student-performance.dot",label="all",impurity=False,proportion=True
                            #,feature_names=list(data_train_att),class_names=["fail","pass"],
                            #filled=True, rounded=True)


a = t.score(data_test_att,data_test_pass)
#print(a)

#scores = cross_val_score(t,data_att,data_pass,cv=5)
#print("accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2))

#for max_depth in range (1,20):
   # t = tree.DecisionTreeClassifier(criterion="entropy",max_depth=max_depth)
    #scores = cross_val_score(t,data_att,data_pass,cv=5)
    #print("accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2))


####model shows that depth of 7 questions would be the best to indentify the correct performance####
depth_acc = np.empty((19,3),float)
i=0
for max_depth in range(1,20):
    t = tree.DecisionTreeClassifier(criterion="entropy",max_depth=max_depth)
    scores = cross_val_score(t,data_att,data_pass,cv=5)
    depth_acc[i,0] = max_depth
    depth_acc[i,1] = scores.mean()
    depth_acc[i,2] = scores.std() * 2
    i += 1 
    print(depth_acc)


"""
Summary
the depth of 7 question would give the information about student performance,
and in this project i have learned about decision trees and also created a model to predict student performance 
"""