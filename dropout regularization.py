#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.constraints import max_norm


# In[ ]:


seed=7
np.random.seed(seed)
dataframe=read_csv("sonar.csv",header=None)
dataset=dataframe.values
x=dataset[:,0:60].astype(float)
y=dataset[:,60]
encoder=LabelEncoder()
encoder.fit(y)
encodedy=encoder.transform(y)


# In[9]:


def build_model():
    model=Sequential()
    model.add(Dense(60,input_dim=60,kernel_initializer='normal',activation='relu'))
    model.add(Dense(30,kernel_initializer='normal',activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))
    sgd=SGD(lr=0.01,momentum=0.8,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=300,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)

kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("BASELINE:%.2f%%(%.2f%%)"%(result.mean(),result.std()))


# In[12]:


def build_model():
    model=Sequential()
    model.add(Dense(60,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='sigmoid'))
    sgd=SGD(lr=0.1,momentum=0.9,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=300,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("visible:%.2f%%(%.2f%%)"%(result.mean(),result.std()))


# # TUNNING THE MODEL

# In[6]:


def build_model():
    model=Sequential()
    model.add(Dense(65,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(34,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='sigmoid'))
    sgd=SGD(lr=0.1,momentum=0.9,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=20,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("visible:%.2f%%(%.2f%%)"%(result.mean(),result.std()))
def build_model():
    model=Sequential()
    model.add(Dense(1260,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(350,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='sigmoid'))
    sgd=SGD(lr=0.1,momentum=0.9,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=500,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("visible:%.2f%%(%.2f%%)"%(result.mean(),result.std()))
def build_model():
    model=Sequential()
    model.add(Dense(40,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='sigmoid'))
    sgd=SGD(lr=0.1,momentum=0.9,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=300,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("visible:%.2f%%(%.2f%%)"%(result.mean(),result.std()))


# # HIDDEN LAYERS 

# In[10]:


def build_model():
    model=Sequential()
    model.add(Dense(60,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(30,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='sigmoid'))
    sgd=SGD(lr=0.1,momentum=0.9,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=300,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("HIDDEN:%.2f%%(%.2f%%)"%(result.mean(),result.std()))


# # TUNNIG FOR HIDDEN LAYERS

# In[ ]:


def build_model():
    model=Sequential()
    model.add(Dense(50,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(40,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(80,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='sigmoid'))
    sgd=SGD(lr=0.1,momentum=0.9,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=500,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("HIDDEN:%.2f%%(%.2f%%)"%(result.mean(),result.std()))
def build_model():
    model=Sequential()
    model.add(Dense(60,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(10,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='sigmoid'))
    sgd=SGD(lr=0.01,momentum=0.8,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=400,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("HIDDEN:%.2f%%(%.2f%%)"%(result.mean(),result.std()))
def build_model():
    model=Sequential()
    model.add(Dense(20,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(10,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(5,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='sigmoid'))
    sgd=SGD(lr=0.1,momentum=0.9,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=600,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("HIDDEN:%.2f%%(%.2f%%)"%(result.mean(),result.std()))


# # 8.1

# In[ ]:


def build_model():
    model=Sequential()
    model.add(Dense(60,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(30,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='sigmoid'))
    sgd=SGD(lr=0.01,momentum=0.8,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=20,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("visible:%.2f%%(%.2f%%)"%(result.mean(),result.std()))
def build_model():
    model=Sequential()
    model.add(Dense(60,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(30,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='sigmoid'))
    sgd=SGD(lr=0.01,momentum=0.8,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=500,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("visible:%.2f%%(%.2f%%)"%(result.mean(),result.std()))
def build_model():
    model=Sequential()
    model.add(Dense(60,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(30,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='sigmoid'))
    sgd=SGD(lr=0.01,momentum=0.8,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=300,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("visible:%.2f%%(%.2f%%)"%(result.mean(),result.std()))


# # 8.2

# In[ ]:


def build_model():
    model=Sequential()
    model.add(Dense(650,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(90,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(100,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(95,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='sigmoid'))
    sgd=SGD(lr=0.01,momentum=0.8,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=20,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("visible:%.2f%%(%.2f%%)"%(result.mean(),result.std()))
def build_model():
    model=Sequential()
    model.add(Dense(60,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(100,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(35,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='sigmoid'))
    sgd=SGD(lr=0.01,momentum=0.8,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=500,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("visible:%.2f%%(%.2f%%)"%(result.mean(),result.std()))
def build_model():
    model=Sequential()
    model.add(Dense(290,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(150,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(280,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='sigmoid'))
    sgd=SGD(lr=0.01,momentum=0.8,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=300,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("visible:%.2f%%(%.2f%%)"%(result.mean(),result.std()))


# # 8.3

# In[11]:


def build_model():
    model=Sequential()
    model.add(Dense(60,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='sigmoid'))
    sgd=SGD(lr=0.01,momentum=0.8,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=300,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("HIDDEN:%.2f%%(%.2f%%)"%(result.mean(),result.std()))


# # 8.4

# In[ ]:


def build_model():
    model=Sequential()
    model.add(Dense(60,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(3.),activation='sigmoid'))
    sgd=SGD(lr=0.1,momentum=0.99,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=300,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("visible:%.2f%%(%.2f%%)"%(result.mean(),result.std()))


# # 8.5

# In[13]:


def build_model():
    model=Sequential()
    model.add(Dense(60,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(4.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30,kernel_initializer='normal',kernel_constraint=max_norm(4.),activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(4.),activation='sigmoid'))
    sgd=SGD(lr=0.1,momentum=0.99,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=300,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("visible:%.2f%%(%.2f%%)"%(result.mean(),result.std()))
def build_model():
    model=Sequential()
    model.add(Dense(60,input_dim=60,kernel_initializer='normal',kernel_constraint=max_norm(5.),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30,kernel_initializer='normal',kernel_constraint=max_norm(5.),activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',kernel_constraint=max_norm(5.),activation='sigmoid'))
    sgd=SGD(lr=0.1,momentum=0.99,decay=0.0,nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=300,batch_size=16,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,x,encodedy,cv=kfold)
print("visible:%.2f%%(%.2f%%)"%(result.mean(),result.std()))


# In[ ]:




