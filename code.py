
# coding: utf-8

# In[23]:


import os
os.chdir(r"C:\Users\santh_000\Desktop\Kaggle\Digit recognizer")


# In[24]:


import numpy as np
import pandas as pd
import tflearn
import tensorflow as tf


# In[25]:


from keras.utils.np_utils import to_categorical


# In[26]:


import warnings
warnings.filterwarnings('ignore')


# In[27]:


data=pd.read_csv(r'C:\Users\santh_000\Desktop\Kaggle\Digit recognizer\train.csv')
test=pd.read_csv(r'C:\Users\santh_000\Desktop\Kaggle\Digit recognizer\test.csv')


# In[28]:


y_train = data.ix[:,0].values
train = data.ix[:,1:].values


# In[29]:


labels_one_hot = to_categorical(y_train)


# In[30]:


tf.reset_default_graph()


# In[31]:


net=tflearn.input_data(shape=[None,784])
net=tflearn.fully_connected (net,128,activation='relu',regularizer='L2', weight_decay=0.001)
net=tflearn.fully_connected (net,100,activation='relu',regularizer='L2', weight_decay=0.001)
net=tflearn.fully_connected (net,10,activation='softmax')
adam = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
net=tflearn.regression(net,loss='categorical_crossentropy',optimizer='adam')
model=tflearn.DNN(net)


# In[32]:


model.fit(train, labels_one_hot, validation_set=0.1, show_metric=True, batch_size=300, n_epoch=50)


# In[33]:


def prediction(predictions):
    return np.argmax(predictions,1)

predictions = prediction(model.predict(test))
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})

submissions.to_csv("submission.csv", index=False, header=True)

