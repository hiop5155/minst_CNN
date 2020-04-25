from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
np.random.seed(10) 
#讀dataset
(x_Train, y_Train),(x_Test, y_Test) = mnist.load_data()

#資料預處理
x_Train4D = x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D = x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')

#normalize
x_Train4D_normalize = x_Train4D/255
x_Test4D_normalize = x_Test4D/255

y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)
#modeling
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
model = Sequential()
#convolution layer 1
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1),
                 activation='relu'))
#pooling layer 1
model.add(MaxPooling2D(pool_size=(2,2)))
#covolution layer 2
model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))
#pooling layer 2
model.add(MaxPooling2D(pool_size=(2,2)))
#Dropout to avoid overfitting
model.add(Dropout(0.25))
#reshape to 1D input
model.add(Flatten())
#hidden layer 128 units
model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))
#output layer
model.add(Dense(10,activation='softmax'))

#training
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
for i in range(10):
    train_history=model.fit(x=x_Train4D_normalize,
                            y=y_TrainOneHot,validation_split=0.2,
                            epochs=10, batch_size=300,verbose=2)

import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.show()
    
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

#prdiction rate
scores = model.evaluate(x_Test4D_normalize, y_TestOneHot)
print('Test loss:', scores[0])
print('accuracy',scores[1])

prediction = model.predict_classes(x_Test4D_normalize)
prediction[:10]

#look lots image function
def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25: num=25
    for i in range(0, num):
        ax = plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title="label="+ str(labels[idx])
        if len(prediction)>0:
            title+=",prediction="+str(prediction[idx])
            
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show
plot_images_labels_prediction(x_Test,y_Test,prediction,idx=0)

#confusion matrix
import pandas as pd
pd.crosstab(y_Test,prediction,
            rownames=['label'],colnames=['predict'])
df = pd.DataFrame({'label':y_Test, 'predict':prediction})
df[:2]

#save model
model.save('minst_prediction_use_CNN.h5')

#load model
model = tf.contrib.keras.models.load_model('minst_prediction_use_CNN.h5')