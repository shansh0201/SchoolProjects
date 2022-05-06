#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import copy
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


# ## Checkpoint and early stopping

# In[ ]:


def Callbacks(ckpt=None):
    
    model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',patience = 5)
    
    
    if ckpt is not None:
        
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            )
        
        print("Checkpoint saved as: '{}'".format(ckpt))

        return [model_early_stopping,model_checkpoint_callback]
    
    else:
        
        return [model_early_stopping]


# ## Modelling functions

# In[ ]:


class Model():
    
    size = (256,256)
    label_list_default = ['Coast', 'Forest', 'Highway', 'Insidecity', 'Mountain',                          'Office', 'OpenCountry', 'Street', 'Suburb', 'TallBuilding',                          'bedroom', 'industrial', 'kitchen', 'livingroom', 'store']
    
    def __init__(self, image_dir='./train', label_list=label_list_default, test_ratio=0.1, has_test=False):
        
        print("*"*60)
        
        self.image_dir = Path(image_dir)
        self.label_list = label_list
        self.test_ratio = test_ratio
        self.has_test = has_test
        
        self.le = preprocessing.LabelEncoder()
        self.le.fit(label_list)
        
        num_label = len(label_list)
        
        print("Constructing model with",num_label, "labels:\n",self.le.inverse_transform(range(num_label)))
        
        return
     
        
    def LoadImagesFromClassFolder(self,class_dir):
        
        images = []
        label = []
        labelname = str(class_dir).split('\\')[-1]
        for filename in os.listdir(class_dir):
            img = cv2.imread(os.path.join(class_dir,filename))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray,self.size, interpolation = cv2.INTER_AREA)
            if gray is not None:
                images.append(gray)
                label.append(labelname)  
                
        return images,label   
    
    
    
    def LoadImages(self):
        
        data=[]
        labels = []

        for foldername in os.listdir(self.image_dir):
            if foldername in self.label_list:
                class_dir = os.path.join(self.image_dir, foldername)
                img,label = self.LoadImagesFromClassFolder(class_dir)
                data.append(img)
                labels.append(label)
            
        data = np.vstack(data)
        labels = np.hstack(labels)

        # convert pixel range to (0,1)               
        data = np.array([x/255.0 for x in data])
        # channel=1
        data = data[...,np.newaxis]
        
        self.data = np.array(data)
        self.labels = np.array(labels)
        
        print("Images loaded.")
        
        return
    
    
    def ImageGenerate(self,x_train,y_train_int,n_copies=10):

        x_train_new = []
        y_train_int_new = []

        img_gen = ImageDataGenerator(featurewise_center=True,
                                     featurewise_std_normalization=True,
                                     zca_whitening=True,
                                    rotation_range=0,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    fill_mode='nearest',
                                    horizontal_flip=True,
                                    vertical_flip=False
                                    )
        for i in range(len(x_train)):
            gen = img_gen.flow(np.reshape(x_train[i],(1,self.size[0],self.size[1],1)), batch_size=1,
            # seed=1
            )
            for j in range(n_copies):
                for x in gen:
                    x_train_new.append(x[0])
                    y_train_int_new.append(y_train_int[i])
                    break
        
        return np.array(x_train_new), np.array(y_train_int_new)
    

    def PreprocessImages(self, n_copies=-1, random_state=1):
    
        num_img = len(self.data)
        num_label = len(set(self.labels))

        # convert label to number
        labels_int = self.le.transform(self.labels)

        # convert to one-hot
        labels_enc = tf.keras.utils.to_categorical(labels_int, num_classes=num_label)  

        
        if self.has_test and self.test_ratio>0:
            # train test split
            x_train, x_test, labels_train, labels_test = train_test_split(self.data, self.labels, test_size=self.test_ratio, random_state=random_state, stratify=self.labels)
            print("Test data splitted.")
        else:
            x_train = self.data
            x_test = None
            labels_train = self.labels
            labels_test = None
            
        if self.test_ratio>0:
            # train validation split
            x_train, x_val, labels_train, labels_val = train_test_split(x_train, labels_train, test_size=self.test_ratio, random_state=random_state, stratify=labels_train)
            print("Validation data splitted.")
        else:
            x_val = None
            labels_val = None

        labels_train = np.array(labels_train)
        y_train_int = self.le.transform(labels_train)
        y_train_enc = tf.keras.utils.to_categorical(y_train_int, num_classes=num_label) 
        
        if self.test_ratio>0:

            labels_val = np.array(labels_val)

            y_val_int = self.le.transform(labels_val)
            
            y_val_enc = tf.keras.utils.to_categorical(y_val_int, num_classes=num_label)  
        
        else:
            
            labels_val = y_val_int = y_val_enc = None
            
        
        
        if self.has_test:
        
            labels_test = np.array(labels_test)
            
            y_test_int = self.le.transform(labels_test)
            
            y_test_enc = tf.keras.utils.to_categorical(y_test_int, num_classes=num_label)  
        
        else:
            labels_test = y_test_int = y_test_enc = None
            

        if n_copies > 0:
            
            print("Creating",n_copies,"copies of transformed images...")
            
            x_train_new, y_train_int_new = self.ImageGenerate(x_train, y_train_int, n_copies)
            y_train_enc_new = tf.keras.utils.to_categorical(y_train_int_new, num_classes=num_label) 

            x_train = x_train_new
            y_train_int = y_train_int_new
            y_train_enc = y_train_enc_new
        
        self.x_train = x_train
        self.y_train_int = y_train_int
        self.y_train_enc = y_train_enc
        
        self.x_val = x_val
        self.y_val_int = y_val_int
        self.y_val_enc = y_val_enc
        
        self.x_test = x_test
        self.y_test_int = y_test_int
        self.y_test_enc = y_test_enc
        
        print("Images preprocessed.")
        
        return

    
    def AddNoise(self,max_noise=0.05):
    
        random.seed(1)
        noise = np.random.uniform(-max_noise,max_noise, size=self.x_train.shape)
        self.x_train = self.x_train + noise
        plt.imshow(self.x_train[0],'gray')
        
        print("Noise added.")
        
        return
        
    
    def ShowImages(self,n=20,cols=5):

        n = np.min([n,len(self.x_train)])
        rows = int(np.ceil(n/cols))
        idx = np.random.randint(0,len(self.x_train)-1,size=n)

        plt.figure(figsize=(3.2*cols,3.2*rows))
        for i in range(n):
            plt.subplot(rows,cols,i+1)
            plt.imshow(self.x_train[idx[i]],'gray')
            class_idx = self.y_train_int[idx[i]]
            plt.title('#{} - {}:{}'.format(i+1,class_idx,self.le.inverse_transform([class_idx])[0]))
            plt.axis('off')
        plt.show()
        
        return
    
    
    def GetModel(self):

        model = models.Sequential()
        model.add(layers.Conv2D(32, (5,5), activation='relu', input_shape=(self.size[0],self.size[1],1)))
        model.add(layers.MaxPooling2D(pool_size=(4, 4)))
        model.add(layers.Conv2D(64, (5,5), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=((4, 4))))
        model.add(layers.Conv2D(128, (5,5), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=((4, 4))))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(len(self.label_list), activation='softmax')) 

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model
        
        print("Model built.")
        
        return
    
    
    def TrainModel(self, ckpt, epochs=20, batch_size=64):
        
        print("Training model...")
    
        tf.random.set_seed(1)
        random.seed(1)
        
        if self.x_val is not None and self.y_val_enc is not None:
            self.history = self.model.fit(self.x_train, self.y_train_enc, validation_data=(self.x_val, self.y_val_enc)
                                          ,epochs=epochs, batch_size=batch_size, callbacks=Callbacks(ckpt))
        else:
            self.history = self.model.fit(self.x_train, self.y_train_enc
                                          ,epochs=epochs, batch_size=batch_size, callbacks=Callbacks(ckpt))

        
        return 
    
    
    def LoadModel(self, ckpt, eval=True):
        
        print("Loading model...")
    
        self.model.load_weights(ckpt).expect_partial()
        
        if eval:

            print("Train data:")
            self.model.evaluate(self.x_train, self.y_train_enc, verbose=2)
             
            if self.has_test:
                print("Test data:")
                self.model.evaluate(self.x_test, self.y_test_enc, verbose=2)    
            elif self.test_ratio>0:
                print("Validation data:")
                self.model.evaluate(self.x_val, self.y_val_enc, verbose=2)   
        

        return
    
    
    def RetrainModel(self, ckpt, ckpt_new=None, epochs=20, batch_size=64):
        
        print("Retraining model...")
    
        tf.random.set_seed(1)
        random.seed(1)
    
        self.model.load_weights(ckpt).expect_partial()
        
        if ckpt_new is None:
            ckpt_new = ckpt
        
        if self.x_val is not None and self.y_val_enc is not None:
            self.history = self.model.fit(self.x_train, self.y_train_enc, validation_data=(self.x_val, self.y_val_enc)
                                          ,epochs=epochs, batch_size=batch_size, callbacks=Callbacks(ckpt_new))
        else:
            self.history = self.model.fit(self.x_train, self.y_train_enc
                                          ,epochs=epochs, batch_size=batch_size, callbacks=Callbacks(ckpt_new))
            

        return 
    
    
    def ShowHistory(self):
        
        history_df = pd.DataFrame(self.history.history)
        history_df['epoch'] = self.history.epoch

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        history_df.plot(x='epoch', y=['loss', 'val_loss'], ax=ax[0])
        history_df.plot(x='epoch', y=['accuracy', 'val_accuracy'], ax=ax[1])

        return
    
    
    def Evaluate(self,mode=2,plot=True):
        ## mode=0: train data
        ## mode=1: train data & val data
        ## mode=2: train data & val data & test data
        
        print("*"*60)
        
        print("Evaluating model...")
        print("Label mapping:")
        for i in range(len(set(self.labels))):
            print(i,self.le.inverse_transform([i])[0])
            
        print("*"*60)    
        
        
        self.y_train_pred = np.argmax(self.model.predict(self.x_train),1)

        if plot:
            print("Train acc:", accuracy_score(self.y_train_int,self.y_train_pred))
            sns.heatmap(confusion_matrix(self.y_train_int,self.y_train_pred),annot=True,cmap="YlGnBu")
            plt.show()
        
        self.precision_train = (np.diagonal(confusion_matrix(self.y_train_int,self.y_train_pred))/np.sum(confusion_matrix(self.y_train_int,self.y_train_pred),0))

        if plot:
            sns.barplot(x=np.arange(len(set(self.labels))),y=self.precision_train)
            plt.show()
            print("*"*60)
        
        
        if self.has_test and mode==2:
            
            self.y_test_pred = np.argmax(self.model.predict(self.x_test),1)
            
            if plot:
                print("Test acc:", accuracy_score(self.y_test_int,self.y_test_pred))
                sns.heatmap(confusion_matrix(self.y_test_int,self.y_test_pred),annot=True,cmap="YlGnBu")
                plt.show()

            self.precision_test = (np.diagonal(confusion_matrix(self.y_test_int,self.y_test_pred))/np.sum(confusion_matrix(self.y_test_int,self.y_test_pred),0))

            if plot:
                sns.barplot(x=np.arange(len(set(self.labels))),y=self.precision_test)
                plt.show()
                print("*"*60) 
            
        elif self.test_ratio>0 and mode>=1:
            
            self.y_val_pred = np.argmax(self.model.predict(self.x_val),1)
            
            if plot:
                print("Validation acc:", accuracy_score(self.y_val_int,self.y_val_pred))
                sns.heatmap(confusion_matrix(self.y_val_int,self.y_val_pred),annot=True,cmap="YlGnBu")
                plt.show()

            self.precision_val = (np.diagonal(confusion_matrix(self.y_val_int,self.y_val_pred))/np.sum(confusion_matrix(self.y_val_int,self.y_val_pred),0))

            if plot:
                sns.barplot(x=np.arange(len(set(self.labels))),y=self.precision_val)
                plt.show()
                print("*"*60)
        
        return
    
    
    def CombineModels(self,model_dict,mode="test"):
    
        if (mode is None or mode=="test") and self.has_test:
            
            y_test_pred_new = self.y_test_pred.copy()

            for idx, mdl in model_dict.items():
                y_test_pred2 = np.argmax(mdl.model.predict(self.x_test[self.y_test_pred==idx]),1)
                labels_repred = mdl.le.inverse_transform(y_test_pred2)
                y_test_pred_modified = self.le.transform(labels_repred)
                y_test_pred_new[self.y_test_pred==idx] = y_test_pred_modified

            self.y_test_pred_new = y_test_pred_new
            

        
        elif (mode is None or mode=="val") and self.test_ratio>0:
            
            y_val_pred_new = self.y_val_pred.copy()

            for idx, mdl in model_dict.items():
                y_val_pred2 = np.argmax(mdl.model.predict(self.x_val[self.y_val_pred==idx]),1)
                labels_repred = mdl.le.inverse_transform(y_val_pred2)
                y_val_pred_modified = self.le.transform(labels_repred)
                y_val_pred_new[self.y_val_pred==idx] = y_val_pred_modified

            self.y_val_pred_new = y_val_pred_new
            

            
        elif mode is None or mode=="train":
            
            y_train_pred_new = self.y_train_pred.copy()

            for idx, mdl in model_dict.items():
                y_train_pred2 = np.argmax(mdl.model.predict(self.x_train[self.y_train_pred==idx]),1)
                labels_repred = mdl.le.inverse_transform(y_train_pred2)
                y_train_pred_modified = self.le.transform(labels_repred)
                y_train_pred_new[self.y_train_pred==idx] = y_train_pred_modified

            self.y_train_pred_new = y_train_pred_new
            
            
        print("Classification updated.")
        print("*"*60)
        
        return 
    
    def Compare(self, mode="test"):
        
        if (mode is None or mode=="test") and self.has_test:
        
            print("Accuracy for self test data: ",accuracy_score(self.y_test_int,self.y_test_pred),"->",accuracy_score(self.y_test_int,self.y_test_pred_new))


            fig, ax = plt.subplots(1,2,figsize=(12,4))
            sns.heatmap(confusion_matrix(self.y_test_int,self.y_test_pred),annot=True,cmap="YlGnBu", ax=ax[0])
            sns.heatmap(confusion_matrix(self.y_test_int,self.y_test_pred_new),annot=True,cmap="YlGnBu", ax=ax[1])
            ax[0].set_title('Before re-classification')
            ax[1].set_title('After re-classification')
            plt.show()

            fig, ax = plt.subplots(1,2,figsize=(12,4))
            self.precision_test_new = (np.diagonal(confusion_matrix(self.y_test_int,self.y_test_pred_new))/np.sum(confusion_matrix(self.y_test_int,self.y_test_pred_new),0))
            sns.barplot(x=np.arange(len(set(self.labels))),y=self.precision_test, ax=ax[0])
            sns.barplot(x=np.arange(len(set(self.labels))),y=self.precision_test_new, ax=ax[1])
            ax[0].set_title('Before re-classification')
            ax[1].set_title('After re-classification')
            plt.show()
            
        elif (mode is None or mode=="val") and self.test_ratio>0:
            
            print("Accuracy for validation data: ",accuracy_score(self.y_val_int,self.y_val_pred),"->",accuracy_score(self.y_val_int,self.y_val_pred_new))


            fig, ax = plt.subplots(1,2,figsize=(12,4))
            sns.heatmap(confusion_matrix(self.y_val_int,self.y_val_pred),annot=True,cmap="YlGnBu", ax=ax[0])
            sns.heatmap(confusion_matrix(self.y_val_int,self.y_val_pred_new),annot=True,cmap="YlGnBu", ax=ax[1])
            ax[0].set_title('Before re-classification')
            ax[1].set_title('After re-classification')
            plt.show()

            fig, ax = plt.subplots(1,2,figsize=(12,4))
            self.precision_val_new = (np.diagonal(confusion_matrix(self.y_val_int,self.y_val_pred_new))/np.sum(confusion_matrix(self.y_val_int,self.y_val_pred_new),0))
            sns.barplot(x=np.arange(len(set(self.labels))),y=self.precision_val, ax=ax[0])
            sns.barplot(x=np.arange(len(set(self.labels))),y=self.precision_val_new, ax=ax[1])
            ax[0].set_title('Before re-classification')
            ax[1].set_title('After re-classification')
            plt.show()
            
        elif mode is None or mode=="train":
            
            print("Accuracy for real test data: ",accuracy_score(self.y_train_int,self.y_train_pred),"->",accuracy_score(self.y_train_int,self.y_train_pred_new))


            fig, ax = plt.subplots(1,2,figsize=(12,4))
            sns.heatmap(confusion_matrix(self.y_train_int,self.y_train_pred),annot=True,cmap="YlGnBu", ax=ax[0])
            sns.heatmap(confusion_matrix(self.y_train_int,self.y_train_pred_new),annot=True,cmap="YlGnBu", ax=ax[1])
            ax[0].set_title('Before re-classification')
            ax[1].set_title('After re-classification')
            plt.show()

            fig, ax = plt.subplots(1,2,figsize=(12,4))
            self.precision_train_new = (np.diagonal(confusion_matrix(self.y_train_int,self.y_train_pred_new))/np.sum(confusion_matrix(self.y_train_int,self.y_train_pred_new),0))
            sns.barplot(x=np.arange(len(set(self.labels))),y=self.precision_train, ax=ax[0])
            sns.barplot(x=np.arange(len(set(self.labels))),y=self.precision_train_new, ax=ax[1])
            ax[0].set_title('Before re-classification')
            ax[1].set_title('After re-classification')
            plt.show()
            
        print("*"*60)
        
        return
        


# # Train

# In[ ]:


def train(train_data_dir="./train",trained_cnn_dir="./",mode=1):
    
    ## mode=0: train new model
    ## mode=1: load saved model
    
    model_code = [-1,5,6,9,12,13]
    
    model_label = [[],
                   ['Office','livingroom'],
                   ['Coast','Highway','Mountain','OpenCountry'],
                   ['Mountain','TallBuilding','industrial'],
                   ['Office','bedroom','kitchen','livingroom','store'],
                   ['Office','bedroom','kitchen','livingroom']
                  ]  
    
    model_ckpt = ["trained_cnn.ckpt",
                  "trained_cnn_5.ckpt",
                  "trained_cnn_6.ckpt",
                  "trained_cnn_9.ckpt",
                  "trained_cnn_12.ckpt",
                  "trained_cnn_13.ckpt"
                 ]
    
    model_list = []
    
    for i,m in enumerate(model_code):
        if m == -1:
            model = Model(train_data_dir)
        else:
            model = Model(train_data_dir,model_label[i])
        model.LoadImages()
        if m == -1:
            model.PreprocessImages(5)
        else:
            model.PreprocessImages(5)
        model.GetModel()
        if mode==0:
            model.TrainModel(trained_cnn_dir+model_ckpt[i])
        elif mode==1:
            model.LoadModel(trained_cnn_dir+model_ckpt[i])
            
        model_list.append(model)
    
    models = dict(zip(model_code,model_list))
    
    models[-1].Evaluate(plot=False)
    models[-1].CombineModels({13:models[13],6:models[6],5:models[5],12:models[12],0:models[6],9:models[9]},"train")

    global train_models
    train_models = models

    return accuracy_score(models[-1].y_train_int,models[-1].y_train_pred_new)


# # Test

# In[ ]:


def test(test_data_dir="./train",trained_cnn_dir="./"):
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    model_code = [-1,5,6,9,12,13]
    
    model_label = [[],
                   ['Office','livingroom'],
                   ['Coast','Highway','Mountain','OpenCountry'],
                   ['Mountain','TallBuilding','industrial'],
                   ['Office','bedroom','kitchen','livingroom','store'],
                   ['Office','bedroom','kitchen','livingroom']
                  ]  
    
    model_ckpt = ["trained_cnn.ckpt",
                  "trained_cnn_5.ckpt",
                  "trained_cnn_6.ckpt",
                  "trained_cnn_9.ckpt",
                  "trained_cnn_12.ckpt",
                  "trained_cnn_13.ckpt"
                 ]
    
    model_list = []
    
    for i,m in enumerate(model_code):
        if m == -1:
            model = Model(test_data_dir)
            model.test_ratio=0
            model.LoadImages()
            model.PreprocessImages()
            model.GetModel()
            model.LoadModel(trained_cnn_dir+model_ckpt[i],eval=False)
        else:
            model = Model(test_data_dir,model_label[i])
            model.test_ratio=0
            model.GetModel()
            model.LoadModel(trained_cnn_dir+model_ckpt[i],eval=False)
            
        model_list.append(model)
    
    models = dict(zip(model_code,model_list))
    
    models[-1].Evaluate(plot=False)
    models[-1].CombineModels({13:models[13],6:models[6],5:models[5],12:models[12],0:models[6],9:models[9]},"train")

    global test_models
    test_models = models
    
    return accuracy_score(models[-1].y_train_int,models[-1].y_train_pred_new)


# # Main

# In[ ]:


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train','test'])
    parser.add_argument('--train_data_dir', default='./train', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./train', help='the directory of testing data')
    parser.add_argument('--model_dir', default='./', help='the pre-trained model')
    opt = parser.parse_args()


    if opt.phase == 'train':
        training_accuracy = train(opt.train_data_dir, opt.model_dir, mode=0)
        print(training_accuracy)

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.model_dir)
        print(testing_accuracy)


# # Code demonstration

# * Set demo=True to show the details.
# * Change the directory of train data.

# In[ ]:


demo = False
train_data_dir = './train'


# In[ ]:


if demo:
    model_ = Model(train_data_dir)
    model_.has_test = True
    model_.test_ratio = 0.2


# ## Image loading

# In[ ]:


if demo:
    model_.LoadImages()


# In[ ]:


if demo:
    model_.data.shape


# In[ ]:


if demo:
    model_.labels.shape


# In[ ]:


if demo:
    plt.imshow(model_.data[0],'gray')


# ## Image preprocessing

# In[ ]:


if demo:
    model_.PreprocessImages()


# In[ ]:


# image count of classes in training data
if demo:
    [model_.y_train_int.tolist().count(i) for i in range(len(set(model_.labels)))]


# In[ ]:


# image count of classes in validation data
if demo:
    [model_.y_val_int.tolist().count(i) for i in range(len(set(model_.labels)))]


# In[ ]:


# image count of classes in test data
if demo:
    [model_.y_test_int.tolist().count(i) for i in range(len(set(model_.labels)))]


# ## Noise adding (optional)

# In[ ]:


if demo:
    model__ = copy.deepcopy(model_)
    model__.AddNoise(0.5)


# ## Preprocessed images

# In[ ]:


if demo:
    model_.ShowImages()


# ## Model building

# In[ ]:


if demo:
    model_.GetModel()
    model_.model.summary()


# ## Model training

# ### Core model

# > Training a new model

# In[ ]:


if demo:
    model_.TrainModel(None,epochs=2)
    model_.ShowHistory()


# > Loading a saved weight

# In[ ]:


if demo:
    model_.LoadModel('./trained_cnn_demo.ckpt')


# > Retraining by initializing a saved weight

# In[ ]:


if demo:
    model_.RetrainModel('./trained_cnn_demo.ckpt','./trained_cnn_demo_retrained.ckpt',epochs=2)
    model_.ShowHistory()


# ### Supplementary model for ['Office','bedroom','kitchen','livingroom']

# In[ ]:


if demo:
    model_s = Model(train_data_dir,['Office','bedroom','kitchen','livingroom'])
    model_s.has_test=True
    model_s.test_ratio=0.2
    model_s.LoadImages()
    model_s.PreprocessImages(5)
    model_s.GetModel()


# In[ ]:


if demo:
    model_s.LoadModel('./trained_cnn_13.ckpt')


# ## Performance Evaluation

# ### Core model

# In[ ]:


if demo:
    model_.Evaluate()


# ### Supplementary models

# In[ ]:


if demo:
    model_s.Evaluate()


# ## Model combination

# > Use a supplementary model to improve precision of output=13 ('livingroom')

# In[ ]:


if demo:
    model_.CombineModels({13:model_s})
    model_.Compare()


# In[ ]:




