
# coding: utf-8

# **End Semester Project work:
# <br/>Dublin City University
# <br/>School of Computing
# <br/>CA684 : Machine Learning
# <br/>Juhi Shrivastava
# <br/>16212548**

# Dataset : Dog Vs Cat Competetion from the Kaggle Competetion
# 
# Importing the necessary packages

# In[5]:

#Importing necessary packages
import tflearn 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL 
from PIL import Image
get_ipython().magic('matplotlib inline')

#For writing text file
import glob
import os
import random

#Reading image from a textfile
from tflearn.data_utils import image_preloader
import math


# **Data Processing**
# 
# Creating text files (declaring the name of the file just) as String constants containing the directory path which will be used later
# 

# In[6]:

Training_data = 'C:/Users/Lenovo/Desktop/sem2/ca684_mc_learning/Project/Training_data'
Train = 'C:/Users/Lenovo/Desktop/sem2/ca684_mc_learning/Project/train.txt'
Test = 'C:/Users/Lenovo/Desktop/sem2/ca684_mc_learning/Project//test.txt'
Validation = 'C:/Users/Lenovo/Desktop/sem2/ca684_mc_learning/Project//validation.txt'
train_prop = 0.7
test_prop = 0.2
validation_prop = 0.1


# Read the image directories and shuffle the data otherwise the model will be fed with a single class data for a long time and network will not learn properly.
# OS.listdir and endswith( ) This short script uses the os.listdir function (that belongs to the OS module) to search through a given path (".") for all files that endswith ".txt". When the for loop finds a match it adds it to the list "newlist" by using the append function.

# In[7]:

Images = os.listdir(Training_data)
random.shuffle(Images)


# In[8]:

#total number of images
No_of_Images = len(Images)
No_of_Images


# When opening a file, it’s preferable to use open() instead of invoking the file constructor directly.
# 
# The first two arguments are : name is the file name to be opened, and mode is a string indicating how the file is to be opened.
# 
# The most commonly-used values of mode are 'r' for reading, 'w' for writing (truncating the file if it already exists), and 'a' for appending. If mode is omitted, it defaults to 'r'.
# 
# math.ceil : The smallest integer greater than or equal to the given number.

# In[9]:

#Get the training data - 70%
files = open(Train, 'w')
train_files = Images[0: int(train_prop*No_of_Images)]
for filename in train_files:
    if filename[0:3] == 'cat':
        files.write(Training_data + '/'+ filename + ' 0\n')
    elif filename[0:3] == 'dog':
        files.write(Training_data + '/'+ filename + ' 1\n')

files.close()

#Get the testing data - 20%
files = open(Test, 'w')
test_files = Images[int(math.ceil(train_prop*No_of_Images)):int(math.ceil((train_prop+test_prop)*No_of_Images))]
for filename in test_files:
    if filename[0:3] == 'cat':
        files.write(Training_data + '/'+ filename + ' 0\n')
    elif filename[0:3] == 'dog':
        files.write(Training_data + '/'+ filename + ' 1\n')
files.close()

#Get the validation data - 10%
files = open(Validation, 'w')
valid_files = Images[int(math.ceil((train_prop+test_prop)*No_of_Images)):No_of_Images]
for filename in valid_files:
    if filename[0:3] == 'cat':
        files.write(Training_data + '/'+ filename + ' 0\n')
    elif filename[0:3] == 'dog':
        files.write(Training_data + '/'+ filename + ' 1\n')
files.close()


# <br/>Images on the fly
# <br/>Reduced dimension to 56 by 56 otherwise alot of RAM will be used. There are three channels(RGB) as can be seen in the shape of the image.
# 
# Importing the data: Train, Test, Validation
# 
# Load images and labels 
# 
# Categorical_labels : Here the labels will be converted into binary vectors
# 
# Cats = [0,1]
# Dogs = [1,0]

# In[10]:

X_train, Y_train = image_preloader(Train, image_shape=(56,56),mode='file', categorical_labels=True,normalize=True)
X_test, Y_test = image_preloader(Test, image_shape=(56,56),mode='file', categorical_labels=True,normalize=True)
X_val, Y_val = image_preloader(Validation, image_shape=(56,56),mode='file', categorical_labels=True,normalize=True)


# In[11]:

print("Dataset")
print("Number of training images {}".format(len(X_train)))
print("Number of testing images {}".format(len(X_test)))
print("Number of validation images {}".format(len(X_val)))
print("Shape of an image {}" .format(X_train[1].shape))
print("Shape of label:{} ,number of classes: {}".format(Y_train[1].shape,len(Y_train[1])))


# In[35]:

#Sample Image 
plt.imshow(X_train[1])
plt.axis('off')
plt.title('Sample image {}'.format(Y_train[1]))
plt.show()


# **Model:**
# <br/> The Model Used is the Convolutional Neural Network on the Tensorflow GPU (CUDA)

# First feed the data into the graphs by creating the placeholders for the input images and the binary class in which the data is to be clasified

# In[18]:

x = tf.placeholder(tf.float32,shape=[None,56,56,3] , name='input_image') 
#input class
y_ = tf.placeholder(tf.float32,shape=[None, 2] , name='input_class')


# **Network Architecture**
# <br/>Modified Version of Alexnet i.e the architecture used for the prediction contains:
# <br/>3 Convolutional Layers
# <br/>2 Fully connected layer
# <br/>1 Softmax output layer

# In[20]:

input_layer = x
#convolutional layer 1 --convolution+RELU activation
conv_layer1=tflearn.layers.conv.conv_2d(input_layer, nb_filter=64, filter_size=5, strides=[1,1,1,1],
                                        padding='same', activation='relu', regularizer="L2", name='conv_layer_1')

#2x2 max pooling layer
out_layer1=tflearn.layers.conv.max_pool_2d(conv_layer1, 2)


#second convolutional layer 
conv_layer2=tflearn.layers.conv.conv_2d(out_layer1, nb_filter=128, filter_size=5, strides=[1,1,1,1],
                                        padding='same', activation='relu',  regularizer="L2", name='conv_layer_2')
out_layer2=tflearn.layers.conv.max_pool_2d(conv_layer2, 2)
# third convolutional layer
conv_layer3=tflearn.layers.conv.conv_2d(out_layer2, nb_filter=128, filter_size=5, strides=[1,1,1,1],
                                        padding='same', activation='relu',  regularizer="L2", name='conv_layer_2')
out_layer3=tflearn.layers.conv.max_pool_2d(conv_layer3, 2)

#fully connected layer1
fcl= tflearn.layers.core.fully_connected(out_layer3, 4096, activation='relu' , name='FCL-1')
fcl_dropout_1 = tflearn.layers.core.dropout(fcl, 0.8)
#fully connected layer2
fc2= tflearn.layers.core.fully_connected(fcl_dropout_1, 4096, activation='relu' , name='FCL-2')
fcl_dropout_2 = tflearn.layers.core.dropout(fc2, 0.8)
#softmax layer output
y_predicted = tflearn.layers.core.fully_connected(fcl_dropout_2, 2, activation='softmax', name='output')


# In[21]:

#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_predicted+np.exp(-10)), reduction_indices=[1]))

#optimiser -
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#calculating accuracy of our model 
correct_prediction = tf.equal(tf.argmax(y_predicted,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[23]:

# session parameters
sess = tf.InteractiveSession()

#initialising variables
init = tf.initialize_all_variables()

#Saving the session
sess.run(init)
saver = tf.train.Saver()
save_path="C:/Users/Lenovo/Desktop/sem2/ca684_mc_learning/Project//mark2.ckpt"


# In[36]:

#Display all the operations of the graph
graph = tf.get_default_graph()

# every operations in our graph
[op.name for op in graph.get_operations()]


# The testing of the model is done on the Test data setting the Epochs and the batch size according to the Hardware of the Computer.

# In[25]:

epoch=5000
#change batch size according to your hardware's power. For GPU's use batch size in powers of 2 like 2,4,8,16...
batch_size=20 
previous_batch=0


# In the below code print the accuracy after every 500 iterations and the Loss after every 100th iteration.

# In[26]:

for i in range(epoch): 
#The Testing data is processed Batch wise and the Optimization is done for the Images in the Testing data
#X_train -- total number of training images    
    if previous_batch >= len(X_train) : 
        previous_batch=0    
    current_batch=previous_batch+batch_size
    x_input=X_train[previous_batch:current_batch]
    
#Reshape the training data images    
    x_images=np.reshape(x_input,[batch_size,56,56,3])
    y_input=Y_train[previous_batch:current_batch]
    
#Reshape the Labels into 2 classes    
    y_label=np.reshape(y_input,[batch_size,2])
    previous_batch=previous_batch+batch_size
    
#Calculate the Loss    
    _,loss=sess.run([train_step, cross_entropy], 
                    feed_dict={x: x_images,y_: y_label}) 
    
    if i%500==0:
#number of test samples        
        n=50 
        x_test_images=np.reshape(X_test[0:n],[n,56,56,3])
        y_test_labels=np.reshape(Y_test[0:n],[n,2])
        Accuracy=sess.run(accuracy,
                           feed_dict={
                        x: x_test_images ,
                        y_: y_test_labels
                      })
#Print the Accuracy after every 500th Iteration and Loss after 100th Iteration to check if the accuracy is increasing 
#and Overfitting is happening        
        print("Iteration no :{} , Accuracy:{} , Loss : {}" .format(i,Accuracy,loss))
        saver.save(sess, save_path, global_step = i)
    elif i % 100 ==0:   
        print("Iteration no :{} Loss :bina {}" .format(i,loss))


# Now once the testing is done. Validate our model with the Validation Data with 5000 images.

# In[ ]:

#Test the Model with the validation data
x_input  = X_val
x_images = np.reshape(x_input,[len(X_val), 56,56,3])
y_input  = Y_val
y_label  = np.reshape(y_input,[len(Y_val),2])

Acc_validation=sess.run(accuracy,
                            feed_dict={
                        x: x_images,
                        y_: y_label
    })


# In[ ]:

Acc_validation=round(Acc_validation*100,2)
print("Accuracy in the validation dataset: {} %" .format(Acc_validation))


# In[30]:

#Define the function to test the Images with the Test data
def process_img(imgage):
        imgage=imgage.resize((56, 56), Image.ANTILALAS) #resize the image
        imgage = np.array(imgage)
        imgage=img/np.max(imgage).astype(float) 
        imgage=np.reshape(imgage, [1,56,56,3])
        return imgage


# In[ ]:

#Test the model with the Test Data 
test_image=Image.open('C:/Users/Lenovo/Desktop/sem2/ca684_mc_learning/Project/Test_Data')
test_image= process_img(test_image)
predicted_array= sess.run(y_predicted, feed_dict={x: test_image})
predicted_class= np.argmax(predicted_array)
if predicted_class==0:
    print("It is a cat")
else :
    print("It is a dog")

