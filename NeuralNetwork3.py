# Name: Steve Regala
# CSCI 561 Homework 3: Neural Network
# Date Due: 11/28/2022

import numpy as np
import random
import csv
import math
import sys

# Convert CSV files into arrays
def convert_CSV(train_data, train_label, test_data):
   # training data
   with open(train_data) as train_data_object:
      final_train_data = np.loadtxt(train_data_object, delimiter=",", dtype=np.float64)
   
   # training label
   with open(train_label) as train_label_object:
      final_train_label = np.loadtxt(train_label_object, dtype=np.int32)

   # testing data
   with open(test_data) as test_data_object:
      final_test_data = np.loadtxt(test_data_object, delimiter=",", dtype=np.float64)

   return final_train_data, final_train_label, final_test_data


# Softmax function for the output layer
# CITATION: https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def act_softmax(val):
   e_x = np.exp(val)
   return e_x / np.sum(e_x, axis=0)


# Assign initial weights
def assign_weights():
   # CITATION: to initialize weights, the following link is used:
   # https://www.analyticsvidhya.com/blog/2021/05/how-to-initialize-weights-in-neural-networks/

   final_weights = {"Weight_1": np.random.randn(hidd_ONE_layer, input_layer) * np.sqrt(1.0/input_layer), # 6x2
                    "Bias_1": np.random.randn(hidd_ONE_layer, 1) * np.sqrt(1.0/input_layer),    # 6x1
                    #"Bias_1": np.zeros((hidd_ONE_layer, 1)) * np.sqrt(1/input_layer),

                    "Weight_2": np.random.randn(output_layer, hidd_ONE_layer) * np.sqrt(1.0/hidd_ONE_layer), # 2x6
                    "Bias_2": np.random.randn(output_layer, 1) * np.sqrt(1.0/hidd_ONE_layer)    # 2x1
                    #"Bias_2": np.zeros((output_layer, 1)) * np.sqrt(1.0/hidd_ONE_layer)
                    }

   return final_weights


# Sigmoid hidden layer activation function: 
# takes any real value as input and outputs values in the range 0 to 1
def act_sigmoid(val):
   return 1/(1 + np.exp(-val))


# Feed forward propagation
def feed_forward(data):
   history = dict()

   # 1 Hidden Layer
   history['pass_1'] = np.dot(weights['Weight_1'], data) + weights['Bias_1'] # 6x40
   history['alpha_1'] = act_sigmoid(history['pass_1'])   # 6x40 ORIGINAL
   #history['alpha_1'] = np.tanh(history['pass_1'])

   history['pass_2'] = np.dot(weights['Weight_2'], history['alpha_1']) + weights['Bias_2'] # 2x40
   history['alpha_2'] = act_softmax(history['pass_2'])

   return history


def back_propagation(data_train, label_train):
   hist = feed_forward(data_train)

   temp_b = data_train.shape[1]

   # Last layer error
   delta_last = hist['alpha_2'] - label_train

   # 2nd WEIGHT Layer
   dW2 = np.dot(delta_last, hist['alpha_1'].T) / temp_b # New Edit
   dB2 = np.sum(delta_last, axis=1, keepdims=True) / temp_b # New Edit

   # HIDDEN Layer
   delta_alpha_1 = np.dot(weights['Weight_2'].T, delta_last)
   delta_pass_1 = delta_alpha_1 * act_sigmoid(hist['pass_1']) * (1-act_sigmoid(hist['pass_1']))
   #delta_pass_1 = delta_alpha_1 * (1 - np.power(hist['alpha_1'], 2)) # tan h activation function

   # 1st WEIGHT Layer
   dW1 = np.dot(delta_pass_1, data_train.T) / temp_b # New Edit
   dB1 = np.sum(delta_pass_1, axis=1, keepdims=True) / temp_b # New Edit

   # regularization
   dW2 += regular * weights['Weight_2']
   dW1 += regular * weights['Weight_1']

   grad = {'weight_1':dW1, 'bias_1':dB1,
           'weight_2':dW2, 'bias_2':dB2}

   return grad


# Create batches to train network
def get_batches(data_train, label_train):
   batch_list = list()
   training_length = data_train.shape[0]
   num_batches = math.floor(training_length/batch_size)
   for i in range(num_batches):
      start = i * batch_size
      end = (i+1) * batch_size
      train_data_permutation = data_train[start:end, :]
      train_label_permutation = label_train[start:end, :]
      temp_batch = (train_data_permutation, train_label_permutation)
      batch_list.append(temp_batch)

   if training_length%batch_size != 0:
      train_data_permutation = data_train[batch_size*math.floor(training_length/batch_size):training_length , :]
      train_label_permutation = label_train[batch_size*math.floor(training_length/batch_size):training_length  , :]
      temp_batch = (train_data_permutation, train_label_permutation)
      batch_list.append(temp_batch)

   return batch_list


# Cross entropy loss function to measure loss
# CITATION: https://vitalflux.com/cross-entropy-loss-explained-with-python-examples/
def cross_entropy(yHat, y):
   cost = (1/y.shape[0]) * np.sum(np.multiply(y, np.log(yHat)) + np.multiply(1-y, np.log(1-yHat)))
   return -cost


# Normalize the data, but found it unnecessary
def normalize(data):
   return ((data - np.min(data)) / (np.max(data) - np.min(data)))


# Used for improving neural network for each epoch
def accuracy(X, y):
   predictions = []
   acc_hist = feed_forward(X)
   res_val = acc_hist['alpha_2']
   pred = np.argmax(res_val, axis=0)
   predictions.append(pred == np.argmax(y, axis=0))

   return np.mean(predictions)



# ----------------- BEGIN RUNNING PROGRAM HERE ----------------- #

# Read in command-line arguments
cla = sys.argv
# 2001x2 || 1x2001 || 500x2
training_data_OG, training_label_OG, testing_data_OG = convert_CSV(cla[1], cla[2], cla[3])

# Concatenate X_1^2 and X_2^2 for TRAINING data
training_data_OG = np.array(training_data_OG, ndmin=2)       # 2001x2
training_data = np.concatenate((training_data_OG, training_data_OG**2), axis=1)  # 2001x4

''' NOT NEEDED CODE''' 
# X * Y
#x_y = np.array(np.multiply(np.array(training_data_OG[:,0],ndmin=2), np.array(training_data_OG[:,1], ndmin=2)), ndmin=2)
#print(x_y.T.shape)
#print(x_y.T)
#training_data = np.concatenate((training_data, x_y.T), axis=1)
''''''


# Concatenate X_1^2 and X_2^2 for TESTING data
testing_data_OG = np.array(testing_data_OG, ndmin=2)         # 500x4
testing_data = np.concatenate((testing_data_OG, testing_data_OG**2), axis=1)
#x_y_test = np.array(np.multiply(np.array(testing_data_OG[:,0],ndmin=2), np.array(testing_data_OG[:,1], ndmin=2)), ndmin=2)
#testing_data = np.concatenate((testing_data, x_y_test.T), axis=1)


# one hot encoding for training label
training_label_OG = np.array(training_label_OG, ndmin=2)     # 1x2001
one_hot = np.zeros((training_label_OG.size, training_label_OG.max()+1))
one_hot[np.arange(training_label_OG.size), training_label_OG] = 1
training_label = one_hot   # 2000x2

# normalize and scale training data
#training_data = normalize(training_data)

input_layer = 4      # coordinates
output_layer = 2     # 2 because we're using softmax at the end
hidd_ONE_layer = 10   # number between number of input and output layer nodes
epochs = 2000

regular = 0.001
learning_rate = 0.1  # this could change depending on what suits my optimizer the best
batch_size = 32

weights = assign_weights()

# Train
for i in range(epochs):

   #print("EPOCH NUMBER: " + str(i))

   #permutation = np.random.permutation(training_data.shape[0])
   rand = np.arange(len(training_data))
   np.random.shuffle(rand)
   train_data_shuff = training_data[rand,:]
   train_label_shuff = training_label[rand,:]

   mini_batches = get_batches(train_data_shuff, train_label_shuff)   # batch size of 40, 25 batches total for 1000 samples
   for batch in mini_batches:
      train, label = batch
      gradients = back_propagation(train.T, label.T)

      weights['Weight_1'] = weights['Weight_1'] - (learning_rate*gradients['weight_1'])
      weights['Bias_1'] = weights['Bias_1'] - (learning_rate*gradients['bias_1'])

      weights['Weight_2'] = weights['Weight_2'] - (learning_rate*gradients['weight_2'])
      weights['Bias_2'] = weights['Bias_2'] - (learning_rate*gradients['bias_2'])

   # calculate training accuracy for each epoch
   training_accuracy = accuracy(training_data.T, training_label.T)
   hist_loss = feed_forward(training_data.T)
   result_loss = hist_loss['alpha_2']
   amax_loss = np.array(np.amax(result_loss, axis=0), ndmin=2).T
   loss = cross_entropy(result_loss.T, training_label)
   print("\n Train accuracy for EPOCH # " + str(i+1) + " : " + str(training_accuracy))
   print("Loss at EPOCH # " + str(i+1) + " : " + str(loss))


final_hist = feed_forward(testing_data.T)
result = final_hist['alpha_2']
prediction = result.argmax(axis=0)
prediction.tofile('test_predictions.csv', sep='\n')