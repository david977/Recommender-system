import numpy as np
import tensorflow as tf

#Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

#Store the MNIST data in /tmp/data
mnist=input_data.read_data_sets("mnist_data/",one_hot=True)

training_digits,training_labels=mnist.train.next_batch(5000)
test_digits,test_labels= mnist.test.next_batch(200)
#We dont know how many images will be there for training so used None keyword.
training_digits_pl=tf.placeholder("float",[None,784])
#
test_digit_pl=tf.placeholder("float",[784])

#Nearest Neighbor calculation using L1 distance
l1_distance=tf.abs(tf.add(training_digits_pl,tf.negative(test_digit_pl)))
#to give the minimum distance of training digits from test digits
distance=tf.reduce_sum(l1_distance,axis=1)

#Prediction: Get min distance index (Nearest neighbor). the minimum value will be at index 0
pred=tf.arg_min(distance,0)

accuracy=0.

#initializing the variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    #loop over test data
    for i in range(len(test_digits)):
        #Get the nearest neighbor
        nn_index=sess.run(pred,feed_dict={training_digits_pl: training_digits,test_digit_pl:test_digits[i,:]})
        print("Test",i,"Prediction:",np.argmax(training_labels[nn_index]),"True Label:",np.argmax(test_labels[i]))

        #Calculate accuracy
        if np.argmax(training_labels[nn_index])==np.argmax(test_labels[i]): accuracy +=1./len(test_digits)

    print("Done!")
    print("Accuracy:",accuracy)

"""Placeholder give the values (or hold a place for a tensor) to the tensors at the runtime (and become an input node). This is done through feed_dict{}."""
"""Here we have pred as 'fetches' i.e. the node to be computed"""
"""Variables are the mutable Tensor values that persist across multiple calls to run Session.run()"""
"""We cant use variables in our program unless we initialize the global_variables_initializer()"""
"""training data is passed in batches through the model several times"""
