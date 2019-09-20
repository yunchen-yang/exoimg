import numpy as np
import cv2
import os
import sys
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

def mb_index(sample_n, img_n):
    index_matrix = np.zeros((sample_n, img_n))
    for i in range(sample_n):
        index_tem = np.arange(img_n)
        np.random.shuffle(index_tem)
        index_matrix[i, :] = index_tem
    return index_matrix

def img2mat(folder_path, label, file_index, dimension=512, channel=1):
    img_list = os.listdir(folder_path)
    ind_len = file_index.shape[1]
    if channel==1:
        X = np.zeros((ind_len, dimension, dimension, 1))
        for i in range(ind_len):
            X[i, :, :, 0] = cv2.imread(folder_path + "/" + img_list[file_index[0, i]], 0)
    else:
        X = np.zeros((ind_len, dimension, dimension, channel))
        for i in range(ind_len):
            X[i, :, :, :] = cv2.imread(folder_path + "/" + img_list[file_index[0, i]])
    if label == 0:
        Y = np.zeros((ind_len,))
    else:
        Y = np.ones((ind_len,))
    return X, Y

#get minibatch
def minibatch_get(mb_n, mb_size, index_matrix):
    ind_mat = index_matrix
    for n in range(0, sample_num):
        sample_path1 = marker_path1 + "/" + sample_list[n]
        sample_path2 = marker_path2 + "/" + sample_list[n]
        if int(sample_list[n]) > 131:
            sample_label = 0
        else:
            sample_label = 1
        get_ind = np.array(np.where((ind_mat[n, :] >= (mb_n * mb_size)) & (ind_mat[n, :] < ((mb_n + 1) * mb_size)))).astype('int')
        marker1, label_mat= img2mat(sample_path1, sample_label, file_index = get_ind)
        marker2, _ = img2mat(sample_path2, sample_label, file_index = get_ind)
        sample_mat = np.append(marker1.astype('float'), marker2.astype('float'), axis = -1)
        if n == 0:
            whole_data = sample_mat
            whole_label = label_mat
        else:
            whole_data = np.append(whole_data, sample_mat, axis = 0)
            whole_label = np.append(whole_label, label_mat, axis = 0)
        y_classes = 2
        label_hot = np.eye(y_classes)[whole_label.reshape(-1).astype('int')]
    return whole_data, label_hot
    #print(str(n+1) + "samples finished")

def dev_set_get(total_batch_num, mb_size, index_matrix):
    ind_mat = index_matrix
    for n in range(0, sample_num):
        sample_path1 = marker_path1 + "/" + sample_list[n]
        sample_path2 = marker_path2 + "/" + sample_list[n]
        if int(sample_list[n]) > 131:
            sample_label = 0
        else:
            sample_label = 1
        get_ind = np.array(np.where(ind_mat[n, :] >= (total_batch_num * mb_size))).astype('int')
        marker1, label_mat= img2mat(sample_path1, sample_label, file_index = get_ind)
        marker2, _ = img2mat(sample_path2, sample_label, file_index = get_ind)
        sample_mat = np.append(marker1.astype('float'), marker2.astype('float'), axis = -1)
        if n == 0:
            whole_data = sample_mat
            whole_label = label_mat
        else:
            whole_data = np.append(whole_data, sample_mat, axis = 0)
            whole_label = np.append(whole_label, label_mat, axis = 0)
        y_classes = 2
        label_hot = np.eye(y_classes)[whole_label.reshape(-1).astype('int')]
    return whole_data, label_hot

def ph_create(n_h0, n_w0, n_c0, n_y):
    X = tf.placeholder(dtype = 'float', shape = (None, n_h0, n_w0, n_c0))
    Y = tf.placeholder(dtype = 'float', shape = (None, n_y))
    return X, Y

def kernel_initializer():
    #Weight matrix of Conv layer: W; W.shape = [filer_height, filter_width, input_channel, Num_filters]
    W1 = tf.get_variable("W1", [5, 5, 2, 8], initializer = tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [4, 4, 8, 16], initializer = tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable("W3", [2, 2, 16, 32], initializer = tf.contrib.layers.xavier_initializer())

    weights = {"W1": W1, "W2": W2, "W3": W3}

    return weights

def forward_prop(training_input, weights):
    #Structure: CONV2D -> BN -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> CONV2D -> MAXPOOL -> FLATTEN -> FC
    W1 = weights["W1"]
    W2 = weights["W2"]
    W3 = weights["W3"]

    Z1 = tf.nn.conv2d(training_input, W1, strides = [1, 1, 1, 1], padding = 'SAME')
    #mean1, variance1 = tf.nn.moments(Z1, axes = [0])
    #B1 = tf.nn.batch_normalization(Z1, mean = mean1, variance = variance1, variance_epsilon = 0.99, offset = None, scale = None)
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1, 8, 8, 1], strides = [1, 8, 8, 1], padding = 'SAME')
    Z2 = tf.nn.conv2d(P1, W2, strides = [1, 1, 1, 1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding = 'SAME')
    Z3 = tf.nn.conv2d(P2, W3, strides = [1, 1, 1, 1], padding = 'SAME')
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    P3 = tf.contrib.layers.flatten(P3)
    Z4 = tf.contrib.layers.fully_connected(P3, 2, activation_fn = None)

    return Z4

def cost_func(Z, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y))
    return cost

def cnn_model(total_batch_num, mb_size, index_matrix, n_h0 = 512, n_w0= 512, n_c0 = 2, n_y = 2, learning_rate = 0.009, epoch_num = 100, print_cost = True):
    ops.reset_default_graph()
    costs = []

    X, Y = ph_create(n_h0, n_w0, n_c0, n_y)
    weights = kernel_initializer()
    Z = forward_prop(X, weights)
    cost = cost_func(Z, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epoch_num):
            mb_cost = 0.
            for mb in range(total_batch_num):
                mb_X, mb_Y = minibatch_get(mb, mb_size, index_matrix)
                mb_X /= 255
                mb_Y /= 255
                _ , cost_current_mb = sess.run([optimizer, cost], feed_dict = {X: mb_X, Y: mb_Y})

                mb_cost += cost_current_mb / total_batch_num

            if print_cost == True and epoch % 5 == 0:
                print ("cost after epoch %i: %f" % (epoch, mb_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(mb_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        predict_op = tf.argmax(Z, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        X_test, Y_test = dev_set_get(total_batch_num, mb_size, index_matrix)
        X_test /= 255
        Y_test /= 255
        #train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        #print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return test_accuracy, weights


path =  sys.path[0]+"/"
marker_path1 = path + "miR_21"
marker_path2 = path + "TTF_1"

sample_list = os.listdir(marker_path1)
sample_num = len(sample_list)
ind_mat = mb_index(sample_num, 169)
_, weights = cnn_model(total_batch_num = 168, mb_size = 1, index_matrix = ind_mat, n_h0 = 512, n_w0= 512, n_c0 = 2, n_y = 2, learning_rate = 0.009, epoch_num = 10, print_cost = True)














#np.save(path + "whole_data.npy", whole_data)
#np.save(path + "whole_label.npy", whole_label)
    



        
        