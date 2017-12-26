"""
Please implement a standard AlexNet model here as defined in the paper
    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Note: Although you will only have to edit a small fraction of the code at the
beginning of the assignment by filling in the blank spaces, you will need to
build on the completed starter code to fully complete the assignment,
We expect that you familiarize yourself with the codebase and learn how to
setup your own experiments taking the assignments as a basis. This code does
not cover all parts of the assignment and only provides a starting point. To
fully complete the assignment significant changes have to be made and new
functions need to be added after filling in the blanks. Also, for your projects
we won't give out any code and you will have to use what you have learned from
your assignments. So please always carefully read through the entire code and
try to understand it. If you have any questions about the code structure,
we will be happy to answer it.

Attention: All sections that need to be changed to complete the starter code
are marked with EDIT!
"""

import os
import numpy as np
import tensorflow as tf

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

def alexnet_model(inputs, train=True, norm=True, **kwargs):
    """
    AlexNet model definition as defined in the paper:
    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

    You will need to EDIT this function. Please put your AlexNet implementation here.
    
    Note: 
    1.) inputs['images'] is a [BATCH_SIZE x HEIGHT x WIDTH x CHANNELS] array coming
    from the data provider.
    2.) You will need to return 'output' which is a dictionary where 
    - output['pred'] is set to the output of your model
    - output['conv1'] is set to the output of the conv1 layer
    - output['conv1_kernel'] is set to conv1 kernels
    - output['conv2'] is set to the output of the conv2 layer
    - output['conv2_kernel'] is set to conv2 kernels
    - and so on...
    The output dictionary should include the following keys for AlexNet:
    ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'pool1', 
     'pool2', 'pool5', 'fc6', 'fc7', 'fc8'] 
    as well as the respective ['*_kernel'] keys for the kernels
    3.) Set your variable scopes to the name of the respective layers, e.g.
        with tf.variable_scope('conv1'):
            outputs['conv1'] = ...
            outputs['pool1'] = ...
    and
        with tf.variable_scope('fc6'):
            outputs['fc6'] = ...
    and so on. 
    4.) Use tf.get_variable() to create variables, while setting name='weights'
    for each kernel, and name='bias' for each bias for all conv and fc layers.
    For the pool layers name='pool'. Use the xavier initializer to initialize
    your conv kernels, and the truncated normal initializer for the fc kernels.
    The biases of the conv and fc layers should be initalized with a constant 
    initializer to 0, except for conv2, fc6, and fc7 whose biases should be
    initialized to 0.1.

    These steps are necessary to correctly load the pretrained alexnet model
    from the database for the second part of the assignment.
    """

    # propagate input targets
    outputs = inputs
    dropout = .5 if train else None
    input_to_network = inputs['images']

    
    with tf.variable_scope('conv1'):
        Wconv1 = tf.get_variable("weights", shape=[11, 11, 3, 96], tf.float32, tf.contrib.layers.xavier_initializer()) 
        bconv1 = tf.get_variable("bias", shape=[96], tf.float32, initializer=tf.zeros_initializer())
        #forward pass
        conv1_out = tf.nn.relu(tf.nn.conv2d(inputs['images'], Wconv1, [1, 4, 4, 1], padding='SAME') + bconv1)
        #local response normalisation
        lrn1 = tf.nn.local_response_normalization(conv1_out,depth_radius=5,alpha=1e-4,beta=0.75,bias=2)
        #maxpool
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name='pool')
        
        outputs['conv1'] = conv1_out
        outputs['pool1'] = pool1
        outputs['conv1_kernel'] = Wconv1
        
    with tf.variable_scope('conv2'):
        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
        Wconv2 = tf.get_variable("weights", shape=[5, 5, 96, 256], tf.float32, tf.contrib.layers.xavier_initializer()) 
        bconv2 = tf.get_variable("bias", shape=[256], tf.float32, initializer=tf.constant_initializer(0.1))
        conv2_out = tf.nn.relu(conv(pool1, Wconv2, bconv2, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group))
        #local response normalisation
        lrn2 = tf.nn.local_response_normalization(conv2_out,depth_radius=5,alpha=1e-4,beta=0.75,bias=2)
        #maxpool
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name='pool')
        
        outputs['conv2'] = conv2_out
        outputs['pool2'] = pool2
        outputs['conv2_kernel'] = Wconv2
        
    with tf.variable_scope('conv3'):
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group=1 #group=1 means full convolution, group =2 means half and half
        Wconv3 = tf.get_variable("weights", shape=[3, 3, 256, 384], tf.float32, tf.contrib.layers.xavier_initializer()) 
        bconv3 = tf.get_variable("bias", shape=[384], tf.float32, initializer=tf.zeros_initializer())
        conv3_out = tf.nn.relu(conv(pool2, Wconv3, bconv3, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group))
        
        outputs['conv3'] = conv3_out
        outputs['conv3_kernel'] = Wconv3
        
    with tf.variable_scope('conv4'):
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
        Wconv4 = tf.get_variable("weights", shape=[3, 3, 384, 384], tf.float32, tf.contrib.layers.xavier_initializer()) 
        bconv4 = tf.get_variable("bias", shape=[384], tf.float32, initializer=tf.zeros_initializer())
        conv4_out = tf.nn.relu(conv(conv3_out, Wconv4, bconv4, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group))
        
        outputs['conv4'] = conv4_out
        outputs['conv4_kernel'] = Wconv4
        
    with tf.variable_scope('conv5'):
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        Wconv5 = tf.get_variable("weights", shape=[3, 3, 384, 256], tf.float32, tf.contrib.layers.xavier_initializer()) 
        bconv5 = tf.get_variable("bias", shape=[256], tf.float32, initializer=tf.zeros_initializer())
        conv5_out = tf.nn.relu(conv(conv4_out, Wconv5, bconv5, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group))
        #maxpool
        pool5 = tf.nn.max_pool(conv5_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',name='pool')
        
        outputs['conv5'] = conv5_out
        outputs['pool5'] = pool5
        outputs['conv5_kernel'] = Wconv5
        
    with tf.variable_scope('fc6'):
        Wfc6 = tf.get_variable("weights", shape=[int(prod(pool5.get_shape()[1:])),4096], tf.float32, tf.truncated_normal_initializer()) 
        bfc6 = tf.get_variable("bias", shape=[4096], tf.float32, initializer=tf.constant_initializer(0.1))
        fc6_out = tf.nn.relu_layer(tf.reshape(pool5, [-1, int(prod(pool5.get_shape()[1:]))]), Wfc6, bfc6)

        outputs['fc6'] = fc6_out
        outputs['fc6_kernel'] = Wfc6
        
    with tf.variable_scope('fc7'):
        Wfc7 = tf.get_variable("weights", shape=[4096,4096], tf.float32, tf.truncated_normal_initializer()) 
        bfc7 = tf.get_variable("bias", shape=[4096], tf.float32, initializer=tf.constant_initializer(0.1))
        fc7_out = tf.nn.relu_layer(fc6_out, Wfc7, bfc7)

        outputs['fc7'] = fc7_out
        outputs['fc7_kernel'] = Wfc7
        
    with tf.variable_scope('fc8'):
        Wfc8 = tf.get_variable("weights", shape=[4096,1000], tf.float32, tf.truncated_normal_initializer()) 
        bfc8 = tf.get_variable("bias", shape=[1000], tf.float32, initializer=tf.constant_initializer(0.0))
        fc8_out = tf.nn.xw_plus_b(fc7_out, Wfc8, bfc8)

        outputs['fc8'] = fc8_out
        outputs['fc8_kernel'] = Wfc8
        outputs['pred'] = fc8_out
        

    for k in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'pool1',
            'pool2', 'pool5', 'fc6', 'fc7', 'fc8', 'conv1_kernel', 'pred']:
        assert k in outputs, '%s was not found in outputs' % k
    return outputs, {}
