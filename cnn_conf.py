import tensorflow as tf
import numpy as np
import os
from cnn_para  import cnn_para

#======================================================================================================#

class cnn_conf(object):

    def __init__( self):

        self.cnn_para = cnn_para()
        
        # These arguments are transfered from 'step02...' file
        self.input_layer_val   = tf.placeholder(tf.float32, [None, self.cnn_para.n_freq, self.cnn_para.n_time, self.cnn_para.n_chan], name="input_layer_val")
        self.expected_classes  = tf.placeholder(tf.float32, [None, self.cnn_para.n_class], name="expected_classes")
        self.mode              = tf.placeholder(tf.bool, name="running_mode")

        ### ======== LAYER 01
        with tf.device('/gpu:0'), tf.variable_scope("conv01")as scope:
             [self.output_layer01, self.conv01_layer] = self.conv_layer(
                                                   self.input_layer_val,

                                                   self.cnn_para.l01_filter_height,
                                                   self.cnn_para.l01_filter_width,
                                                   self.cnn_para.l01_pre_filter_num,
                                                   self.cnn_para.l01_filter_num,
                                                   self.cnn_para.l01_conv_padding,
                                                   self.cnn_para.l01_conv_stride,

                                                   self.cnn_para.l01_is_norm,

                                                   self.cnn_para.l01_conv_act_func,

                                                   self.cnn_para.l01_is_pool,
                                                   self.cnn_para.l01_pool_type,
                                                   self.cnn_para.l01_pool_padding,
                                                   self.cnn_para.l01_pool_stride,
                                                   self.cnn_para.l01_pool_ksize,

                                                   self.cnn_para.l01_is_drop,
                                                   self.cnn_para.l01_drop_prob,

                                                   self.mode,
                                                   scope=scope
                                                  )   

        ### ======== LAYER 02
        with tf.device('/gpu:0'), tf.variable_scope("conv02")as scope:
             [self.output_layer02, self.conv02_layer] = self.conv_layer(
                                                   self.output_layer01,

                                                   self.cnn_para.l02_filter_height,
                                                   self.cnn_para.l02_filter_width,
                                                   self.cnn_para.l02_pre_filter_num,
                                                   self.cnn_para.l02_filter_num,
                                                   self.cnn_para.l02_conv_padding,
                                                   self.cnn_para.l02_conv_stride,

                                                   self.cnn_para.l02_is_norm,

                                                   self.cnn_para.l02_conv_act_func,

                                                   self.cnn_para.l02_is_pool,
                                                   self.cnn_para.l02_pool_type,
                                                   self.cnn_para.l02_pool_padding,
                                                   self.cnn_para.l02_pool_stride,
                                                   self.cnn_para.l02_pool_ksize,

                                                   self.cnn_para.l02_is_drop,
                                                   self.cnn_para.l02_drop_prob,

                                                   self.mode,
                                                   scope=scope
                                                  )   
        ### ======== LAYER 03
        with tf.device('/gpu:0'), tf.variable_scope("conv03")as scope:
             [self.output_layer03, self.conv03_layer] = self.conv_layer(
                                                   self.output_layer02,

                                                   self.cnn_para.l03_filter_height,
                                                   self.cnn_para.l03_filter_width,
                                                   self.cnn_para.l03_pre_filter_num,
                                                   self.cnn_para.l03_filter_num,
                                                   self.cnn_para.l03_conv_padding,
                                                   self.cnn_para.l03_conv_stride,

                                                   self.cnn_para.l03_is_norm,

                                                   self.cnn_para.l03_conv_act_func,

                                                   self.cnn_para.l03_is_pool,
                                                   self.cnn_para.l03_pool_type,
                                                   self.cnn_para.l03_pool_padding,
                                                   self.cnn_para.l03_pool_stride,
                                                   self.cnn_para.l03_pool_ksize,

                                                   self.cnn_para.l03_is_drop,
                                                   self.cnn_para.l03_drop_prob,

                                                   self.mode,
                                                   scope=scope
                                                  )   
        ### ======== LAYER 04
        with tf.device('/gpu:0'), tf.variable_scope("conv04")as scope:
             [self.output_layer04, self.conv04_layer] = self.conv_layer(
                                                   self.output_layer03,

                                                   self.cnn_para.l04_filter_height,
                                                   self.cnn_para.l04_filter_width,
                                                   self.cnn_para.l04_pre_filter_num,
                                                   self.cnn_para.l04_filter_num,
                                                   self.cnn_para.l04_conv_padding,
                                                   self.cnn_para.l04_conv_stride,

                                                   self.cnn_para.l04_is_norm,

                                                   self.cnn_para.l04_conv_act_func,

                                                   self.cnn_para.l04_is_pool,
                                                   self.cnn_para.l04_pool_type,
                                                   self.cnn_para.l04_pool_padding,
                                                   self.cnn_para.l04_pool_stride,
                                                   self.cnn_para.l04_pool_ksize,

                                                   self.cnn_para.l04_is_drop,
                                                   self.cnn_para.l04_drop_prob,

                                                   self.mode,
                                                   scope=scope
                                                  )   


             #print self.output_layer04.get_shape() #nx256 if GLOBAL POOL
             #exit()

             #01 GLOBAL MEAN POOL
             [_,col] = self.output_layer04.get_shape()   #nx1x256
             #print col

             #02 FREQ MEAN
             #[_,freq_bin,time_bin,channel] = self.output_layer04.get_shape() 
             #print freq_bin, time_bin, channel 
              
             #exit()

        ###========= Plattenning output of conv layer02
        with tf.device('/gpu:0'), tf.variable_scope("platterning-l04") as scope:
             #01 GLOBAL
             output_layer04_flat_dim = int(col)

             #02 FREQ MEAN
             #output_layer04_flat_dim = int(channel*time_bin*freq_bin)

             self.output_layer04_flatten  = tf.reshape(self.output_layer04, [-1, output_layer04_flat_dim])
             #print output_layer04_flatten.get_shape()
             #exit()
             
        ### ======== Layer 05: full connection
        with tf.device('/gpu:0'), tf.variable_scope("fully_layer05") as scope:
            self.output_layer05 = self.fully_layer(
                                                    self.output_layer04_flatten,
                                                    output_layer04_flat_dim,

                                                    self.cnn_para.l05_fc, 

                                                    self.cnn_para.l05_is_act,
                                                    self.cnn_para.l05_act_func, 

                                                    self.cnn_para.l05_is_drop,
                                                    self.cnn_para.l05_drop_prob,
                                                    scope=scope
                                                    )
        ### ======== Layer 06: full connection
        with tf.device('/gpu:0'), tf.variable_scope("fully_layer06") as scope:
            self.output_layer06 = self.fully_layer(
                                                    self.output_layer05,
                                                    self.cnn_para.l05_fc,

                                                    self.cnn_para.l06_fc, 

                                                    self.cnn_para.l06_is_act,
                                                    self.cnn_para.l06_act_func, 

                                                    self.cnn_para.l06_is_drop,
                                                    self.cnn_para.l06_drop_prob,
                                                    scope=scope
                                                    )

        ### ======== Layer 07: full connection
        with tf.device('/gpu:0'), tf.variable_scope("fully_layer07") as scope:
            self.output_layer07 = self.fully_layer(
                                                    self.output_layer06,
                                                    self.cnn_para.l06_fc,

                                                    self.cnn_para.l07_fc, 

                                                    self.cnn_para.l07_is_act,
                                                    self.cnn_para.l07_act_func, 

                                                    self.cnn_para.l07_is_drop,
                                                    self.cnn_para.l07_drop_prob,
                                                    scope=scope
                                                    )
 
            self.output_layer = self.output_layer07
            #print self.output_layer05.get_shape()           #n x nClassa
            #exit()
            self.prob_output_layer = tf.nn.softmax(self.output_layer07)

        ### ======================================== LOSS FUNCTION AND ACCURACY =========================
        ### loss function
        with tf.device('/gpu:0'), tf.variable_scope("loss") as scope:
  
            # l2 loss  
            l2_loss = self.cnn_para.l2_lamda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

            # main loss
            losses  = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.expected_classes, logits=self.output_layer)
 
            # final loss
            self.loss = tf.reduce_mean(losses) + l2_loss    #reduce_sum or reduce_mean
            #self.loss = tf.reduce_mean(losses)             

        ### Calculate Accuracy  
        with tf.device('/gpu:0'), tf.name_scope("accuracy") as scope:
            self.correct_prediction = tf.equal(tf.argmax(self.output_layer,1), tf.argmax(self.expected_classes,1))
            self.accuracy      = tf.reduce_mean(tf.cast(self.correct_prediction,"float", name="accuracy" ))


###==================================================== OTHER FUNCTION ============================
    #02/ CONV LAYER
    def conv_layer(self, 
                  input_value, 

                  filter_height, 
                  filter_width, 
                  pre_filter_num, 
                  filter_num, 
                  conv_padding, 
                  conv_stride,

                  is_norm,

                  act_func,

                  is_pool, 
                  pool_type, 
                  pool_padding, 
                  pool_stride, 
                  pool_ksize, 

                  is_drop,
                  drop_prob,

                  mode,
                  scope=None
                 ):
        #------------------------------#
        def reduce_var(x, axis=None, keepdims=False, name=None):
            m = tf.reduce_mean(x, axis=axis, keepdims=True, name=name) #keep same dimension for subtraction
            devs_squared = tf.square(x - m)
            return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims, name=name)
        
        def reduce_std(x, axis=None, keepdims=False, name=None):
            return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims, name=name))

        #------------------------------#

        with tf.variable_scope(scope or 'conv-layer') as scope:

            # shape: [5,5,1,32] or [5,5,32,64]
            filter_shape = [filter_height, filter_width, pre_filter_num, filter_num]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")   # this is kernel 
            b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")

            #Convolution layer 
            conv_output = tf.nn.conv2d(
                                 input_value,
                                 W,
                                 strides = conv_stride,
                                 padding = conv_padding,
                                 name="conv"
                                 )  #default: data format = NHWC


            #Active function layer
            if (act_func == 'RELU'):
                act_func_output = tf.nn.relu(tf.nn.bias_add(conv_output, b), name="RELU")
            elif (act_func == 'TANH'):
                act_func_output = tf.nn.tanh(tf.nn.bias_add(conv_output, b), name="TANH")

            #BachNorm Layer
            if(is_norm == True):
                batch_output = tf.contrib.layers.batch_norm(
                                                             act_func_output, 
                                                             is_training = mode, 
                                                             decay = 0.9,
                                                             zero_debias_moving_mean=True
                                                           )
            else:     
                batch_output = act_func_output

            #Pooling layer
            if(is_pool == True):
                if (pool_type == 'MEAN'):
                    pool_output = tf.nn.avg_pool(
                                          batch_output,
                                          ksize   = pool_ksize,   
                                          strides = pool_stride,
                                          padding = pool_padding,
                                          name="mean_pool"
                                         )
                elif (pool_type == 'MAX'):
                    pool_output = tf.nn.max_pool(
                                          batch_output,
                                          ksize   = pool_ksize,   
                                          strides = pool_stride,
                                          padding = pool_padding,
                                          name="max_pool"
                                         )
                elif (pool_type == 'GLOBAL_MEAN'):
                    pool_output = tf.reduce_mean(
                                          batch_output,
                                          axis=[1,2],
                                          name='global_moment01_pool'
                                         )
                elif (pool_type == 'GLOBAL_STD'):   #only for testing (not apply for training)
                    pool_output = reduce_std(
                                          batch_output,
                                          axis=[1,2],
                                          name = "global_moment02_pool"
                                         )
                    #print pool_output.get_shape()
                    #exit()
            else:
                pool_output = batch_output

            #Dropout
            if(is_drop == True):
                drop_output = tf.layers.dropout(
                                                pool_output, 
                                                rate = drop_prob,
                                                training = mode,
                                                name = 'Dropout'
                                               )
            else:     
                drop_output = pool_output

            return drop_output, conv_output

    ### 02/ FULL CONNECTTION  LAYER
    def fully_layer(
                     self, 
                     input_val, 
                     input_size, 
                     output_size, 
                     is_act,
                     act_func,
                     is_drop,
                     drop_prob, 
                     scope=None
                   ):

        with tf.variable_scope(scope or 'fully-layer') as scope:
            #initial parameter
            W    = tf.random_normal([input_size, output_size], stddev=0.1, dtype=tf.float32)
            bias = tf.random_normal([output_size], stddev=0.1, dtype=tf.float32)
            W    = tf.Variable(W)
            bias = tf.Variable(bias)

            #Dense 
            dense_output = tf.add(tf.matmul(input_val, W), bias)  

            #Active function
            if(is_act == True):
                if (act_func == 'RELU'):    
                    act_func_output = tf.nn.relu(dense_output)   
                elif (act_func == 'SOFTMAX'):
                    act_func_output  = tf.nn.softmax(dense_output)             
                elif (act_func == 'TANH'):
                    act_func_output  = tf.nn.tanh(dense_output)                 
            else:
                act_func_output = dense_output

            #Drop out
            if(is_drop == True):
                drop_output = tf.layers.dropout(
                                                act_func_output, 
                                                rate = drop_prob,
                                                training = mode,
                                                name = 'Dropout'
                                               )
            else:
                drop_output = act_func_output

            #Return 
            return drop_output


#===================================================================================
