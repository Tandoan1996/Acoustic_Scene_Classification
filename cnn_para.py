
#import tensorflow as tf
import numpy as np
import os

class cnn_para(object):
    """
    define a class to store parameters,
    the input should be feature matrix of training and testing
    """

    def __init__(self):

        #======================= Trainging parameters
        self.n_class            = 10  # Final output classes TODO
        self.n_output           = 10  # Final output   TODO
        self.l2_lamda           = 0.0001  # lamda prarameter

        #========================  Input parameters
        # batchxheightxwidthxchannel is input image size
        # batchx52x40x1, channel = 1 due to binary image
        self.n_freq             = 128  #height
        self.n_time             = 128  #width
        self.n_chan             = 1  #width
  
        #========================  CNN structure parameters
        #=======Layer 01: conv 
        #conv
        self.l01_filter_height  = 3
        self.l01_filter_width   = 3
        self.l01_pre_filter_num = 1
        self.l01_filter_num     = 32
        self.l01_conv_padding   = 'SAME'        #SAME: zero padding; VALID: without padding
        self.l01_conv_stride    = [1,1,1,1]      
        #batch
        self.l01_is_norm        = False   #False: not using batch layer, True: using batch layer
        #act
        self.l01_conv_act_func  = 'RELU'
        #pool
        self.l01_is_pool        = True   #False: not using pool layer, True: using pool layer
        self.l01_pool_type      = 'MAX' #MAX or MEAN  or GLOBAL (If GLOBAL, other pool parameters are not used)
        self.l01_pool_padding   = 'VALID'  #or 'SAME'
        self.l01_pool_stride    = [1,2,2,1]
        self.l01_pool_ksize     = [1,2,2,1]
        #drop
        self.l01_is_drop        = False
        self.l01_drop_prob      = 0.2

        #=======Layer 02: conv 
        #conv
        self.l02_filter_height  = 3
        self.l02_filter_width   = 3
        self.l02_pre_filter_num = 32
        self.l02_filter_num     = 64
        self.l02_conv_padding   = 'SAME'        #SAME: zero padding; VALID: without padding
        self.l02_conv_stride    = [1,1,1,1]      
        #batch
        self.l02_is_norm        = False   #False: not using batch layer, True: using batch layer
        #act
        self.l02_conv_act_func  = 'RELU'
        #pool
        self.l02_is_pool        = True   #False: not using pool layer, True: using pool layer
        self.l02_pool_type      = 'MAX' #MAX or MEAN  or GLOBAL (If GLOBAL, other pool parameters are not used)
        self.l02_pool_padding   = 'VALID'  #or 'SAME'
        self.l02_pool_stride    = [1,2,2,1]
        self.l02_pool_ksize     = [1,2,2,1]
        #drop
        self.l02_is_drop        = False
        self.l02_drop_prob      = 0.2

        #=======Layer 03: conv 
        #conv
        self.l03_filter_height  = 3
        self.l03_filter_width   = 3
        self.l03_pre_filter_num = 64
        self.l03_filter_num     = 128
        self.l03_conv_padding   = 'SAME'        #SAME: zero padding; VALID: without padding
        self.l03_conv_stride    = [1,1,1,1]      
        #batch
        self.l03_is_norm        = False   #False: not using batch layer, True: using batch layer
        #act
        self.l03_conv_act_func  = 'RELU'
        #pool
        self.l03_is_pool        = True   #False: not using pool layer, True: using pool layer
        self.l03_pool_type      = 'MAX' #MAX or MEAN  or GLOBAL (If GLOBAL, other pool parameters are not used)
        self.l03_pool_padding   = 'VALID'  #or 'SAME'
        self.l03_pool_stride    = [1,2,2,1]
        self.l03_pool_ksize     = [1,2,2,1]
        #drop
        self.l03_is_drop        = False
        self.l03_drop_prob      = 0.2

        #=======Layer 04: conv 
        #conv
        self.l04_filter_height  = 3
        self.l04_filter_width   = 3
        self.l04_pre_filter_num = 128
        self.l04_filter_num     = 256
        self.l04_conv_padding   = 'SAME'        #SAME: zero padding; VALID: without padding
        self.l04_conv_stride    = [1,1,1,1]      
        #batch
        self.l04_is_norm        = False   #False: not using batch layer, True: using batch layer
        #act
        self.l04_conv_act_func  = 'RELU'
        #pool
        self.l04_is_pool        = True   #False: not using pool layer, True: using pool layer
        self.l04_pool_type      = 'GLOBAL_MEAN' #MAX or MEAN or GLOBAL
        self.l04_pool_padding   = 'VALID'  #or 'SAME'
        self.l04_pool_stride    = [1,2,2,1]
        self.l04_pool_ksize     = [1,2,2,1]
        #drop
        self.l04_is_drop        = False
        self.l04_drop_prob      = 1

        #=======Layer 05: full connection
        self.l05_fc             = 512 # node number of first full-connected layer 
        self.l05_is_act         = True
        self.l05_act_func       = 'RELU'
        self.l05_is_drop        = False
        self.l05_drop_prob      = 1

        #=======Layer 06: full connection
        self.l06_fc             = 1024  # node number of first full-connected layer 
        self.l06_is_act         = True
        self.l06_act_func       = 'RELU'
        self.l06_is_drop        = False
        self.l06_drop_prob      = 1

        #=======Layer 07: Final layer
        self.l07_fc             = 10   # output node number = class numbe
        self.l07_is_act         = False
        self.l07_act_func       = 'RELU'
        self.l07_is_drop        = False
        self.l07_drop_prob      = 1


