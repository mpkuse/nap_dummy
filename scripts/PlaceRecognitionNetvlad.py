""" The code NetVLAD PlaceRecognition

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 11th May, 2017

"""


import rospy
import rospkg


import numpy as np
import Queue
import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image


import tensorflow as tf
import tensorflow.contrib.slim as slim

from CartWheelFlow import VGGDescriptor
import DimRed
from Quaternion import Quat

#
import TerminalColors
tcolor = TerminalColors.bcolors()

class PlaceRecognitionNetvlad:
    def __init__(self, PARAM_MODEL, PARAM_CALLBACK_SKIP=2, PARAM_K=48):
        #
        # Init netvlad - def computational graph, load trained model
        self.tf_x = tf.placeholder( 'float', [None,240,320,3], name='x' )
        self.is_training = tf.placeholder( tf.bool, [], name='is_training')
        self.vgg_obj = VGGDescriptor(b=1, K=PARAM_K)
        self.tf_vlad_word = self.vgg_obj.vgg16(self.tf_x, self.is_training)
        self.tensorflow_session = tf.Session()
        tensorflow_saver = tf.train.Saver()
        print tcolor.OKGREEN,'Restore model from : ', PARAM_MODEL, tcolor.ENDC
        tensorflow_saver.restore( self.tensorflow_session, PARAM_MODEL )




        #
        # Setup internal Image queue
        self.im_queue = Queue.Queue()
        self.im_timestamp_queue = Queue.Queue()
        self.call_q = 0
        self.PARAM_CALLBACK_SKIP = PARAM_CALLBACK_SKIP

        self.PARAM_MODEL = PARAM_MODEL
        self.PARAM_MODEL_DIM_RED = None

    def load_siamese_dim_red_module( self, PARAM_MODEL_DIM_RED, PARAM_input_dim, PARAM_net_intermediate_dim, PARAM_net_out_dim ):
        #
        # Init DimRed Mapping (Dimensionality Reduction by Learning Invariant Mapping)
        if PARAM_MODEL_DIM_RED is not None:
            self.dm_vlad_word = tf.placeholder( 'float', [None,None], name='vlad_word' )

            self.net = DimRed.DimRed(n_input_dim=PARAM_input_dim, n_intermediate_dim=PARAM_net_intermediate_dim, n_output_dim=PARAM_net_out_dim)
            self.dm_vlad_char = self.net.fc( self.dm_vlad_word )
            tensorflow_saver2 = tf.train.Saver( self.net.return_vars() )
            print tcolor.OKGREEN,'Restore model from : ', PARAM_MODEL_DIM_RED, tcolor.ENDC
            tensorflow_saver2.restore( self.tensorflow_session, PARAM_MODEL_DIM_RED )

            self.PARAM_MODEL_DIM_RED = PARAM_MODEL_DIM_RED



    ## Subscribes to image message and store in internal queue
    def callback_image( self, data ):
        n_SKIP = self.PARAM_CALLBACK_SKIP

        # rospy.logdebug( 'Received Image : %d,%d' %( data.height, data.width ) )
        cv_image = CvBridge().imgmsg_to_cv2( data, 'bgr8' )

        if self.call_q%n_SKIP == 0: #only use 1 out of 10 images
            # self.im_queue.put( cv_image )
            self.im_queue.put(cv2.resize(cv_image, (320,240) ) )
            self.im_timestamp_queue.put(data.header.stamp)
        self.call_q = self.call_q + 1

    ## Normalize colors for an image - used in NetVLAD descriptor extraction
    def rgbnormalize( self, im ):
        im_R = im[:,:,0].astype('float32')
        im_G = im[:,:,1].astype('float32')
        im_B = im[:,:,2].astype('float32')
        S = im_R + im_G + im_B
        out_im = np.zeros(im.shape)
        out_im[:,:,0] = im_R / (S+1.0)
        out_im[:,:,1] = im_G / (S+1.0)
        out_im[:,:,2] = im_B / (S+1.0)

        return out_im

    ## Normalize a batch of images. Calls `rgbnormalize()` - used in netvlad computation
    def normalize_batch( self, im_batch ):
        im_batch_normalized = np.zeros(im_batch.shape)
        for b in range(im_batch.shape[0]):
            im_batch_normalized[b,:,:,0] = self.zNormalize( im_batch[b,:,:,0])
            im_batch_normalized[b,:,:,1] = self.zNormalize( im_batch[b,:,:,1])
            im_batch_normalized[b,:,:,2] = self.zNormalize( im_batch[b,:,:,2])
            # im_batch_normalized[b,:,:,:] = self.rgbnormalize( im_batch[b,:,:,:] )


        # cv2.imshow( 'org_', (im_batch[0,:,:,:]).astype('uint8') )
        # cv2.imshow( 'out_', (im_batch_normalized[0,:,:,:]*255.).astype('uint8') )
        # cv2.waitKey(0)
        # code.interact(local=locals())
        return im_batch_normalized

    ## M is a 2D matrix
    def zNormalize(self,M):
        return (M-M.mean())/(M.std()+0.0001)


    ## 'x' can also be a vector
    def logistic( self, x ):
        #y = np.array(x)
        #return (1.0 / (1.0 + np.exp( 11.0*y - 3.0 )) + 0.01)
        # return (1.0 / (1.0 + 0.6*np.exp( 22.0*y - 2.0 )) + 0.04)
        filt = [0.1,0.2,0.4,0.2,0.1]
        if len(x) < len(filt):
            return (1.0 / (1.0 + np.exp( 11.0*x - 3.0 )) + 0.01)

        y = np.convolve( np.array(x), filt, 'same' )
        return (1.0 / (1.0 + np.exp( 11.0*y - 3.0 )) + 0.01)



    def extract_reduced_descriptor( self, im_raw ):
        # Convert to RGB format (from BGR) and resize
        im_ = cv2.resize( im_raw, (320,240) )
        A = cv2.cvtColor( im_.astype('uint8'), cv2.COLOR_RGB2BGR )


        ############# descriptors compute ##################
        # VLAD Computation
        im_batch = np.zeros( (1,A.shape[0],A.shape[1],A.shape[2]) )
        im_batch[0,:,:,:] = A.astype('float32')
        im_batch_normalized = self.normalize_batch( im_batch )

        feed_dict = {self.tf_x : im_batch_normalized,\
                     self.is_training:False,\
                     self.vgg_obj.initial_t: 0}
        tff_vlad_word = self.tensorflow_session.run( self.tf_vlad_word, feed_dict )
        d_WORD = tff_vlad_word[0,:]


        # Dim Reduction
        if self.PARAM_MODEL_DIM_RED is None:
            rospy.logerror( "PARAM_MODEL_DIM_RED is not loaded. You should probably be calling `extract_descriptor()`")
            return None

        dmm_vlad_char = self.tensorflow_session.run( self.dm_vlad_char, feed_dict={self.dm_vlad_word: tff_vlad_word})
        # dmm_vlad_char = dmm_vlad_char
        d_CHAR = dmm_vlad_char[0,:]

        return d_CHAR, d_WORD

    def extract_descriptor( self, im_raw ):
        # Convert to RGB format (from BGR) and resize
        im_ = cv2.resize( im_raw, (320,240) )
        A = cv2.cvtColor( im_.astype('uint8'), cv2.COLOR_RGB2BGR )


        ############# descriptors compute ##################
        # VLAD Computation
        im_batch = np.zeros( (1,A.shape[0],A.shape[1],A.shape[2]) )
        im_batch[0,:,:,:] = A.astype('float32')
        im_batch_normalized = self.normalize_batch( im_batch )

        feed_dict = {self.tf_x : im_batch_normalized,\
                     self.is_training:False,\
                     self.vgg_obj.initial_t: 0}

        # tff_vlad_word = self.tensorflow_session.run( self.tf_vlad_word, feed_dict )

        tff_vlad_word, tff_sm = self.tensorflow_session.run( [self.tf_vlad_word, self.vgg_obj.nl_sm], feed_dict )
        #b=1
        Assgn_matrix = np.reshape( tff_sm, [1,60,80,-1] ).argmax( axis=-1 ) #assuming batch size = 1
        self.Assgn_matrix = Assgn_matrix

        d_WORD = tff_vlad_word[0,:]

        return d_WORD
