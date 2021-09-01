import tensorflow as tf
from tensorflow.keras.losses import *
from tensorflow.keras import initializers
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.utils import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback
from codes_externes.LearningRate import *
from Networks.Loss_Constructor import *
from various_functions.tensor_tf_data_fonctions import *
from various_functions.rapports_pdf_fonctions import *
from various_functions.numpy_data_fonctions import *
from Networks.model_management import *
from various_functions.directory_management import *
from various_functions.custom_filters import *

import tensorflow_addons as tfa

from Networks.Compilation_NN import *
'''
Branch for Stylized High frequency details. Transfer Style with highpassband filter from a style image into the MAIN model output image. The branch is residual.
'''

# a. module
def Tb(x,nb,nombre_kernel,name_i):
    '''
    Tensorflow (keras API) T module 
    ''' 
    Conv2D_1 = Conv2D(filters=nb, kernel_size=nombre_kernel, strides=1, padding="same",name="T_conv0_"+str(name_i))(x)
    BN1 = tf.keras.layers.BatchNormalization(name="T_Bn0_"+str(name_i))(Conv2D_1)
    activation = tf.keras.layers.Activation('relu',name="T_act0_"+str(name_i))(BN1)
    Conv2D_2 = Conv2D(filters=nb, kernel_size=nombre_kernel, strides=1, padding = "same",name="T_conv1_"+str(name_i))(activation)
    BN2 = tf.keras.layers.BatchNormalization(name="T_Bn1_"+str(name_i))(Conv2D_2)
    activation = tf.keras.layers.Activation('relu',name="T_act1_"+str(name_i))(BN2)
    #res = concatenate([BN2,x],axis=3,name="T_"+str(name_i))
    res = Add(name="T_"+str(name_i))([BN2,x])
    return(res)  

    
# b. Branch
def STy_branch(model,main_network,taille_inputx:int,taille_inputy:int, filtres_branch:int, border:int, DOG_init:bool, DOG_fin: bool, BN_init:bool,BN_fin:bool,nombre_kernel:int,w_h_s,w_v_s,style_gram=0,loss_pixel=True, loss_perceptuelle=[], loss_style=[], ponderation_pixel=1.0, ponderation_perc=1.0, ponderation_style=1.0  ,profondeur=1, learning_rate=0.001):   
    '''
    Tensorflow (keras API) for Stycbcr branch trained on the top of pre-trained MAIN model  ; and (sizex,sizey,3) size input tensor (USED FOR TRAINING ON GIVEN PATCHES WITH DEFINED SIZES)    
    ''' 
    # Main model features loading
    if main_network=="SR_EDSR":
        input_network_border = Input((int(taille_inputx/4),int(taille_inputy/4),3))
        m=model.layers[-1].output            
        SR= Lambda(lambda x: (x)/255.,name="output_edsr_rgb_01.")(m)
        SR = Lambda(lambda x: tf_rgb2ycbcr(x)/255.,name="output_edsr_ycbcr_01")(SR)    
        
        SR_y = Lambda(lambda x: x[:,:,:,0],name="SR_y")(SR) 
        SR_y = Reshape((taille_inputx,taille_inputy,1), name = "SR_y_reshape")(SR_y) 
        y_1=tf.keras.layers.Activation('linear',name="copy")(SR_y)
        
        cb_1= Lambda(lambda x: x[:,:,:,1], name="cb_edsr")(SR)
        cb_1=tf.keras.layers.Reshape((taille_inputx,taille_inputy,1), name="cb_edsr_reshape")(cb_1)
    
        cr_1= Lambda(lambda x: x[:,:,:,2], name="cr_edsr")(SR)
        cr_1=tf.keras.layers.Reshape((taille_inputx,taille_inputy,1), name="cr_edsr_reshape")(cr_1)
    else:
        y_1 = model.get_layer('bicubic_crop').output
        cb_1 = model.get_layer('cb_input_crop').output
        cr_1 = model.get_layer('cr_input_crop').output
    
        SR = model.get_layer('SR').output
        SR_y = Lambda(lambda x: x[:,:,:,0],name="SR_y")(SR) 
        SR_y = Reshape((taille_inputx,taille_inputy,1), name = "SR_y_reshape")(SR_y) 
    
    y_crop=tf.keras.layers.Activation('linear',name="STm_i_y")(y_1)
    cb_crop=tf.keras.layers.Activation('linear',name="STm_i_cb")(cb_1)
    cr_crop=tf.keras.layers.Activation('linear',name="STm_i_cr")(cr_1)

    if DOG_init:
        conv1_lisse_y = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_yDoG_h")(y_crop)
        conv1_lisse_y = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_yDoG")(conv1_lisse_y)
        
        conv1_lisse_cb = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_cbDoG_h")(cb_crop)
        conv1_lisse_cb = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_cbDoG")(conv1_lisse_cb)  
        
        conv1_lisse_cr = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_crDoG_h")(cr_crop)
        conv1_lisse_cr = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_crDoG")(conv1_lisse_cr)       
    else:
        conv1_lisse_y = tf.keras.layers.Activation("linear",name="STm_i_yDoG")(y_crop)
        conv1_lisse_cb = tf.keras.layers.Activation("linear",name="STm_i_cbDoG")(cb_crop)
        conv1_lisse_cr = tf.keras.layers.Activation("linear",name="STm_i_crDoG")(cr_crop)
        
    if BN_init:
        conv1_lisse_y = tf.keras.layers.BatchNormalization(name="STm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb = tf.keras.layers.BatchNormalization(name="STm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr = tf.keras.layers.BatchNormalization(name="STm_i_crDoGBn")(conv1_lisse_cr)
    else:
        conv1_lisse_y=tf.keras.layers.Activation("linear",name="STm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb=tf.keras.layers.Activation("linear",name="STm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr=tf.keras.layers.Activation("linear",name="STm_i_crDoGBn")(conv1_lisse_cr)
        
    concat_ycbcr=concatenate([conv1_lisse_y,conv1_lisse_cb,conv1_lisse_cr],axis=3, name="concat_rgb_bn_style") 
    conv1=Conv2D(filters=filtres_branch, kernel_size=nombre_kernel, strides=1, padding="same",name="T_conv_0_base")(concat_ycbcr)
    conv1 = tf.keras.layers.BatchNormalization(name="T_BN_0_base")(conv1)
    conv1 = tf.keras.layers.Activation('relu',name="T_Act_0_base")(conv1)

    T1 = Tb(conv1,filtres_branch,nombre_kernel,name_i=0)
    T2 = Tb(T1,filtres_branch,nombre_kernel,name_i=1)
    T3 = Tb(T2,filtres_branch,nombre_kernel,name_i=2)
    T4 = Tb(T3,filtres_branch,nombre_kernel,name_i=3)
    
    residu_y = Conv2D(filters=1, kernel_size=1, strides=1, padding="same", name="gather_st",activation="relu")(T4)#
    y_out = Reshape((taille_inputx,taille_inputy,1),name="STm_o_y")(residu_y)
    
    if DOG_fin:   
        y_out = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_o_yDoG_h")(y_out)
        y_out = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_o_yDoG")(y_out)
    else:
        y_out = tf.keras.layers.Activation('linear',name="STm_o_yDoG")(y_out)
        
    if BN_fin:
        y_out = tf.keras.layers.BatchNormalization(name="STm_o_yDoGBn")(y_out)
    else:
        y_out = tf.keras.layers.Activation('linear',name="STm_o_yDoGBn")(y_out)
    
    # Tanh
    S=tf.keras.layers.Activation("tanh")(y_out) 
    
    # Median filter - custom 
    kernel_median = tf.expand_dims(tf.expand_dims(tf.ones((2,2)),axis=-1),axis=-1)
    sum_S = Conv2D(1, (kernel_median.shape[0]),trainable=False,padding="same",weights=[kernel_median],use_bias=False,name="sum_s")(S)
    
    max_S=tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same",name="max_S")(S)
    min_S=tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same",name="min_S")(-S)
    
    S=(sum_S-max_S+min_S)/2
    S=tf.keras.layers.Activation("linear",name="STm_o_yFinal_residu")(S)
    
    #Fb
    S=tf.keras.layers.BatchNormalization(name="F_BN_outputi_y_style_fb", trainable=True, scale=False,center=False)(S) 
    S  = S*(tf.math.sqrt(tf.math.reduce_variance(SR_y)))/2
    
    # Residual
    y_out = concatenate([S,SR_y],axis=3, name = "concatenated_ST_y") 
    y_out=tf.reduce_sum(y_out,axis=3,name="ST_y") 
    ST=Reshape((taille_inputx,taille_inputy,1),name="STm_o_yFinal")(y_out)
   
    # TO RGB FOR VGG
    ST_ycbcr=concatenate([ST,cb_crop,cr_crop],axis=3, name = "ST_ycbcr") 
    ST_ycbcr = tf.keras.layers.Reshape((taille_inputx,taille_inputy,3),name="ST_ycbcr_reshape")(ST_ycbcr)


    ST_rgb = Lambda(lambda x: tensor_ycbcr2rgb(x)/255.,name="ST_rgb")(ST_ycbcr)    
    ST_rgb= Reshape((taille_inputx,taille_inputy,3),name="ST_rgb_reshape")(ST_rgb)
    ST_rgb = Lambda(lambda x: tf.keras.backend.clip(x, 0, 1), name = "ST")(ST_rgb)  
    
    
    loss_list,outputs,weight_list = compilation_StyleTransfer(True,loss_perceptuelle, loss_style, loss_pixel,ponderation_pixel, ponderation_perc, ponderation_style,ST_rgb, style_gram)  
    new_model = Model(model.inputs, outputs)    

    new_model.compile(optimizer = Adam(lr=learning_rate), loss = loss_list, loss_weights=weight_list, metrics = [PSNR],run_eagerly=True)
    return(new_model,loss_list)
    
def STy_branch_none(model, main_network,filtres_branch:int, border:int, DOG_init:bool, DOG_fin: bool, BN_init:bool,BN_fin:bool,nombre_kernel:int,w_h_s,w_v_s):   
    '''
    Tensorflow (keras API) for Sty branch trained on the top of pre-trained MAIN model ; and (None,None,3) size input tensor (USED FOR INFERING ON ANY IMAGE)
    ''' 
    if main_network=="SR_EDSR":
        input_network_border = Input((None,None,3))
        m=model.layers[-1].output            
        SR= Lambda(lambda x: (x)/255.,name="output_edsr_rgb_01.")(m)
        SR = Lambda(lambda x: tf_rgb2ycbcr(x)/255.,name="output_edsr_ycbcr_01")(SR)    
        
        SR_y = Lambda(lambda x: x[:,:,:,0],name="SR_y")(SR) 
        SR_y = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="SR_y_reshape")(SR_y)
        y_1=tf.keras.layers.Activation('linear',name="copy")(SR_y)
        
        cb_1= Lambda(lambda x: x[:,:,:,1], name="cb_edsr")(SR)
        cb_1 = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="cb_edsr_reshape")(cb_1)

        cr_1= Lambda(lambda x: x[:,:,:,2], name="cr_edsr")(SR)
        cr_1 = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="cr_edsr_reshape")(cr_1)
        
    else:
        y_1 = model.get_layer('bicubic_crop').output
        cb_1 = model.get_layer('cb_input_crop').output
        cr_1 = model.get_layer('cr_input_crop').output
    
        SR = model.get_layer('SR').output
        SR_y = Lambda(lambda x: x[:,:,:,0],name="SR_y")(SR) 
        SR_y = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="SR_y_reshape")(SR_y)

    y_crop=tf.keras.layers.Activation('linear',name="STm_i_y")(y_1)
    cb_crop=tf.keras.layers.Activation('linear',name="STm_i_cb")(cb_1)
    cr_crop=tf.keras.layers.Activation('linear',name="STm_i_cr")(cr_1)
    
    if DOG_init:
        conv1_lisse_y = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_yDoG_h")(y_crop)
        conv1_lisse_y = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_yDoG")(conv1_lisse_y)
        
        conv1_lisse_cb = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_cbDoG_h")(cb_crop)
        conv1_lisse_cb = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_cbDoG")(conv1_lisse_cb)  
        
        conv1_lisse_cr = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_crDoG_h")(cr_crop)
        conv1_lisse_cr = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_crDoG")(conv1_lisse_cr)       
    else:
        conv1_lisse_y = tf.keras.layers.Activation("linear",name="STm_i_yDoG")(y_crop)
        conv1_lisse_cb = tf.keras.layers.Activation("linear",name="STm_i_cbDoG")(cb_crop)
        conv1_lisse_cr = tf.keras.layers.Activation("linear",name="STm_i_crDoG")(cr_crop)
        
    if BN_init:
        conv1_lisse_y = tf.keras.layers.BatchNormalization(name="STm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb = tf.keras.layers.BatchNormalization(name="STm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr = tf.keras.layers.BatchNormalization(name="STm_i_crDoGBn")(conv1_lisse_cr)
    else:
        conv1_lisse_y=tf.keras.layers.Activation("linear",name="STm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb=tf.keras.layers.Activation("linear",name="STm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr=tf.keras.layers.Activation("linear",name="STm_i_crDoGBn")(conv1_lisse_cr)
        
    concat_ycbcr=concatenate([conv1_lisse_y,conv1_lisse_cb,conv1_lisse_cr],axis=3, name="concat_rgb_bn_style") 
    conv1=Conv2D(filters=filtres_branch, kernel_size=nombre_kernel, strides=1, padding="same",name="T_conv_0_base")(concat_ycbcr)
    conv1 = tf.keras.layers.BatchNormalization(name="T_BN_0_base")(conv1)
    conv1 = tf.keras.layers.Activation('relu',name="T_Act_0_base")(conv1)

    T1 = Tb(conv1,filtres_branch,nombre_kernel,name_i=0)
    T2 = Tb(T1,filtres_branch,nombre_kernel,name_i=1)
    T3 = Tb(T2,filtres_branch,nombre_kernel,name_i=2)
    T4 = Tb(T3,filtres_branch,nombre_kernel,name_i=3)
    
    residu_y = Conv2D(filters=1, kernel_size=1, strides=1, padding="same", name="gather_st",activation="relu")(T4)#,activation="relu"
    y_out=tf.keras.layers.Activation("linear",name="STm_o_y")(residu_y)
    if DOG_fin:   
        y_out = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_o_yDoG_h")(y_out)
        y_out = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_o_yDoG")(y_out)
    else:
        y_out = tf.keras.layers.Activation('linear',name="STm_o_yDoG")(y_out)
        
    if BN_fin:
        y_out = tf.keras.layers.BatchNormalization(name="STm_o_yDoGBn")(y_out)
    else:
        y_out = tf.keras.layers.Activation('linear',name="STm_o_yDoGBn")(y_out)
    
    # Tanh
    S=tf.keras.layers.Activation("tanh")(y_out) 
    
    # Median filter - Custom 
    kernel_median = tf.expand_dims(tf.expand_dims(tf.ones((2,2)),axis=-1),axis=-1)
    sum_S = Conv2D(1, (kernel_median.shape[0]),trainable=False,padding="same",weights=[kernel_median],use_bias=False,name="sum_s")(S)
    
    max_S=tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same",name="max_S")(S)
    min_S=tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same",name="min_S")(-S)
    
    S=(sum_S-max_S+min_S)/2
    S=tf.keras.layers.Activation("linear",name="STm_o_yFinal_residu")(S)

    #Fb
    S=tf.keras.layers.BatchNormalization(name="F_BN_outputi_y_style_fb", trainable=True, scale=False,center=False)(S)
    S  = S*(tf.math.sqrt(tf.math.reduce_variance(SR_y)))/2  

    # Residual
    y_out = concatenate([S,SR_y],axis=3, name = "concatenated_ST_y") 
    y_out=tf.reduce_sum(y_out,axis=3,name="ST_y") 
    y_out = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="youtexapnd")(y_out)
    ST=tf.keras.layers.Activation("linear",name="STm_o_yFinal")(y_out)
    
    # TO RGB FOR VGG
    ST_ycbcr=concatenate([ST,cb_crop,cr_crop],axis=3, name = "ST_ycbcr") 
    ST_ycbcr=tf.keras.layers.Activation("linear",name="ST_ycbcr_reshape")(ST_ycbcr)

    ST_rgb = Lambda(lambda x: tensor_ycbcr2rgb(x)/255.,name="ST_rgb")(ST_ycbcr)    
    ST_rgb = Lambda(lambda x: tf.keras.backend.clip(x, 0, 1), name = "ST")(ST_rgb) 
     
    new_model = Model(model.inputs, ST_rgb)    

    new_model.compile(optimizer = Adam(lr=1), loss = [mse_loss1], loss_weights=[1], metrics = [PSNR],run_eagerly=True)
    return(new_model)


def STy_residual_branch_none(filtres_branch:int, border:int, DOG_init:bool, DOG_fin: bool, BN_init:bool,BN_fin:bool,nombre_kernel:int,w_h_s,w_v_s):   
    '''
    Tensorflow (keras API) for Sty branch trained on the top of pre-trained MAIN model ; and (None,None,3) size input tensor (USED FOR INFERING ON ANY IMAGE)
    ''' 
    input_network_border = Input((None,None,3))
        
    y_1 = Lambda(lambda x: x[:,:,:,0],name="y_cropped")(input_network_border) 
    y_1 = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="y_reshape")(y_1)
        
    cb_1= Lambda(lambda x: x[:,:,:,1], name="cb_cropped")(input_network_border)
    cb_1 = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="cb_reshape")(cb_1)

    cr_1= Lambda(lambda x: x[:,:,:,2], name="cr_cropped")(input_network_border)
    cr_1 = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="cr_reshape")(cr_1)

    y_crop=tf.keras.layers.Activation('linear',name="STm_i_y")(y_1)
    cb_crop=tf.keras.layers.Activation('linear',name="STm_i_cb")(cb_1)
    cr_crop=tf.keras.layers.Activation('linear',name="STm_i_cr")(cr_1)
    
    if DOG_init:
        conv1_lisse_y = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_yDoG_h")(y_crop)
        conv1_lisse_y = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_yDoG")(conv1_lisse_y)
        
        conv1_lisse_cb = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_cbDoG_h")(cb_crop)
        conv1_lisse_cb = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_cbDoG")(conv1_lisse_cb)  
        
        conv1_lisse_cr = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_crDoG_h")(cr_crop)
        conv1_lisse_cr = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_crDoG")(conv1_lisse_cr)       
    else:
        conv1_lisse_y = tf.keras.layers.Activation("linear",name="STm_i_yDoG")(y_crop)
        conv1_lisse_cb = tf.keras.layers.Activation("linear",name="STm_i_cbDoG")(cb_crop)
        conv1_lisse_cr = tf.keras.layers.Activation("linear",name="STm_i_crDoG")(cr_crop)
        
    if BN_init:
        conv1_lisse_y = tf.keras.layers.BatchNormalization(name="STm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb = tf.keras.layers.BatchNormalization(name="STm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr = tf.keras.layers.BatchNormalization(name="STm_i_crDoGBn")(conv1_lisse_cr)
    else:
        conv1_lisse_y=tf.keras.layers.Activation("linear",name="STm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb=tf.keras.layers.Activation("linear",name="STm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr=tf.keras.layers.Activation("linear",name="STm_i_crDoGBn")(conv1_lisse_cr)
        
    concat_ycbcr=concatenate([conv1_lisse_y,conv1_lisse_cb,conv1_lisse_cr],axis=3, name="concat_rgb_bn_style") 
    conv1=Conv2D(filters=filtres_branch, kernel_size=nombre_kernel, strides=1, padding="same",name="T_conv_0_base")(concat_ycbcr)
    conv1 = tf.keras.layers.BatchNormalization(name="T_BN_0_base")(conv1)
    conv1 = tf.keras.layers.Activation('relu',name="T_Act_0_base")(conv1)

    T1 = Tb(conv1,filtres_branch,nombre_kernel,name_i=0)
    T2 = Tb(T1,filtres_branch,nombre_kernel,name_i=1)
    T3 = Tb(T2,filtres_branch,nombre_kernel,name_i=2)
    T4 = Tb(T3,filtres_branch,nombre_kernel,name_i=3)
    
    residu_y = Conv2D(filters=1, kernel_size=1, strides=1, padding="same", name="gather_st",activation="relu")(T4)
    y_out=tf.keras.layers.Activation("linear",name="STm_o_y")(residu_y)
    if DOG_fin:   
        y_out = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_o_yDoG_h")(y_out)
        y_out = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_o_yDoG")(y_out)
    else:
        y_out = tf.keras.layers.Activation('linear',name="STm_o_yDoG")(y_out)
        
    if BN_fin:
        y_out = tf.keras.layers.BatchNormalization(name="STm_o_yDoGBn")(y_out)
    else:
        y_out = tf.keras.layers.Activation('linear',name="STm_o_yDoGBn")(y_out)
    
    # Tanh
    S=tf.keras.layers.Activation("tanh")(y_out) 
    
    # Median filter - Custom 
    kernel_median = tf.expand_dims(tf.expand_dims(tf.ones((2,2)),axis=-1),axis=-1)
    sum_S = Conv2D(1, (kernel_median.shape[0]),trainable=False,padding="same",weights=[kernel_median],use_bias=False,name="sum_s")(S)
    
    max_S=tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same",name="max_S")(S)
    min_S=tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same",name="min_S")(-S)
    
    S=(sum_S-max_S+min_S)/2
    S=tf.keras.layers.Activation("linear",name="STm_o_yFinal_residu")(S)

    #Fb
    S=tf.keras.layers.BatchNormalization(name="F_BN_outputi_y_style_fb", trainable=True, scale=False,center=False)(S) 
    S  = S*(tf.math.sqrt(tf.math.reduce_variance(y_1)))/2  
     
    new_model = Model(input_network_border, S)    
    new_model.compile(optimizer = Adam(lr=1), loss = [mse_loss1], loss_weights=[1], metrics = [PSNR],run_eagerly=True)
    return(new_model)
    
# St(y) BRANCH WITH COLOR PRED --------------------------------------------------------------------------
def STy_branch_col(model,main_network,taille_inputx:int, taille_inputy:int,filtres_branch:int, border:int, DOG_init:bool, DOG_fin: bool, BN_init:bool,BN_fin:bool,nombre_kernel:int,w_h_s,w_v_s,style_gram=0,loss_pixel=True, loss_perceptuelle=[], loss_style=[], ponderation_pixel=1.0, ponderation_perc=1.0, ponderation_style=1.0  ,profondeur=1, learning_rate=0.001):   
    '''
    Tensorflow (keras API) for Stycbcr branch trained on the top of pre-trained MAIN model  ; and (sizex,sizey,3) size input tensor (USED FOR TRAINING ON GIVEN PATCHES WITH DEFINED SIZES)    
    ''' 
    # Main model features loading
    if main_network=="SR_EDSR":
        input_network_border = Input((int(taille_inputx/4),int(taille_inputy/4),3))
        m=model.layers[-1].output            
        SR= Lambda(lambda x: (x)/255.,name="output_edsr_rgb_01.")(m)
        SR = Lambda(lambda x: tf_rgb2ycbcr(x)/255.,name="output_edsr_ycbcr_01")(SR)    
        
        SR_y = Lambda(lambda x: x[:,:,:,0],name="SR_y")(SR) 
        SR_y = Reshape((taille_inputx,taille_inputy,1), name = "SR_y_reshape")(SR_y) 
        y_1=tf.keras.layers.Activation('linear',name="copy")(SR_y)
        
        cb_1= Lambda(lambda x: x[:,:,:,1], name="cb_edsr")(SR)
        cb_1=tf.keras.layers.Reshape((taille_inputx,taille_inputy,1), name="cb_edsr_reshape")(cb_1)
    
        cr_1= Lambda(lambda x: x[:,:,:,2], name="cr_edsr")(SR)
        cr_1=tf.keras.layers.Reshape((taille_inputx,taille_inputy,1), name="cr_edsr_reshape")(cr_1)
    else:
        y_1 = model.get_layer('bicubic_crop').output
        cb_1 = model.get_layer('cb_input_crop').output
        cr_1 = model.get_layer('cr_input_crop').output
    
        SR = model.get_layer('SR').output
        SR_y = Lambda(lambda x: x[:,:,:,0],name="SR_y")(SR) 
        SR_y = Reshape((taille_inputx,taille_inputy,1), name = "SR_y_reshape")(SR_y) 
    
    y_crop=tf.keras.layers.Activation('linear',name="STm_i_y")(y_1)
    cb_crop=tf.keras.layers.Activation('linear',name="STm_i_cb")(cb_1)
    cr_crop=tf.keras.layers.Activation('linear',name="STm_i_cr")(cr_1)
    
    if DOG_init:
        conv1_lisse_y = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_yDoG_h")(y_crop)
        conv1_lisse_y = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_yDoG")(conv1_lisse_y)
        
        conv1_lisse_cb = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_cbDoG_h")(cb_crop)
        conv1_lisse_cb = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_cbDoG")(conv1_lisse_cb)  
        
        conv1_lisse_cr = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_crDoG_h")(cr_crop)
        conv1_lisse_cr = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_crDoG")(conv1_lisse_cr)       
    else:
        conv1_lisse_y = tf.keras.layers.Activation("linear",name="STm_i_yDoG")(y_crop)
        conv1_lisse_cb = tf.keras.layers.Activation("linear",name="STm_i_cbDoG")(cb_crop)
        conv1_lisse_cr = tf.keras.layers.Activation("linear",name="STm_i_crDoG")(cr_crop)
        
    if BN_init:
        conv1_lisse_y = tf.keras.layers.BatchNormalization(name="STm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb = tf.keras.layers.BatchNormalization(name="STm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr = tf.keras.layers.BatchNormalization(name="STm_i_crDoGBn")(conv1_lisse_cr)
    else:
        conv1_lisse_y=tf.keras.layers.Activation("linear",name="STm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb=tf.keras.layers.Activation("linear",name="STm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr=tf.keras.layers.Activation("linear",name="STm_i_crDoGBn")(conv1_lisse_cr)
        
    concat_ycbcr=concatenate([conv1_lisse_y,conv1_lisse_cb,conv1_lisse_cr],axis=3, name="concat_rgb_bn_style") 
    conv1=Conv2D(filters=filtres_branch, kernel_size=nombre_kernel, strides=1, padding="same",name="T_conv_0_base")(concat_ycbcr)
    conv1 = tf.keras.layers.BatchNormalization(name="T_BN_0_base")(conv1)
    conv1 = tf.keras.layers.Activation('relu',name="T_Act_0_base")(conv1)
    
    # Architecture originale St(y) ; pas d'auto encodeur car c'est de la SR
    T1 = Tb(conv1,filtres_branch,nombre_kernel,name_i=0)
    T2 = Tb(T1,filtres_branch,nombre_kernel,name_i=1)
    T3 = Tb(T2,filtres_branch,nombre_kernel,name_i=2)
    T4 = Tb(T3,filtres_branch,nombre_kernel,name_i=3)
    
    residu_y = Conv2D(filters=3, kernel_size=1, strides=1, padding="same", name="gather_st",activation="relu")(T4) #T4  upconv2
    
    y_out = Lambda(lambda x: x[:,:,:,0],name="y_out")(residu_y) 
    y_out = Reshape((taille_inputx,taille_inputy,1),name="STm_o_y")(y_out)
    
    cb_out = Lambda(lambda x: x[:,:,:,1],name="cb_out")(residu_y) 
    cb_out = Reshape((taille_inputx,taille_inputy,1),name="STm_o_cb")(cb_out)
    cb_out=tf.keras.layers.BatchNormalization(name="BN_cb")(cb_out)
    cb_out=tf.keras.layers.Activation("sigmoid")(cb_out) 

    cr_out = Lambda(lambda x: x[:,:,:,2],name="cr_out")(residu_y) 
    cr_out = Reshape((taille_inputx,taille_inputy,1),name="STm_o_cr")(cr_out)
    cr_out=tf.keras.layers.BatchNormalization(name="BN_cr")(cr_out)
    cr_out=tf.keras.layers.Activation("sigmoid")(cr_out) 
    
    if DOG_fin:   
        y_out = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_o_yDoG_h")(y_out)
        y_out = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_o_yDoG")(y_out)
    else:
        y_out = tf.keras.layers.Activation('linear',name="STm_o_yDoG")(y_out)
    if BN_fin:
        y_out = tf.keras.layers.BatchNormalization(name="STm_o_yDoGBn")(y_out)
    else:
        y_out = tf.keras.layers.Activation('linear',name="STm_o_yDoGBn")(y_out)
    
    # Tanh
    S=tf.keras.layers.Activation("tanh")(y_out) 
   
    # Median filter - custom 
    kernel_median = tf.expand_dims(tf.expand_dims(tf.ones((2,2)),axis=-1),axis=-1)
    sum_S = Conv2D(1, (kernel_median.shape[0]),trainable=False,padding="same",weights=[kernel_median],use_bias=False,name="sum_s")(S)
    
    max_S=tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same",name="max_S")(S)
    min_S=tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same",name="min_S")(-S)
    
    S=(sum_S-max_S+min_S)/2
    S=tf.keras.layers.Activation("linear",name="STm_o_yFinal_residu")(S)
    
    #Fb
    S=tf.keras.layers.BatchNormalization(name="F_BN_outputi_y_style_fb", trainable=True, scale=False,center=False)(S) #LayerNormalization
    S  = S*(tf.math.sqrt(tf.math.reduce_variance(SR_y)))/2
    
    # Residual
    y_out = concatenate([S,SR_y],axis=3, name = "concatenated_ST_y") 
    y_out=tf.reduce_sum(y_out,axis=3,name="ST_y") 
    ST=Reshape((taille_inputx,taille_inputy,1),name="STm_o_yFinal")(y_out)
    
    # TO RGB FOR VGG
    ST_ycbcr=concatenate([ST,cb_out,cr_out],axis=3, name = "ST_ycbcr") 
    ST_ycbcr = tf.keras.layers.Reshape((taille_inputx,taille_inputy,3),name="ST_ycbcr_reshape")(ST_ycbcr)

    ST_rgb = Lambda(lambda x: tensor_ycbcr2rgb(x)/255.,name="ST_rgb")(ST_ycbcr)    
    ST_rgb= Reshape((taille_inputx,taille_inputy,3),name="ST_rgb_reshape")(ST_rgb)
    ST_rgb = Lambda(lambda x: tf.keras.backend.clip(x, 0, 1), name = "ST")(ST_rgb)  #/ retiré si histogram matching
    
    # normalization true
    loss_list,outputs,weight_list = compilation_StyleTransfer(True,loss_perceptuelle, loss_style, loss_pixel,ponderation_pixel, ponderation_perc, ponderation_style, ST_rgb, style_gram)  
    new_model = Model(model.inputs, outputs)    

    new_model.compile(optimizer = Adam(lr=learning_rate), loss = loss_list, loss_weights=weight_list, metrics = [PSNR],run_eagerly=True)
    return(new_model,loss_list)
      
def STy_branch_col_none(model, main_network, filtres_branch:int, border:int, DOG_init:bool, DOG_fin: bool, BN_init:bool,BN_fin:bool,nombre_kernel:int,w_h_s,w_v_s):   
    '''
    Tensorflow (keras API) for StyCBCR branch trained on the top of pre-trained MAIN model ; and (None,None,3) size input tensor (USED FOR INFERING ON ANY IMAGE)
    ''' 
    # Main model features loading
    if main_network=="SR_EDSR":
        input_network_border = Input((None,None,3))
        m=model.layers[-1].output            
        SR= Lambda(lambda x: (x)/255.,name="output_edsr_rgb_01.")(m)
        SR = Lambda(lambda x: tf_rgb2ycbcr(x)/255.,name="output_edsr_ycbcr_01")(SR)    
        
        SR_y = Lambda(lambda x: x[:,:,:,0],name="SR_y")(SR) 
        SR_y = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="SR_y_reshape")(SR_y)
        y_1=tf.keras.layers.Activation('linear',name="copy")(SR_y)
        
        cb_1= Lambda(lambda x: x[:,:,:,1], name="cb_edsr")(SR)
        cb_1 = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="cb_edsr_reshape")(cb_1)

        cr_1= Lambda(lambda x: x[:,:,:,2], name="cr_edsr")(SR)
        cr_1 = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="cr_edsr_reshape")(cr_1)
        
    else:
        y_1 = model.get_layer('bicubic_crop').output
        cb_1 = model.get_layer('cb_input_crop').output
        cr_1 = model.get_layer('cr_input_crop').output
    
        SR = model.get_layer('SR').output
        SR_y = Lambda(lambda x: x[:,:,:,0],name="SR_y")(SR) 
        SR_y = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="SR_y_reshape")(SR_y)

    y_crop=tf.keras.layers.Activation('linear',name="STm_i_y")(y_1)
    cb_crop=tf.keras.layers.Activation('linear',name="STm_i_cb")(cb_1)
    cr_crop=tf.keras.layers.Activation('linear',name="STm_i_cr")(cr_1)
    
    if DOG_init:
        conv1_lisse_y = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_yDoG_h")(y_crop)
        conv1_lisse_y = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_yDoG")(conv1_lisse_y)
        
        conv1_lisse_cb = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_cbDoG_h")(cb_crop)
        conv1_lisse_cb = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_cbDoG")(conv1_lisse_cb)  
        
        conv1_lisse_cr = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_crDoG_h")(cr_crop)
        conv1_lisse_cr = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_crDoG")(conv1_lisse_cr)       
    else:
        conv1_lisse_y = tf.keras.layers.Activation("linear",name="STm_i_yDoG")(y_crop)
        conv1_lisse_cb = tf.keras.layers.Activation("linear",name="STm_i_cbDoG")(cb_crop)
        conv1_lisse_cr = tf.keras.layers.Activation("linear",name="STm_i_crDoG")(cr_crop)
        
    if BN_init:
        conv1_lisse_y = tf.keras.layers.BatchNormalization(name="STm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb = tf.keras.layers.BatchNormalization(name="STm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr = tf.keras.layers.BatchNormalization(name="STm_i_crDoGBn")(conv1_lisse_cr)
    else:
        conv1_lisse_y=tf.keras.layers.Activation("linear",name="STm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb=tf.keras.layers.Activation("linear",name="STm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr=tf.keras.layers.Activation("linear",name="STm_i_crDoGBn")(conv1_lisse_cr)
        
    concat_ycbcr=concatenate([conv1_lisse_y,conv1_lisse_cb,conv1_lisse_cr],axis=3, name="concat_rgb_bn_style") 
    conv1=Conv2D(filters=filtres_branch, kernel_size=nombre_kernel, strides=1, padding="same",name="T_conv_0_base")(concat_ycbcr)
    conv1 = tf.keras.layers.BatchNormalization(name="T_BN_0_base")(conv1)
    conv1 = tf.keras.layers.Activation('relu',name="T_Act_0_base")(conv1)

    # Original main architecture St(y) 
    T1 = Tb(conv1,filtres_branch,nombre_kernel,name_i=0)
    T2 = Tb(T1,filtres_branch,nombre_kernel,name_i=1)
    T3 = Tb(T2,filtres_branch,nombre_kernel,name_i=2)
    T4 = Tb(T3,filtres_branch,nombre_kernel,name_i=3)

    residu_y = Conv2D(filters=3, kernel_size=1, strides=1, padding="same", name="gather_st",activation="relu")(T4)
    
    y_out = Lambda(lambda x: x[:,:,:,0],name="y_out")(residu_y) 
    y_out = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="STm_o_y")(y_out)
    
    cb_out = Lambda(lambda x: x[:,:,:,1],name="cb_out")(residu_y) 
    cb_out = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="STm_o_cb")(cb_out)
    cb_out=tf.keras.layers.Activation("sigmoid")(cb_out) 
    
    cr_out = Lambda(lambda x: x[:,:,:,2],name="cr_out")(residu_y) 
    cr_out = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="STm_o_cr")(cr_out)
    cr_out=tf.keras.layers.Activation("sigmoid")(cr_out) 
   
    if DOG_fin:   
        y_out = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_o_yDoG_h")(y_out)
        y_out = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_o_yDoG")(y_out)
    else:
        y_out = tf.keras.layers.Activation('linear',name="STm_o_yDoG")(y_out)
        
    if BN_fin:
        y_out = tf.keras.layers.BatchNormalization(name="STm_o_yDoGBn")(y_out)
    else:
        y_out = tf.keras.layers.Activation('linear',name="STm_o_yDoGBn")(y_out)
    
    # Tanh
    S=tf.keras.layers.Activation("tanh")(y_out) 
    
    # Median filter - Custom 
    kernel_median = tf.expand_dims(tf.expand_dims(tf.ones((2,2)),axis=-1),axis=-1)
    sum_S = Conv2D(1, (kernel_median.shape[0]),trainable=False,padding="same",weights=[kernel_median],use_bias=False,name="sum_s")(S)
    
    max_S=tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same",name="max_S")(S)
    min_S=tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same",name="min_S")(-S)
    
    S=(sum_S-max_S+min_S)/2
    S=tf.keras.layers.Activation("linear",name="STm_o_yFinal_residu")(S)
    

    #Fb
    S=tf.keras.layers.BatchNormalization(name="F_BN_outputi_y_style_fb", trainable=True, scale=False,center=False)(S)
    S  = S*(tf.math.sqrt(tf.math.reduce_variance(SR_y)))/2  # standart deviation needs to be reduced for the details not to saturate and all style to learn the same magnitude of details

    # Residual
    y_out = concatenate([S,SR_y],axis=3, name = "concatenated_ST_y") 
    y_out=tf.reduce_sum(y_out,axis=3,name="ST_y") 
    y_out = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="youtexapnd")(y_out)
    ST=tf.keras.layers.Activation("linear",name="STm_o_yFinal")(y_out)
    
    # TO RGB FOR VGG
    ST_ycbcr=concatenate([ST,cb_out,cr_out],axis=3, name = "ST_ycbcr") 

    ST_ycbcr=tf.keras.layers.Activation("linear",name="ST_ycbcr_reshape")(ST_ycbcr)


    ST_rgb = Lambda(lambda x: tensor_ycbcr2rgb(x)/255.,name="ST_rgb")(ST_ycbcr)    
    ST_rgb = Lambda(lambda x: tf.keras.backend.clip(x, 0, 1), name = "ST")(ST_rgb) 
     
    new_model = Model(model.inputs, ST_rgb)    

    new_model.compile(optimizer = Adam(lr=1), loss = [mse_loss1], loss_weights=[1], metrics = [PSNR],run_eagerly=True)
    return(new_model) 


def STy_residual_branch_col_none(filtres_branch:int, border:int, DOG_init:bool, DOG_fin: bool, BN_init:bool,BN_fin:bool,nombre_kernel:int,w_h_s,w_v_s):   
    '''
    Tensorflow (keras API) for StyCBCR branch trained on the top of pre-trained MAIN model ; and (None,None,3) size input tensor (USED FOR INFERING ON ANY IMAGE)
    ''' 
    input_network_border = Input((None,None,3))
        
    y_1 = Lambda(lambda x: x[:,:,:,0],name="SR_y")(input_network_border) 
    y_1 = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="SR_y_reshape")(y_1)
        
    cb_1= Lambda(lambda x: x[:,:,:,1], name="cb_edsr")(input_network_border)
    cb_1 = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="cb_edsr_reshape")(cb_1)

    cr_1= Lambda(lambda x: x[:,:,:,2], name="cr_edsr")(input_network_border)
    cr_1 = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="cr_edsr_reshape")(cr_1)

    y_crop=tf.keras.layers.Activation('linear',name="STm_i_y")(y_1)
    cb_crop=tf.keras.layers.Activation('linear',name="STm_i_cb")(cb_1)
    cr_crop=tf.keras.layers.Activation('linear',name="STm_i_cr")(cr_1)
    
    if DOG_init:
        conv1_lisse_y = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_yDoG_h")(y_crop)
        conv1_lisse_y = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_yDoG")(conv1_lisse_y)
        
        conv1_lisse_cb = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_cbDoG_h")(cb_crop)
        conv1_lisse_cb = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_cbDoG")(conv1_lisse_cb)  
        
        conv1_lisse_cr = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_i_crDoG_h")(cr_crop)
        conv1_lisse_cr = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_i_crDoG")(conv1_lisse_cr)       
    else:
        conv1_lisse_y = tf.keras.layers.Activation("linear",name="STm_i_yDoG")(y_crop)
        conv1_lisse_cb = tf.keras.layers.Activation("linear",name="STm_i_cbDoG")(cb_crop)
        conv1_lisse_cr = tf.keras.layers.Activation("linear",name="STm_i_crDoG")(cr_crop)
        
    if BN_init:
        conv1_lisse_y = tf.keras.layers.BatchNormalization(name="STm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb = tf.keras.layers.BatchNormalization(name="STm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr = tf.keras.layers.BatchNormalization(name="STm_i_crDoGBn")(conv1_lisse_cr)
    else:
        conv1_lisse_y=tf.keras.layers.Activation("linear",name="STm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb=tf.keras.layers.Activation("linear",name="STm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr=tf.keras.layers.Activation("linear",name="STm_i_crDoGBn")(conv1_lisse_cr)
        
    concat_ycbcr=concatenate([conv1_lisse_y,conv1_lisse_cb,conv1_lisse_cr],axis=3, name="concat_rgb_bn_style") 
    conv1=Conv2D(filters=filtres_branch, kernel_size=nombre_kernel, strides=1, padding="same",name="T_conv_0_base")(concat_ycbcr)
    conv1 = tf.keras.layers.BatchNormalization(name="T_BN_0_base")(conv1)
    conv1 = tf.keras.layers.Activation('relu',name="T_Act_0_base")(conv1)

    # Original main architecture St(y) 
    T1 = Tb(conv1,filtres_branch,nombre_kernel,name_i=0)
    T2 = Tb(T1,filtres_branch,nombre_kernel,name_i=1)
    T3 = Tb(T2,filtres_branch,nombre_kernel,name_i=2)
    T4 = Tb(T3,filtres_branch,nombre_kernel,name_i=3)

    residu_y = Conv2D(filters=3, kernel_size=1, strides=1, padding="same", name="gather_st",activation="relu")(T4)
    
    y_out = Lambda(lambda x: x[:,:,:,0],name="y_out")(residu_y) 
    y_out = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="STm_o_y")(y_out)
    
    cb_out = Lambda(lambda x: x[:,:,:,1],name="cb_out")(residu_y) 
    cb_out = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="STm_o_cb")(cb_out)
    cb_out=tf.keras.layers.Activation("sigmoid")(cb_out) 
    
    cr_out = Lambda(lambda x: x[:,:,:,2],name="cr_out")(residu_y) 
    cr_out = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="STm_o_cr")(cr_out)
    cr_out=tf.keras.layers.Activation("sigmoid")(cr_out) 
   
    if DOG_fin:   
        y_out = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="STm_o_yDoG_h")(y_out)
        y_out = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="STm_o_yDoG")(y_out)
    else:
        y_out = tf.keras.layers.Activation('linear',name="STm_o_yDoG")(y_out)
    if BN_fin:
        y_out = tf.keras.layers.BatchNormalization(name="STm_o_yDoGBn")(y_out)
    else:
        y_out = tf.keras.layers.Activation('linear',name="STm_o_yDoGBn")(y_out)
    
    # Tanh
    S=tf.keras.layers.Activation("tanh")(y_out) 
    
    # Median filter - Custom 
    kernel_median = tf.expand_dims(tf.expand_dims(tf.ones((2,2)),axis=-1),axis=-1)
    sum_S = Conv2D(1, (kernel_median.shape[0]),trainable=False,padding="same",weights=[kernel_median],use_bias=False,name="sum_s")(S)
    
    max_S=tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same",name="max_S")(S)
    min_S=tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same",name="min_S")(-S)
    
    S=(sum_S-max_S+min_S)/2
    S=tf.keras.layers.Activation("linear",name="STm_o_yFinal_residu")(S)
    
    #Fb
    S=tf.keras.layers.BatchNormalization(name="F_BN_outputi_y_style_fb", trainable=True, scale=False,center=False)(S) 
    S  = S*(tf.math.sqrt(tf.math.reduce_variance(y_1)))/2  
     
    new_model = Model(input_network_border, [S,cb_out,cr_out])    
    new_model.compile(optimizer = Adam(lr=1), loss = [mse_loss1,mse_loss1,mse_loss1], loss_weights=[1,1,1], metrics = [PSNR],run_eagerly=True)
    return(new_model) 
    



