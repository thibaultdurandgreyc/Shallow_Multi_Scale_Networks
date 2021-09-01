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

from various_functions.tensor_tf_data_fonctions import *
from various_functions.rapports_pdf_fonctions import *
from various_functions.numpy_data_fonctions import *
from Networks.model_management import *
from various_functions.directory_management import *
from various_functions.custom_filters import *
from various_functions.tensor_tf_data_fonctions import *

import tensorflow_addons as tfa
from Networks.Compilation_NN import *
from Networks.Loss_Constructor import *

'''
Functions for building MAIN NN ( SR / DENOISING / BLURRING)
'''

# a. Module
def Rb(x,nb,nombre_kernel,name_i):
    '''
    Tensorflow (keras API) R module for MAIN NN
    ''' 
    Conv2D_1 = Conv2D(filters=nb, kernel_size=nombre_kernel, strides=1, padding="same",name="R_conv_"+str(name_i), kernel_initializer = tf.keras.initializers.GlorotUniform())(x)
    BN1 = tf.keras.layers.BatchNormalization(name="R_Bn_"+str(name_i))(Conv2D_1)
    res = tf.keras.layers.Activation('relu',name="R_Act_"+str(name_i))(BN1)
    #res = concatenate([res,x],axis=nombre_kernel,name="R_"+str(name_i))
    res = Add(name="R_"+str(name_i))([res,x])
    return(res)


# b. Core NN    
def MAIN_network(sizex:int,sizey:int,w_h,w_v, ponderation_features=[0.5,0.9,0.9,1.1,1.4,1.3],nombre_class=6, filtres=12, kernel=3,loss_pixel=True, loss_perceptuelle=[5,9,13], loss_style=[],ponderation_pixel=1.0, ponderation_perc=1.0, ponderation_style=1.0,  learning_rate=0.001,BN_init=True,BN_fin=True,DOG_init=True,DOG_fin=True):
    '''
    Tensorflow (keras API) for MAIN NN build with MAIN BRANCHES and (sizex,sizey,3) size input tensor (USED FOR TRAINING ON GIVEN PATCHES WITH DEFINED SIZES)
    '''
    
    border=15  

    input_network_border = Input((sizex+2*border,sizey+2*border,3))
    
    y = Lambda(lambda x: x[:,:,:,0])(input_network_border)
    y=tf.keras.layers.Reshape((sizex+2*border,sizey+2*border,1))(y)
    y=tf.keras.layers.Activation('linear',name="SRm_i_y")(y)

    cb= Lambda(lambda x: x[:,:,:,1])(input_network_border)
    cb=tf.keras.layers.Reshape((sizex+2*border,sizey+2*border,1))(cb)
    cb=tf.keras.layers.Activation('linear',name="SRm_i_cb")(cb)

    cr = Lambda(lambda x: x[:,:,:,2])(input_network_border)
    cr=tf.keras.layers.Reshape((sizex+2*border,sizey+2*border,1))(cr)
    cr=tf.keras.layers.Activation('linear',name="SRm_i_cr")(cr)
    
    y_crop=tf.keras.layers.Cropping2D(cropping=((border,border),(border,border)),input_shape=((sizex+2*border,sizey+2*border,1)),name="lambda_y")(y)
    cb_crop=tf.keras.layers.Cropping2D(cropping=((border,border),(border,border)),input_shape=((sizex+2*border,sizey+2*border,1)),name="lambda_cb")(cb)
    cr_crop=tf.keras.layers.Cropping2D(cropping=((border,border),(border,border)),input_shape=((sizex+2*border,sizey+2*border,1)),name="lambda_cr")(cr)
    
    bicubic_crop=tf.keras.layers.Reshape((sizex,sizey,1),name="bicubic_crop")(y_crop)
    cb_crop=tf.keras.layers.Reshape((sizex,sizey,1), name="cb_input_crop")(cb_crop)
    cr_crop=tf.keras.layers.Reshape((sizex,sizey,1),name = "cr_input_crop")(cr_crop)
    
    filters=[]           
    conv_1f=tf.map_fn(fn=lambda chan: MAIN_branch(filters,ponderation_features[chan],sizex,sizey,border,chan,filtres,kernel,BN_init,BN_fin,DOG_init,DOG_fin,nombre_class, w_h,w_v,y,cb,cr,cb_crop,cr_crop,bicubic_crop) , elems = tf.range(0, nombre_class, 1,  dtype=tf.float32) )

    filters.append(bicubic_crop)         
    details = concatenate(filters,axis=3, name = "concatenated_SRi")
 
    end=tf.reduce_sum(details,axis=3) 
    end= Reshape((sizex,sizey,1),name="out_SRy")(end)
    
    end=concatenate([end,cb_crop,cr_crop],axis=3, name = "concatenated_SR")
    end= Reshape((sizex,sizey,3),name='out_SR')(end)
    end = Lambda(lambda x: tf.keras.backend.clip(x, 0, 1),name="SR")(end) # CLIPInG VALUES

    loss_list,outputs,weight_list = compilation_MAIN_network(loss_perceptuelle, loss_style, loss_pixel, ponderation_pixel, ponderation_perc, ponderation_style,end)
    model = Model(input_network_border, [outputs])           
    opt=Adam(lr=learning_rate,clipvalue=0.5, name="AdamOpp")
    model.compile(optimizer = opt, loss = loss_list, loss_weights=weight_list, metrics = [PSNR],run_eagerly=True)
    return(model,loss_list,opt)   
    
# b2. Core NN    SR
def MAIN_network_none(w_h,w_v, ponderation_features=[0.5,0.9,0.9,1.1,1.4,1.3],nombre_class=6, filtres=12, kernel=3,   R=4,BN_init=True,BN_fin=True,DOG_init=True,DOG_fin=True):
    '''  
    Tensorflow (keras API) for MAIN NN built with MAIN branches and (None,None,3) size input tensor (USED FOR INFERING ON ANY IMAGE)
    https://github.com/keras-team/keras/issues/12679
    '''
    border=15  

    input_network_border = Input((None,None,3))
    
    y = Lambda(lambda x: x[:,:,:,0])(input_network_border)
    y = Lambda(lambda x: tf.expand_dims(x,axis=-1))(y)
    y=tf.keras.layers.Activation('linear',name="SRm_i_y")(y)

    cb= Lambda(lambda x: x[:,:,:,1])(input_network_border)
    cb = Lambda(lambda x: tf.expand_dims(x,axis=-1))(cb)
    cb=tf.keras.layers.Activation('linear',name="SRm_i_cb")(cb)

    cr = Lambda(lambda x: x[:,:,:,2])(input_network_border)
    cr = Lambda(lambda x: tf.expand_dims(x,axis=-1))(cr)
    cr=tf.keras.layers.Activation('linear',name="SRm_i_cr")(cr)
    
    bicubic_crop=tf.keras.layers.Cropping2D(cropping=((border,border),(border,border)),name="bicubic_crop")(y)
    cb_crop=tf.keras.layers.Cropping2D(cropping=((border,border),(border,border)),name="cb_input_crop")(cb)
    cr_crop=tf.keras.layers.Cropping2D(cropping=((border,border),(border,border)),name="cr_input_crop")(cr)
   
    filters=[]       
    conv_1f=tf.map_fn(fn=lambda chan: MAIN_branch_none(filters,ponderation_features[chan],border,chan,filtres,kernel,BN_init,BN_fin,DOG_init,DOG_fin,nombre_class, w_h,w_v,y,cb,cr,cb_crop,cr_crop,bicubic_crop) , elems = tf.range(0, nombre_class, 1,  dtype=tf.float32) )

    filters.append(bicubic_crop)         
    details = concatenate(filters,axis=3, name = "concatenated_SRi")
 
    end=tf.reduce_sum(details,axis=3) 
    end = Lambda(lambda x: tf.expand_dims(x,axis=-1))(end)
    end=tf.keras.layers.Activation('linear',name="out_SRy")(end)

    end=concatenate([end,cb_crop,cr_crop],axis=3, name = "concatenated_SR")
    end=tf.keras.layers.Activation('linear',name='out_SR')(end)
    end = Lambda(lambda x: tf.keras.backend.clip(x, 0, 1),name="SR")(end) # CLIPInG VALUES
        
    model = Model([input_network_border], end)           
    opt=Adam(lr=1,clipvalue=0.5, name="AdamOpp")
    model.compile(optimizer = opt, loss = [mse_loss1], loss_weights=[1], metrics = [PSNR],run_eagerly=True) # model needs to be compiled in keras
    return(model)   


# c. Branch
def MAIN_branch_none(filters,ponderation:float,border:int,channel:int, filtres:int,kernel:int, BN_init:bool,BN_fin:bool,DOG_init:bool,DOG_fin:bool, nombre_class:int,w_h, w_v,  y, cb, cr ,cb_crop, cr_crop, bicubic_crop):
    '''
    Tensorflow (keras API) for MAIN branch and (None,None,3) size input tensor (USED FOR INFERING ON ANY IMAGE)
    ''' 
    filtres=int(ponderation*filtres)
    
    if DOG_init:
        conv1_lisse_y = Conv2D(1, (w_h[2*channel].shape[0],1),trainable=False,padding="same",weights=[(w_h[2*channel+1]-w_h[2*channel]).reshape(w_h[2*channel].shape[0],1,1,1)],use_bias=False,name="SRm_i_yDoG_h"+str(int(channel.numpy())))(y)
        conv1_lisse_y = Conv2D(1, (1,w_v[2*channel].shape[1]),trainable=False,padding="same",weights=[(w_v[2*channel+1]-w_v[2*channel]).reshape(1,w_v[2*channel].shape[1],1,1)],use_bias=False,name="SRm_i_yDoG"+str(int(channel.numpy())))(conv1_lisse_y)
        
        conv1_lisse_cb = Conv2D(1, (w_h[2*channel].shape[0],1),trainable=False,padding="same",weights=[(w_h[2*channel+1]-w_h[2*channel]).reshape(w_h[2*channel].shape[0],1,1,1)],use_bias=False,name="SRm_i_cbDoG_h"+str(int(channel.numpy())))(cb)
        conv1_lisse_cb = Conv2D(1, (1,w_v[2*channel].shape[1]),trainable=False,padding="same",weights=[(w_v[2*channel+1]-w_v[2*channel]).reshape(1,w_v[2*channel].shape[1],1,1)],use_bias=False,name="SRm_i_cbDoG"+str(int(channel.numpy())))(conv1_lisse_cb)  
        
        conv1_lisse_cr = Conv2D(1, (w_h[2*channel].shape[0],1),trainable=False,padding="same",weights=[(w_h[2*channel+1]-w_h[2*channel]).reshape(w_h[2*channel].shape[0],1,1,1)],use_bias=False,name="SRm_i_crDoG_h"+str(int(channel.numpy())))(cr)
        conv1_lisse_cr = Conv2D(1, (1,w_v[2*channel].shape[1]),trainable=False,padding="same",weights=[(w_v[2*channel+1]-w_v[2*channel]).reshape(1,w_v[2*channel].shape[1],1,1)],use_bias=False,name="SRm_i_crDoG"+str(int(channel.numpy())))(conv1_lisse_cr)        
        
    else:
        conv1_lisse_y = tf.keras.layers.Activation("linear",name="SRm_i_yDoG"+str(int(channel.numpy())))(y)
        conv1_lisse_cb = tf.keras.layers.Activation("linear",name="SRm_i_cbDoG"+str(int(channel.numpy())))(cb)
        conv1_lisse_cr = tf.keras.layers.Activation("linear",name="SRm_i_crDoG"+str(int(channel.numpy())))(cr)
               
    if BN_init:
        conv1_lisse_y = tf.keras.layers.BatchNormalization(name="SRm_i_yDoGBn"+str(int(channel.numpy())))(conv1_lisse_y)
        conv1_lisse_cb = tf.keras.layers.BatchNormalization(name="SRm_i_cbDoGBn"+str(int(channel.numpy())))(conv1_lisse_cb)
        conv1_lisse_cr = tf.keras.layers.BatchNormalization(name="SRm_i_crDoGBn"+str(int(channel.numpy())))(conv1_lisse_cr)
    else:
        conv1_lisse_y = tf.keras.layers.Activation("linear",name="SRm_i_yDoGBn"+str(int(channel.numpy())))(conv1_lisse_y)
        conv1_lisse_cb = tf.keras.layers.Activation("linear",name="SRm_i_cbDoGBn"+str(int(channel.numpy())))(conv1_lisse_cb)
        conv1_lisse_cr = tf.keras.layers.Activation("linear",name="SRm_i_crDoGBn"+str(int(channel.numpy())))(conv1_lisse_cr)
        
    conv1=concatenate([conv1_lisse_y,conv1_lisse_cb,conv1_lisse_cr],axis=3, name="concat_"+str(int(channel.numpy())))  
    conv1=Conv2D(filters=filtres, kernel_size=kernel, strides=1, padding="same",name="R_conv_0_base"+str(int(channel.numpy())), kernel_initializer = tf.keras.initializers.GlorotUniform())(conv1) #retirer le activation relu
    conv1 = tf.keras.layers.BatchNormalization(name="R_BN_0_base"+str(int(channel.numpy())))(conv1)
    conv1 = tf.keras.layers.Activation('relu',name="R_Act_0_base"+str(int(channel.numpy())))(conv1)
    
    Rb1 = Rb(conv1,filtres,kernel,"0_channel"+str(int(channel.numpy())))
    Rb2 = Rb(Rb1,filtres,kernel,"1_channel"+str(int(channel.numpy())))
    Rb3 = Rb(Rb2,filtres,kernel,"2_channel"+str(int(channel.numpy())))
    
    SR_i = Conv2D(1, 1,  name="SRi"+str(int(channel.numpy())), padding="same")(Rb3)   
    conv1_f=tf.keras.layers.Cropping2D(name="SRm_o_y"+str(int(channel.numpy())),cropping=((border,border),(border,border)))(SR_i)

    if DOG_fin:
        conv1_f = Conv2D(1, (w_h[2*channel].shape[0],1),trainable=False,padding="same",weights=[(w_h[2*channel+1]-w_h[2*channel]).reshape(w_h[2*channel].shape[0],1,1,1)],use_bias=False,name="SRm_o_yDoG_h"+str(int(channel.numpy())))(conv1_f)
        conv1_f = Conv2D(1, (1,w_v[2*channel].shape[1]),trainable=False,padding="same",weights=[(w_v[2*channel+1]-w_v[2*channel]).reshape(1,w_v[2*channel].shape[1],1,1)],use_bias=False,name="SRm_o_yDoG"+str(int(channel.numpy())))(conv1_f)
    
    else:
        conv1_f = tf.keras.layers.Activation('linear',name="SRm_o_yDoG"+str(int(channel.numpy())))(conv1_f)
    
    if BN_fin:
        conv1_f = tf.keras.layers.BatchNormalization(name="SRm_o_yDoGBn"+str(int(channel.numpy())))(conv1_f)
    
    conv1_f = tf.keras.layers.Activation('tanh',name="tanh"+str(int(channel.numpy())))(conv1_f) # OPTIONNEL
    
    if ((nombre_class-1)==channel) : 
        conv1_f = Lambda(lambda x: tfa.image.median_filter2d(x,filter_shape=[2,2]),name="median_filter_highScale_SR"+str(int(channel.numpy())))(conv1_f) # BRYAN
    
    conv1_f = tf.keras.layers.Activation('linear',name="SRm_o_yFinal"+str(int(channel.numpy())))(conv1_f)
    
    filters.append(conv1_f) 
    return(conv1_f)


def MAIN_branch(filters,ponderation:float,sizex:int,sizey:int,border:int,channel:int, filtres:int,kernel:int, BN_init:bool,BN_fin:bool,DOG_init:bool,DOG_fin:bool, nombre_class:int,w_h, w_v,  y, cb, cr ,cb_crop, cr_crop, bicubic_crop):
    '''
    Tensorflow (keras API) for MAIN branch and (sizex,sizey,3) size input tensor (USED FOR TRAINING ON GIVEN PATCHES WITH DEFINED SIZES)
    ''' 

    filtres=int(ponderation*filtres)

    if DOG_init:
        
        #NOYAU 1D
        conv1_lisse_y = Conv2D(1, (w_h[2*channel].shape[0],1),trainable=False,padding="same",weights=[(w_h[2*channel+1]-w_h[2*channel]).reshape(w_h[2*channel].shape[0],1,1,1)],use_bias=False,name="SRm_i_yDoG_h"+str(int(channel.numpy())))(y)
        conv1_lisse_y = Conv2D(1, (1,w_v[2*channel].shape[1]),trainable=False,padding="same",weights=[(w_v[2*channel+1]-w_v[2*channel]).reshape(1,w_v[2*channel].shape[1],1,1)],use_bias=False,name="SRm_i_yDoG"+str(int(channel.numpy())))(conv1_lisse_y)
        
        conv1_lisse_cb = Conv2D(1, (w_h[2*channel].shape[0],1),trainable=False,padding="same",weights=[(w_h[2*channel+1]-w_h[2*channel]).reshape(w_h[2*channel].shape[0],1,1,1)],use_bias=False,name="SRm_i_cbDoG_h"+str(int(channel.numpy())))(cb)
        conv1_lisse_cb = Conv2D(1, (1,w_v[2*channel].shape[1]),trainable=False,padding="same",weights=[(w_v[2*channel+1]-w_v[2*channel]).reshape(1,w_v[2*channel].shape[1],1,1)],use_bias=False,name="SRm_i_cbDoG"+str(int(channel.numpy())))(conv1_lisse_cb)  
        
        conv1_lisse_cr = Conv2D(1, (w_h[2*channel].shape[0],1),trainable=False,padding="same",weights=[(w_h[2*channel+1]-w_h[2*channel]).reshape(w_h[2*channel].shape[0],1,1,1)],use_bias=False,name="SRm_i_crDoG_h"+str(int(channel.numpy())))(cr)
        conv1_lisse_cr = Conv2D(1, (1,w_v[2*channel].shape[1]),trainable=False,padding="same",weights=[(w_v[2*channel+1]-w_v[2*channel]).reshape(1,w_v[2*channel].shape[1],1,1)],use_bias=False,name="SRm_i_crDoG"+str(int(channel.numpy())))(conv1_lisse_cr)        
    else:
        conv1_lisse_y = tf.keras.layers.Activation("linear",name="SRm_i_yDoG"+str(int(channel.numpy())))(y)
        conv1_lisse_cb = tf.keras.layers.Activation("linear",name="SRm_i_cbDoG"+str(int(channel.numpy())))(cb)
        conv1_lisse_cr = tf.keras.layers.Activation("linear",name="SRm_i_crDoG"+str(int(channel.numpy())))(cr)
               
    if BN_init:
        conv1_lisse_y = tf.keras.layers.BatchNormalization(name="SRm_i_yDoGBn"+str(int(channel.numpy())))(conv1_lisse_y)
        conv1_lisse_cb = tf.keras.layers.BatchNormalization(name="SRm_i_cbDoGBn"+str(int(channel.numpy())))(conv1_lisse_cb)
        conv1_lisse_cr = tf.keras.layers.BatchNormalization(name="SRm_i_crDoGBn"+str(int(channel.numpy())))(conv1_lisse_cr)
    else:
        conv1_lisse_y = tf.keras.layers.Activation("linear",name="SRm_i_yDoGBn"+str(int(channel.numpy())))(conv1_lisse_y)
        conv1_lisse_cb = tf.keras.layers.Activation("linear",name="SRm_i_cbDoGBn"+str(int(channel.numpy())))(conv1_lisse_cb)
        conv1_lisse_cr = tf.keras.layers.Activation("linear",name="SRm_i_crDoGBn"+str(int(channel.numpy())))(conv1_lisse_cr)
        
    conv1=concatenate([conv1_lisse_y,conv1_lisse_cb,conv1_lisse_cr],axis=3, name="concat_"+str(int(channel.numpy())))  
    conv1=Conv2D(filters=filtres, kernel_size=kernel, strides=1, padding="same",name="R_conv_0_base"+str(int(channel.numpy())), kernel_initializer = tf.keras.initializers.GlorotUniform())(conv1) #retirer le activation relu
    conv1 = tf.keras.layers.BatchNormalization(name="R_BN_0_base"+str(int(channel.numpy())))(conv1)
    conv1 = tf.keras.layers.Activation('relu',name="R_Act_0_base"+str(int(channel.numpy())))(conv1)
    
    Rb1 = Rb(conv1,filtres,kernel,"0_channel"+str(int(channel.numpy())))
    Rb2 = Rb(Rb1,filtres,kernel,"1_channel"+str(int(channel.numpy())))
    Rb3 = Rb(Rb2,filtres,kernel,"2_channel"+str(int(channel.numpy())))
    
    SR_i = Conv2D(1, 1,  name="SRi"+str(int(channel.numpy())), padding="same")(Rb3)   
    conv1_f=tf.keras.layers.Cropping2D(name="SRm_o_y"+str(int(channel.numpy())),cropping=((border,border),(border,border)),input_shape=((sizex+2*border,sizey+2*border,1)))(SR_i)

    if DOG_fin:
        conv1_f = Conv2D(1, (w_h[2*channel].shape[0],1),trainable=False,padding="same",weights=[(w_h[2*channel+1]-w_h[2*channel]).reshape(w_h[2*channel].shape[0],1,1,1)],use_bias=False,name="SRm_o_yDoG_h"+str(int(channel.numpy())))(conv1_f)
        conv1_f = Conv2D(1, (1,w_v[2*channel].shape[1]),trainable=False,padding="same",weights=[(w_v[2*channel+1]-w_v[2*channel]).reshape(1,w_v[2*channel].shape[1],1,1)],use_bias=False,name="SRm_o_yDoG"+str(int(channel.numpy())))(conv1_f)
    else:
        conv1_f = tf.keras.layers.Activation('linear',name="SRm_o_yDoG"+str(int(channel.numpy())))(conv1_f)
    
    if BN_fin:
        conv1_f = tf.keras.layers.BatchNormalization(name="SRm_o_yDoGBn"+str(int(channel.numpy())))(conv1_f)
    
    conv1_f = tf.keras.layers.Activation('tanh',name="tanh"+str(int(channel.numpy())))(conv1_f) # OPTIONNEL
    
    if ((nombre_class-1)==channel) : 
        conv1_f = Lambda(lambda x: tfa.image.median_filter2d(x,filter_shape=[2,2]),name="median_filter_highScale_SR"+str(int(channel.numpy())))(conv1_f) # BRYAN
        conv1_f = Reshape((sizex,sizey,1))(conv1_f)
    
    conv1_f = tf.keras.layers.Activation('linear',name="SRm_o_yFinal"+str(int(channel.numpy())))(conv1_f)
    
    filters.append(conv1_f) 

    return(conv1_f)









