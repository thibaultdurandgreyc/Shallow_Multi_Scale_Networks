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
from various_functions.tensor_tf_data_fonctions import *

from Networks.Compilation_NN import *

'''
Branch for Color transfer . Transfer Colours from a style image into the SR image. Not residual
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
def STrcol_branch(model,main_network,taille_inputx:int,taille_inputy:int, filtres_branch:int, border:int, DOG_init:bool, DOG_fin: bool, BN_init:bool,BN_fin:bool,nombre_kernel:int,clusters:int,w_h_s,w_v_s,hist_style=0,style=0, loss_pixel=True, loss_perceptuelle=[], loss_style=[], ponderation_pixel=1.0, ponderation_perc=1.0, ponderation_style=1.0  ,profondeur=1, learning_rate=0.001):   
    '''
    Tensorflow (keras API) for COL branch trained on the top of pre-trained MAIN model ; and (sizex,sizey,3) size input tensor (USED FOR TRAINING ON GIVEN PATCHES WITH DEFINED SIZES)
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
    
    y_crop=tf.keras.layers.Activation('linear',name="COLm_i_y")(y_1)
    cb_crop=tf.keras.layers.Activation('linear',name="COLm_i_cb")(cb_1)
    cr_crop=tf.keras.layers.Activation('linear',name="COLm_i_cr")(cr_1)
    
    if DOG_init:
        conv1_lisse_y = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="COLm_i_yDoG_h")(y_crop)
        conv1_lisse_y = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="COLm_i_yDoG")(conv1_lisse_y)
        
        conv1_lisse_cb = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="COLm_i_cbDoG_h")(cb_crop)
        conv1_lisse_cb = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="COLm_i_cbDoG")(conv1_lisse_cb)  
        
        conv1_lisse_cr = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="COLm_i_crDoG_h")(cr_crop)
        conv1_lisse_cr = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="COLm_i_crDoG")(conv1_lisse_cr)       
    else:
        conv1_lisse_y = tf.keras.layers.Activation("linear",name="COLm_i_yDoG")(y_crop)
        conv1_lisse_cb = tf.keras.layers.Activation("linear",name="COLm_i_cbDoG")(cb_crop)
        conv1_lisse_cr = tf.keras.layers.Activation("linear",name="COLm_i_crDoG")(cr_crop)
        
    if BN_init:
        conv1_lisse_y=tf.keras.layers.BatchNormalization(name="COLm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb=tf.keras.layers.BatchNormalization(name="COLm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr=tf.keras.layers.BatchNormalization(name="COLm_i_crDoGBn")(conv1_lisse_cr)
    else:
        conv1_lisse_y=tf.keras.layers.Activation("linear",name="COLm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb=tf.keras.layers.Activation("linear",name="COLm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr=tf.keras.layers.Activation("linear",name="COLm_i_crDoGBn")(conv1_lisse_cr)
    
    concat_ycbcr=concatenate([conv1_lisse_y,conv1_lisse_cb,conv1_lisse_cr],axis=3, name="concat_ycbcr_bn_col") 
    conv1=Conv2D(filters=filtres_branch, kernel_size=nombre_kernel, padding="same",name="T_conv_0_base")(concat_ycbcr)
    conv1 = tf.keras.layers.BatchNormalization(name="T_BN_0_base")(conv1)
    conv1 = tf.keras.layers.Activation('relu',name="T_Act_0_base")(conv1)

    T1 = Tb(conv1,filtres_branch,nombre_kernel,name_i=0)
    T2 = Tb(T1,filtres_branch,nombre_kernel,name_i=1)
    T3 = Tb(T2,filtres_branch,nombre_kernel,name_i=2)
    T4 = Tb(T3,filtres_branch,nombre_kernel,name_i=3)
    
    residu = Conv2D(2, 1, padding="same",activation="relu", name="gather_col")(T4)
    
    cb_i= Lambda(lambda x: x[:,:,:,0],name="residu_cb")(residu)#concat_ycbcr
    cb=tf.keras.layers.Reshape((taille_inputx,taille_inputy,1),name="residu_cb_reshape")(cb_i)
    cb=tf.keras.layers.BatchNormalization(name="BN_cb")(cb)
    conv1_f_cb=tf.keras.layers.Activation('linear',name="COLm_o_cb")(cb)

    cr_i= Lambda(lambda x: x[:,:,:,1],name="residu_cr")(residu)
    cr=tf.keras.layers.Reshape((taille_inputx,taille_inputy,1),name="residu_cr_reshape")(cr_i)
    cr=tf.keras.layers.BatchNormalization(name="BN_cr")(cr)
    conv1_f_cr=tf.keras.layers.Activation('linear',name="COLm_o_cr")(cr)    
    
    if DOG_fin:
        conv1_f_cb = tf.keras.layers.Activation('linear',name="COLm_o_cbDoG")(conv1_f_cb)
        conv1_f_cr = tf.keras.layers.Activation('linear',name="COLm_o_crDoG")(conv1_f_cr)
    else:
        conv1_f_cb = tf.keras.layers.Activation('linear',name="COLm_o_cbDoG")(conv1_f_cb)
        conv1_f_cr = tf.keras.layers.Activation('linear',name="COLm_o_crDoG")(conv1_f_cr)
    if BN_fin:
        conv1_f_cb = tf.keras.layers.Activation('linear',name="COLm_o_cbDoGBn")(conv1_f_cb)
        conv1_f_cr = tf.keras.layers.Activation('linear',name="COLm_o_crDoGBn")(conv1_f_cr)
        
    else:
        conv1_f_cb = tf.keras.layers.Activation('linear',name="COLm_o_cbDoGBn")(conv1_f_cb)
        conv1_f_cr = tf.keras.layers.Activation('linear',name="COLm_o_crDoGBn")(conv1_f_cr)
        
    cb_out = tf.keras.layers.Activation('sigmoid',name="COLm_o_cbFinal_residu")(conv1_f_cb)
    cr_out = tf.keras.layers.Activation('sigmoid',name="COLm_o_crFinal_residu")(conv1_f_cr)
    
    # Residual
    cb_out=Reshape((int(taille_inputx),int(taille_inputy),1),name="COLm_o_cbFinal")(cb_out)    
    cr_out=Reshape((int(taille_inputx),int(taille_inputy),1),name="COLm_o_crFinal")(cr_out)
    
    # TO RGB FOR VGG
    COL_tr_ycbcr=concatenate([SR_y,cb_out,cr_out],axis=3, name = "COL_ycbcr") 
    COL_tr_ycbcr = tf.keras.layers.Reshape((taille_inputx,taille_inputy,3),name="COL_ycbcr_reshape")(COL_tr_ycbcr)

    COL_tr_rgb = Lambda(lambda x: tensor_ycbcr2rgb(x)/255.,name="COL_rgb")(COL_tr_ycbcr)    
    COL_tr_rgb= Reshape((taille_inputx,taille_inputy,3),name="COL_rgb_reshape")(COL_tr_rgb)
    COL_tr_rgb = Lambda(lambda x: tf.keras.backend.clip(x, 0, 1), name = "COL_rgb_clip")(COL_tr_rgb)  # ??
     
    loss_list,outputs,weight_list = compilation_ColTransfer(False,loss_perceptuelle, loss_style, loss_pixel,ponderation_pixel, ponderation_perc, ponderation_style, COL_tr_rgb, hist_style,clusters)  
    new_model = Model(model.inputs, outputs)    

    new_model.compile(optimizer = Adam(lr=learning_rate), loss = loss_list, loss_weights=weight_list, metrics = [PSNR],run_eagerly=True)
    return(new_model,loss_list)
    

def STrcol_branch_none(model,main_network,filtres_branch:int, border:int, DOG_init:bool, DOG_fin: bool, BN_init:bool,BN_fin:bool,nombre_kernel:int,w_h_s,w_v_s):   
    '''
    Tensorflow (keras API) for COL branch trained on the top of pre-trained MAIN model ; and (None,None,3) size input tensor (USED FOR INFERING ON ANY IMAGE)
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

    y_crop=tf.keras.layers.Activation('linear',name="COLm_i_y")(y_1)
    cb_crop=tf.keras.layers.Activation('linear',name="COLm_i_cb")(cb_1)
    cr_crop=tf.keras.layers.Activation('linear',name="COLm_i_cr")(cr_1)
    
    if DOG_init:
        conv1_lisse_y = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="COLm_i_yDoG_h")(y_crop)
        conv1_lisse_y = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="COLm_i_yDoG")(conv1_lisse_y)
        
        conv1_lisse_cb = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="COLm_i_cbDoG_h")(cb_crop)
        conv1_lisse_cb = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="COLm_i_cbDoG")(conv1_lisse_cb)  
        
        conv1_lisse_cr = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="COLm_i_crDoG_h")(cr_crop)
        conv1_lisse_cr = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="COLm_i_crDoG")(conv1_lisse_cr)       
    else:
        conv1_lisse_y = tf.keras.layers.Activation("linear",name="COLm_i_yDoG")(y_crop)
        conv1_lisse_cb = tf.keras.layers.Activation("linear",name="COLm_i_cbDoG")(cb_crop)
        conv1_lisse_cr = tf.keras.layers.Activation("linear",name="COLm_i_crDoG")(cr_crop)
        
    if BN_init:
        conv1_lisse_y=tf.keras.layers.BatchNormalization(name="COLm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb=tf.keras.layers.BatchNormalization(name="COLm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr=tf.keras.layers.BatchNormalization(name="COLm_i_crDoGBn")(conv1_lisse_cr)
    else:
        conv1_lisse_y=tf.keras.layers.Activation("linear",name="COLm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb=tf.keras.layers.Activation("linear",name="COLm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr=tf.keras.layers.Activation("linear",name="COLm_i_crDoGBn")(conv1_lisse_cr)
    
    concat_ycbcr=concatenate([conv1_lisse_y,conv1_lisse_cb,conv1_lisse_cr],axis=3, name="concat_ycbcr_bn_col") 
    conv1=Conv2D(filters=filtres_branch, kernel_size=nombre_kernel, padding="same",name="T_conv_0_base")(concat_ycbcr)
    conv1 = tf.keras.layers.BatchNormalization(name="T_BN_0_base")(conv1)
    conv1 = tf.keras.layers.Activation('relu',name="T_Act_0_base")(conv1)

    T1 = Tb(conv1,filtres_branch,nombre_kernel,name_i=0)
    T2 = Tb(T1,filtres_branch,nombre_kernel,name_i=1)
    T3 = Tb(T2,filtres_branch,nombre_kernel,name_i=2)
    T4 = Tb(T3,filtres_branch,nombre_kernel,name_i=3)
    
    residu = Conv2D(2, 1, padding="same",activation="relu", name="gather_col")(T4)
    
    cb_i= Lambda(lambda x: x[:,:,:,0],name="residu_cb")(residu)
    cb_i = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="residu_cb_reshape")(cb_i)
    cb_i=tf.keras.layers.BatchNormalization(name="BN_cb")(cb_i)
    conv1_f_cb=tf.keras.layers.Activation('linear',name="COLm_o_cb")(cb_i)

    cr_i= Lambda(lambda x: x[:,:,:,1],name="residu_cr")(residu)
    cr_i = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="residu_cr_reshape")(cr_i)
    cr_i=tf.keras.layers.BatchNormalization(name="BN_cr")(cr_i)
    conv1_f_cr=tf.keras.layers.Activation('linear',name="COLm_o_cr")(cr_i)    
    
    if DOG_fin:
        conv1_f_cb = tf.keras.layers.Activation('linear',name="COLm_o_cbDoG")(conv1_f_cb)
        conv1_f_cr = tf.keras.layers.Activation('linear',name="COLm_o_crDoG")(conv1_f_cr)
    else:
        conv1_f_cb = tf.keras.layers.Activation('linear',name="COLm_o_cbDoG")(conv1_f_cb)
        conv1_f_cr = tf.keras.layers.Activation('linear',name="COLm_o_crDoG")(conv1_f_cr)
    if BN_fin:
        conv1_f_cb = tf.keras.layers.Activation('linear',name="COLm_o_cbDoGBn")(conv1_f_cb)
        conv1_f_cr = tf.keras.layers.Activation('linear',name="COLm_o_crDoGBn")(conv1_f_cr)
    else:
        conv1_f_cb = tf.keras.layers.Activation('linear',name="COLm_o_cbDoGBn")(conv1_f_cb)
        conv1_f_cr = tf.keras.layers.Activation('linear',name="COLm_o_crDoGBn")(conv1_f_cr)
    
    cb_out = tf.keras.layers.Activation('sigmoid',name="COLm_o_cbFinal_residu")(conv1_f_cb)
    cr_out = tf.keras.layers.Activation('sigmoid',name="COLm_o_crFinal_residu")(conv1_f_cr)
    
    cb_out = tf.keras.layers.Activation('linear',name="COLm_o_cbFinal")(cb_out)
    cb_out = Lambda(lambda x: tf.keras.backend.clip(x,0,1), name = "cb_clip")(cb_out) #0.063,0.9411

    cr_out = tf.keras.layers.Activation('linear',name="COLm_o_crFinal")(cr_out)
    cr_out = Lambda(lambda x: tf.keras.backend.clip(x, 0,1), name = "cr_clip")(cr_out) 
    
    # TO RGB FOR VGG
    COL_tr_ycbcr=concatenate([SR_y,cb_out,cr_out],axis=3, name = "COL_ycbcr") 
    COL_tr_rgb = Lambda(lambda x: tensor_ycbcr2rgb(x)/255.,name="COL_rgb")(COL_tr_ycbcr)    
    COL_tr_rgb = Lambda(lambda x: tf.keras.backend.clip(x, 0, 1), name = "COL_rgb_clip")(COL_tr_rgb)  # ??
     
    new_model = Model(model.inputs, COL_tr_rgb)    
    new_model.compile(optimizer = Adam(lr=1), loss = [mse_loss1], loss_weights=[1], metrics = [PSNR],run_eagerly=True)
    return(new_model)    
    
def STrcol_residual_branch_none(filtres_branch:int, border:int, DOG_init:bool, DOG_fin: bool, BN_init:bool,BN_fin:bool,nombre_kernel:int,w_h_s,w_v_s):   
    '''
    Tensorflow (keras API) for COL branch trained on the top of pre-trained MAIN model ; and (None,None,3) size input tensor (USED FOR INFERING ON ANY IMAGE)
    ''' 
    
    input_network_border = Input((None,None,3))
        
    y_1 = Lambda(lambda x: x[:,:,:,0],name="y_cropped")(input_network_border) 
    y_1 = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="SR_y_reshape")(y_1)
        
    cb_1= Lambda(lambda x: x[:,:,:,1], name="cb_cropped")(input_network_border)
    cb_1 = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="cb_edsr_reshape")(cb_1)

    cr_1= Lambda(lambda x: x[:,:,:,2], name="cr_cropped")(input_network_border)
    cr_1 = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="cr_edsr_reshape")(cr_1)

    y_crop=tf.keras.layers.Activation('linear',name="COLm_i_y")(y_1)
    cb_crop=tf.keras.layers.Activation('linear',name="COLm_i_cb")(cb_1)
    cr_crop=tf.keras.layers.Activation('linear',name="COLm_i_cr")(cr_1)
    
    if DOG_init:
        conv1_lisse_y = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="COLm_i_yDoG_h")(y_crop)
        conv1_lisse_y = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="COLm_i_yDoG")(conv1_lisse_y)
        
        conv1_lisse_cb = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="COLm_i_cbDoG_h")(cb_crop)
        conv1_lisse_cb = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="COLm_i_cbDoG")(conv1_lisse_cb)  
        
        conv1_lisse_cr = Conv2D(1, (w_h_s[1].shape[0],1),trainable=False,padding="same",weights=[(w_h_s[1]-w_h_s[0]).reshape(w_h_s[1].shape[0],1,1,1)],use_bias=False,name="COLm_i_crDoG_h")(cr_crop)
        conv1_lisse_cr = Conv2D(1, (1,w_v_s[1].shape[1]),trainable=False,padding="same",weights=[(w_v_s[1]-w_v_s[0]).reshape(1,w_v_s[1].shape[1],1,1)],use_bias=False,name="COLm_i_crDoG")(conv1_lisse_cr)       
    else:
        conv1_lisse_y = tf.keras.layers.Activation("linear",name="COLm_i_yDoG")(y_crop)
        conv1_lisse_cb = tf.keras.layers.Activation("linear",name="COLm_i_cbDoG")(cb_crop)
        conv1_lisse_cr = tf.keras.layers.Activation("linear",name="COLm_i_crDoG")(cr_crop)
        
    if BN_init:
        conv1_lisse_y=tf.keras.layers.BatchNormalization(name="COLm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb=tf.keras.layers.BatchNormalization(name="COLm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr=tf.keras.layers.BatchNormalization(name="COLm_i_crDoGBn")(conv1_lisse_cr)
    else:
        conv1_lisse_y=tf.keras.layers.Activation("linear",name="COLm_i_yDoGBn")(conv1_lisse_y)
        conv1_lisse_cb=tf.keras.layers.Activation("linear",name="COLm_i_cbDoGBn")(conv1_lisse_cb)
        conv1_lisse_cr=tf.keras.layers.Activation("linear",name="COLm_i_crDoGBn")(conv1_lisse_cr)
    
    concat_ycbcr=concatenate([conv1_lisse_y,conv1_lisse_cb,conv1_lisse_cr],axis=3, name="concat_ycbcr_bn_col") 
    conv1=Conv2D(filters=filtres_branch, kernel_size=nombre_kernel, padding="same",name="T_conv_0_base")(concat_ycbcr)
    conv1 = tf.keras.layers.BatchNormalization(name="T_BN_0_base")(conv1)
    conv1 = tf.keras.layers.Activation('relu',name="T_Act_0_base")(conv1)

    T1 = Tb(conv1,filtres_branch,nombre_kernel,name_i=0)
    T2 = Tb(T1,filtres_branch,nombre_kernel,name_i=1)
    T3 = Tb(T2,filtres_branch,nombre_kernel,name_i=2)
    T4 = Tb(T3,filtres_branch,nombre_kernel,name_i=3)
    
    residu = Conv2D(2, 1, padding="same",activation="relu", name="gather_col")(T4)
    
    
    cb_i= Lambda(lambda x: x[:,:,:,0],name="residu_cb")(residu)
    cb_i = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="residu_cb_reshape")(cb_i)
    cb_i=tf.keras.layers.BatchNormalization(name="BN_cb")(cb_i)
    conv1_f_cb=tf.keras.layers.Activation('linear',name="COLm_o_cb")(cb_i)

    cr_i= Lambda(lambda x: x[:,:,:,1],name="residu_cr")(residu)
    cr_i = Lambda(lambda x: tf.expand_dims(x,axis=-1),name="residu_cr_reshape")(cr_i)
    cr_i=tf.keras.layers.BatchNormalization(name="BN_cr")(cr_i)
    conv1_f_cr=tf.keras.layers.Activation('linear',name="COLm_o_cr")(cr_i)    
    
    if DOG_fin:
        conv1_f_cb = tf.keras.layers.Activation('linear',name="COLm_o_cbDoG")(conv1_f_cb)
        conv1_f_cr = tf.keras.layers.Activation('linear',name="COLm_o_crDoG")(conv1_f_cr)
    else:
        conv1_f_cb = tf.keras.layers.Activation('linear',name="COLm_o_cbDoG")(conv1_f_cb)
        conv1_f_cr = tf.keras.layers.Activation('linear',name="COLm_o_crDoG")(conv1_f_cr)
    if BN_fin:
        conv1_f_cb = tf.keras.layers.Activation('linear',name="COLm_o_cbDoGBn")(conv1_f_cb)
        conv1_f_cr = tf.keras.layers.Activation('linear',name="COLm_o_crDoGBn")(conv1_f_cr)
        
    else:
        conv1_f_cb = tf.keras.layers.Activation('linear',name="COLm_o_cbDoGBn")(conv1_f_cb)
        conv1_f_cr = tf.keras.layers.Activation('linear',name="COLm_o_crDoGBn")(conv1_f_cr)
        
    # softmax (no residual)
    cb_out = tf.keras.layers.Activation('sigmoid',name="COLm_o_cbFinal_residu")(conv1_f_cb)
    cr_out = tf.keras.layers.Activation('sigmoid',name="COLm_o_crFinal_residu")(conv1_f_cr)
    

    cb_out = tf.keras.layers.Activation('linear',name="COLm_o_cbFinal")(cb_out)
    cb_out = Lambda(lambda x: tf.keras.backend.clip(x,0,1), name = "cb_clip")(cb_out) 

    cr_out = tf.keras.layers.Activation('linear',name="COLm_o_crFinal")(cr_out)
    cr_out = Lambda(lambda x: tf.keras.backend.clip(x, 0,1), name = "cr_clip")(cr_out) 
     
    new_model = Model(input_network_border, [cb_out,cr_out])    
    new_model.compile(optimizer = Adam(lr=1), loss = [mse_loss1,mse_loss1], loss_weights=[1,1], metrics = [PSNR],run_eagerly=True)
    return(new_model)   
   





    
    