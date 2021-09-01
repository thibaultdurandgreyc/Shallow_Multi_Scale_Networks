from __future__ import division

import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow.keras.optimizers import *
import numpy as np

'''
Modules for Computing specific feature outputs for a specific input data
'''
# 1. Functions used for extracting features ---
def extract_layer_output(full_model,nbre_classe,output):
    """
    Extract feature output when layer output is composed of 1 single feature
    """
    model_new = tf.compat.v1.keras.models.Model([full_model.input], [full_model.get_layer(output+str(c)).output for c in range(0,nbre_classe)])#conv1final_class_
    model_new.compile(optimizer=Adam(),loss='mse',metrics=['mse'])#
    return(model_new)

def extract_layer(full_model,output):
    """
    Extract features output when layer output is composed of multiple features single feature
    """
    model_new = tf.compat.v1.keras.models.Model([full_model.input], [full_model.get_layer(output).output])#conv1final_class_
    model_new.compile(optimizer=Adam(),loss='mse',metrics=['mse'])#
    return(model_new)
    
# 2. Ablation models extraction ---  
def extract_layers_STr_col(model,BN_init,BN_fin,DOG_init,DOG_fin):
    '''
    Extract layers from ST branch specialized on y channel ("STcol")
    '''
    COLm_i_y,COLm_i_cb,COLm_i_cr, COLm_i_yDoG,COLm_i_cbDoG,COLm_i_crDoG,   COLm_i_yDoGBn,COLm_i_cbDoGBn,COLm_i_crDoGBn ,COLm_o_cb,COLm_o_cbDoG,COLm_o_cbDoGBn,COLm_o_cbFinal,COLm_o_cbFinalresidu=None,None,None,None,None,None,None,None,None,None,None,None,None,None
    COLm_o_cr,COLm_o_crDoG,COLm_o_crDoGBn,COLm_o_crFinal,COLm_o_crFinalresidu=None,None,None,None,None
    
    COLm_i_y=extract_layer(model,"COLm_i_y")
    COLm_i_cb=extract_layer(model,"COLm_i_cb") 
    COLm_i_cr=extract_layer(model,"COLm_i_cr") 
    if BN_init:
        COLm_i_yDoGBn=extract_layer(model,"COLm_i_yDoGBn") 
        COLm_i_cbDoGBn=extract_layer(model,"COLm_i_cbDoGBn") 
        COLm_i_crDoGBn=extract_layer(model,"COLm_i_crDoGBn")  
    if DOG_init:
        COLm_i_yDoG=extract_layer(model,"COLm_i_yDoG") 
        COLm_i_cbDoG=extract_layer(model,"COLm_i_cbDoG") 
        COLm_i_crDoG=extract_layer(model,"COLm_i_crDoG")    
        
    COLm_o_cb=extract_layer(model,"COLm_o_cb")    
    COLm_o_cr=extract_layer(model,"COLm_o_cr")    
    if DOG_fin:
        COLm_o_cbDoG=extract_layer(model,"COLm_o_cbDoG") 
        COLm_o_crDoG=extract_layer(model,"COLm_o_crDoG")     
    if BN_fin:
        COLm_o_cbDoGBn=extract_layer(model,"COLm_o_cbDoGBn")  
        COLm_o_crDoGBn=extract_layer(model,"COLm_o_crDoGBn")  
        
    COLm_o_cbFinal = extract_layer(model,"COLm_o_cbFinal")
    COLm_o_crFinal = extract_layer(model,"COLm_o_crFinal")
    
    COLm_o_cbFinalresidu= extract_layer(model,"COLm_o_cbFinal_residu")
    COLm_o_crFinalresidu= extract_layer(model,"COLm_o_crFinal_residu")
    return(COLm_i_y,COLm_i_cb,COLm_i_cr, COLm_i_yDoG,COLm_i_cbDoG,COLm_i_crDoG,   COLm_i_yDoGBn,COLm_i_cbDoGBn,COLm_i_crDoGBn ,COLm_o_cb,COLm_o_cbDoG,COLm_o_cbDoGBn,COLm_o_cbFinal,COLm_o_cbFinalresidu,COLm_o_cr,COLm_o_crDoG,COLm_o_crDoGBn,COLm_o_crFinal,COLm_o_crFinalresidu)   
     
def extract_layers__STr_y(model,BN_init,BN_fin,DOG_init,DOG_fin):
    
    '''
    Extract layers from ST branch specialized on cb,cr channels ("STy")
    '''
    
    STm_i_y,STm_i_cb,STm_i_cr, STm_i_yDoG,STm_i_cbDoG,STm_i_crDoG,   STm_i_yDoGBn,STm_i_cbDoGBn,STm_i_crDoGBn ,STm_o_y,STm_o_yDoG,STm_o_yDoGBn,STm_o_yFinal,STm_o_yFinal_residu=None,None,None,None,None,None,None,None,None,None,None,None,None,None
    
    STm_i_y=extract_layer(model,"STm_i_y") 
    STm_i_cb=extract_layer(model,"STm_i_cb") 
    STm_i_cr=extract_layer(model,"STm_i_cr") 
    if BN_init:
        STm_i_yDoGBn=extract_layer(model,"STm_i_yDoGBn") 
        STm_i_cbDoGBn=extract_layer(model,"STm_i_cbDoGBn") 
        STm_i_crDoGBn=extract_layer(model,"STm_i_crDoGBn")  
    if DOG_init:
        STm_i_yDoG=extract_layer(model,"STm_i_yDoG") 
        STm_i_cbDoG=extract_layer(model,"STm_i_cbDoG") 
        STm_i_crDoG=extract_layer(model,"STm_i_crDoG")          
    STm_o_y=extract_layer(model,"STm_o_y")    
    if DOG_fin:
        STm_o_yDoG=extract_layer(model,"STm_o_yDoG")     
    if BN_fin:
        STm_o_yDoGBn=extract_layer(model,"STm_o_yDoGBn")         
    STm_o_yFinal = extract_layer(model,"STm_o_yFinal")
    STm_o_yFinal_residu = extract_layer(model,"STm_o_yFinal_residu")
    return(STm_i_y,STm_i_cb,STm_i_cr, STm_i_yDoG,STm_i_cbDoG,STm_i_crDoG,   STm_i_yDoGBn,STm_i_cbDoGBn,STm_i_crDoGBn ,STm_o_y,STm_o_yDoG,STm_o_yDoGBn,STm_o_yFinal,STm_o_yFinal_residu)   


def extract_layers_STr3(model,BN_init,BN_fin,DOG_init,DOG_fin):
    '''
    Extract layers from ST branch specialized on y channel ("ST3")
    '''
    ST3m_i_y,ST3m_i_cb,ST3m_i_cr, ST3m_i_yDoG,ST3m_i_cbDoG,ST3m_i_crDoG,   ST3m_i_yDoGBn,ST3m_i_cbDoGBn,ST3m_i_crDoGBn ,ST3m_o_cb,ST3m_o_cbDoG,ST3m_o_cbDoGBn,ST3m_o_cbFinal,ST3m_o_cbFinalresidu=None,None,None,None,None,None,None,None,None,None,None,None,None,None
    ST3m_o_cr,ST3m_o_crDoG,ST3m_o_crDoGBn,ST3m_o_crFinal,ST3m_o_crFinalresidu=None,None,None,None,None
    ST3m_o_y,ST3m_o_yDoG,ST3m_o_yDoGBn,ST3m_o_yFinal,ST3m_o_yFinalresidu=None,None,None,None,None

    ST3m_i_y=extract_layer(model,"ST3m_i_y")
    ST3m_i_cb=extract_layer(model,"ST3m_i_cb") 
    ST3m_i_cr=extract_layer(model,"ST3m_i_cr") 
    if BN_init:
        ST3m_i_yDoGBn=extract_layer(model,"ST3m_i_yDoGBn") 
        ST3m_i_cbDoGBn=extract_layer(model,"ST3m_i_cbDoGBn") 
        ST3m_i_crDoGBn=extract_layer(model,"ST3m_i_crDoGBn")  
    if DOG_init:
        ST3m_i_yDoG=extract_layer(model,"ST3m_i_yDoG") 
        ST3m_i_cbDoG=extract_layer(model,"ST3m_i_cbDoG") 
        ST3m_i_crDoG=extract_layer(model,"ST3m_i_crDoG")    
      
    ST3m_o_y=extract_layer(model,"ST3m_o_y") 
    ST3m_o_cb=extract_layer(model,"ST3m_o_cb")    
    ST3m_o_cr=extract_layer(model,"ST3m_o_cr")    
    if DOG_fin:
        ST3m_o_yDoG=extract_layer(model,"ST3m_o_yDoG") 
        ST3m_o_cbDoG=extract_layer(model,"ST3m_o_cbDoG") 
        ST3m_o_crDoG=extract_layer(model,"ST3m_o_crDoG")     
    if BN_fin:
        ST3m_o_yDoGBn=extract_layer(model,"ST3m_o_yDoGBn")  
        ST3m_o_cbDoGBn=extract_layer(model,"ST3m_o_cbDoGBn")  
        ST3m_o_crDoGBn=extract_layer(model,"ST3m_o_crDoGBn")  
    
    ST3m_o_yFinal = extract_layer(model,"ST3m_o_yFinal")
    ST3m_o_cbFinal = extract_layer(model,"ST3m_o_cbFinal")
    ST3m_o_crFinal = extract_layer(model,"ST3m_o_crFinal")
    
    ST3m_o_yFinalresidu= extract_layer(model,"ST3m_o_yFinal_residu")
    ST3m_o_cbFinalresidu= extract_layer(model,"ST3m_o_cbFinal_residu")
    ST3m_o_crFinalresidu= extract_layer(model,"ST3m_o_crFinal_residu")
    return(ST3m_i_y,ST3m_i_cb,ST3m_i_cr, ST3m_i_yDoG,ST3m_i_cbDoG,ST3m_i_crDoG,   ST3m_i_yDoGBn,ST3m_i_cbDoGBn,ST3m_i_crDoGBn ,ST3m_o_y,ST3m_o_cb,ST3m_o_yDoG,ST3m_o_cbDoG,ST3m_o_cbDoGBn,ST3m_o_yDoGBn,ST3m_o_cbFinal,
           ST3m_o_yFinal,ST3m_o_cbFinalresidu,ST3m_o_yFinalresidu,ST3m_o_cr,ST3m_o_crDoG,ST3m_o_crDoGBn,ST3m_o_crFinal,ST3m_o_crFinalresidu)   
    

def extract_layers_MAIN(model,nombre_class,DOG_init,DOG_fin,BN_init,BN_fin):
        """
        Extract layers from MAIN model ( SR / DENOISING / BLURRING) 
        """    
        SRm_i_y,SRm_i_cb,SRm_i_cr,SRm_i_yDoG,SRm_i_cbDoG,SRm_i_crDoG,SRm_i_yDoGBn,SRm_i_cbDoGBn,SRm_i_crDoGBn,SRm_o_y,SRm_o_yDoG,SRm_o_yDoGBn,SRm_o_yFinal = None,None,None,None,None,None,None,None,None,None,None,None,None

        SRm_i_y=extract_layer(model,"SRm_i_y") 
        SRm_i_cb=extract_layer(model,"SRm_i_cb") 
        SRm_i_cr=extract_layer(model,"SRm_i_cr") 
        if DOG_init:
            SRm_i_yDoG=extract_layer_output(model,nombre_class,"SRm_i_yDoG") 
            SRm_i_cbDoG=extract_layer_output(model,nombre_class,"SRm_i_cbDoG") 
            SRm_i_crDoG=extract_layer_output(model,nombre_class,"SRm_i_crDoG") 
        if BN_init:
            SRm_i_yDoGBn=extract_layer_output(model,nombre_class,"SRm_i_yDoGBn") 
            SRm_i_cbDoGBn=extract_layer_output(model,nombre_class,"SRm_i_cbDoGBn") 
            SRm_i_crDoGBn=extract_layer_output(model,nombre_class,"SRm_i_crDoGBn") 
        SRm_o_y=extract_layer_output(model,nombre_class,"SRm_o_y") 
        if DOG_fin:
            SRm_o_yDoG=extract_layer_output(model,nombre_class,"SRm_o_yDoG") 
        if BN_fin:
            SRm_o_yDoGBn=extract_layer_output(model,nombre_class,"SRm_o_yDoGBn") 
        SRm_o_yFinal=extract_layer_output(model,nombre_class,"SRm_o_yFinal") 
        return(SRm_i_y,SRm_i_cb,SRm_i_cr,SRm_i_yDoG,SRm_i_cbDoG,SRm_i_crDoG,SRm_i_yDoGBn,SRm_i_cbDoGBn,SRm_i_crDoGBn,SRm_o_y,SRm_o_yDoG,SRm_o_yDoGBn,SRm_o_yFinal)

        
# 2.b Copy parameters from model_patch trained with (K,sizex,sizey,n) patches into the new model with the same architecture
def copy_parameters(model,model_patch) : 
    '''
    Copy parameters from model_patch onto the model graph (assuming layers are the same)
    '''
    for layer in model.layers:
        try:
            layer.set_weights(model_patch.get_layer(name=layer.name).get_weights())
        except ValueError:
            print("{} is a layer not found or without any parameters".format(layer.name))
            pass                
    return (model) 

# 3. Output features computation out of model for a given tensor
def compute_features_MAIN(SRm_i_y,SRm_i_cb,SRm_i_cr,SRm_i_yDoG,SRm_i_cbDoG,SRm_i_crDoG,SRm_i_yDoGBn,SRm_i_cbDoGBn,SRm_i_crDoGBn,SRm_o_y,SRm_o_yDoG,SRm_o_yDoGBn,SRm_o_yFinal,DOG_init,DOG_fin,BN_init,BN_fin,model, INPUT ) :
    ''' 
    Compute Intermediate features out of cut models and INPUT patch tensor (MAIN NN)
    '''
    #a. Output
    pred=model.predict([INPUT])
    try:
        e=(pred.shape)  # If pixel-wise loss
        prediction = np.reshape(pred[:,:,:,:],(pred.shape[0],pred.shape[1],pred.shape[2],3) ) 
    except AttributeError: # If perceptual loss
        e=(pred[0].shape)
        prediction = np.reshape(pred[0][:,:,:,:],(pred[0].shape[0],pred[0].shape[1],pred[0].shape[2],3) ) 
    
    SRf_o_yFinal=SRm_o_yFinal.predict([INPUT])
    SRf_o_y=SRm_o_y.predict([INPUT]) 

    if DOG_fin:
        SRf_o_yDoG=SRm_o_yDoG.predict([INPUT]) 
    else:
        SRf_o_yDoG=0
    if BN_fin:
        SRf_o_yDoGBn=SRm_o_yDoGBn.predict([INPUT])   
    else:
        SRf_o_yDoGBn=0
    if DOG_init:
        SRf_i_yDoG=SRm_i_yDoG.predict([INPUT])
        SRf_i_cbDoG=SRm_i_cbDoG.predict([INPUT])
        SRf_i_crDoG=SRm_i_crDoG.predict([INPUT])
    else:
        SRf_i_yDoG=SRf_i_cbDoG=SRf_i_crDoG=0
    if BN_init:
        SRf_i_yDoGBn=SRm_i_yDoGBn.predict([INPUT])
        SRf_i_cbDoGBn=SRm_i_cbDoGBn.predict([INPUT])
        SRf_i_crDoGBn=SRm_i_crDoGBn.predict([INPUT])
    else:
        SRf_i_yDoGBn,SRf_i_cbDoGBn,SRf_i_crDoGBn=0
    
    
    return (prediction, SRf_i_yDoG,SRf_i_cbDoG,SRf_i_crDoG,SRf_i_yDoGBn,SRf_i_cbDoGBn,SRf_i_crDoGBn,SRf_o_y,SRf_o_yDoG,SRf_o_yDoGBn,SRf_o_yFinal)  #SRf_i_y,SRf_i_cb,SRf_i_cr,


def compute_features_Sty(STm_i_y,STm_i_cb,STm_i_cr,STm_i_yDoG,STm_i_cbDoG,STm_i_crDoG,STm_i_yDoGBn,STm_i_cbDoGBn,STm_i_crDoGBn,STm_o_y,STm_o_yDoG,STm_o_yDoGBn,STm_o_yFinal,DOG_init,DOG_fin,BN_init,BN_fin,model,INPUT ) :
    '''
    Compute Intermediate features out of cut models and INPUT patch tensor (STy)
    '''
    #a. Output
    pred=model.predict([INPUT])
    try:
        e=(pred.shape)  # If pixel-wise loss
        prediction = np.reshape(pred,(pred.shape[0],pred.shape[1],pred.shape[2],3) ) 
    except AttributeError: # If perceptual loss
        e=(pred[0].shape)
        prediction = np.reshape(pred[0],(pred[0].shape[0],pred[0].shape[1],pred[0].shape[2],3) )
        
    STf_o_yFinal=STm_o_yFinal.predict([INPUT])
    STf_o_y=STm_o_y.predict([INPUT]) 
    
    if DOG_fin:
        STf_o_yDoG=STm_o_yDoG.predict([INPUT]) 
    else:
        STf_o_yDoG=0
    if BN_fin:
        STf_o_yDoGBn=STm_o_yDoGBn.predict([INPUT])      
    else:
        STf_o_yDoGBn=0
    if DOG_init:
        STf_i_yDoG=STm_i_yDoG.predict([INPUT])
        STf_i_cbDoG=STm_i_cbDoG.predict([INPUT])
        STf_i_crDoG=STm_i_crDoG.predict([INPUT])
    else:
        STf_i_yDoG=STf_i_cbDoG=STf_i_crDoG=0
    if BN_init:
        STf_i_yDoGBn=STm_i_yDoGBn.predict([INPUT])
        STf_i_cbDoGBn=STm_i_cbDoGBn.predict([INPUT])
        STf_i_crDoGBn=STm_i_crDoGBn.predict([INPUT])
    else:
        STf_i_yDoGBn=STf_i_cbDoGBn=STf_i_crDoGBn=0
        
    return (prediction, STf_i_yDoG,STf_i_cbDoG,STf_i_crDoG,STf_i_yDoGBn,STf_i_cbDoGBn,STf_i_crDoGBn,STf_o_y,STf_o_yDoG,STf_o_yDoGBn,STf_o_yFinal) #STf_i_y,STf_i_cb,STf_i_cr,



def compute_features_Stcol(COLm_i_y,COLm_i_cb,COLm_i_cr, COLm_i_yDoG,COLm_i_cbDoG,COLm_i_crDoG,   COLm_i_yDoGBn,COLm_i_cbDoGBn,COLm_i_crDoGBn ,COLm_o_cb,COLm_o_cbDoG,COLm_o_cbDoGBn,COLm_o_cbFinal,COLm_o_cbFinalresidu,COLm_o_cr,COLm_o_crDoG,COLm_o_crDoGBn,COLm_o_crFinal,COLm_o_crFinalresidu,DOG_init,DOG_fin,BN_init,BN_fin,model,INPUT ) :
    '''
    Compute Intermediate features out of cut models and INPUT patch tensor (STcol)
    '''
    pred=model.predict([INPUT])
    try:
        e=(pred.shape)  # If pixel-wise loss
        prediction = np.reshape(pred,(pred.shape[0],pred.shape[1],pred.shape[2],3) ) 
    except AttributeError: # If perceptual loss
        e=(pred[0].shape)
        prediction = np.reshape(pred[0],(pred[0].shape[0],pred[0].shape[1],pred[0].shape[2],3) )
        
    COLf_o_cbFinalresidu=COLm_o_cbFinalresidu.predict(INPUT)     
    COLf_o_crFinalresidu=COLm_o_crFinalresidu.predict(INPUT)     

    return (prediction ,COLf_o_cbFinalresidu,COLf_o_crFinalresidu)


def compute_features_St3(ST3m_i_y,ST3m_i_cb,ST3m_i_cr, ST3m_i_yDoG,ST3m_i_cbDoG,ST3m_i_crDoG,   ST3m_i_yDoGBn,ST3m_i_cbDoGBn,ST3m_i_crDoGBn ,ST3m_o_y,ST3m_o_cb,ST3m_o_yDoG,ST3m_o_cbDoG,ST3m_o_cbDoGBn,ST3m_o_yDoGBn,ST3m_o_cbFinal,ST3m_o_yFinal,ST3m_o_cbFinalresidu,ST3m_o_yFinalresidu,ST3m_o_cr,ST3m_o_crDoG,ST3m_o_crDoGBn,ST3m_o_crFinal,ST3m_o_crFinalresidu,DOG_init,DOG_fin,BN_init,BN_fin,model,INPUT ) :
    '''
    Compute Intermediate features out of cut models and INPUT patch tensor (ST3)
    '''
    pred=model.predict([INPUT])
    try:
        e=(pred.shape)  # If pixel-wise loss
        prediction = np.reshape(pred,(pred.shape[0],pred.shape[1],pred.shape[2],3) ) 
    except AttributeError: # If perceptual loss
        e=(pred[0].shape)
        prediction = np.reshape(pred[0],(pred[0].shape[0],pred[0].shape[1],pred[0].shape[2],3) )
        
    ST3f_o_yFinal_residu=ST3m_o_yFinalresidu.predict([INPUT]) 
    ST3f_o_cbFinal_residu=ST3m_o_cbFinalresidu.predict([INPUT])     
    ST3f_o_crFinal_residu=ST3m_o_crFinalresidu.predict([INPUT])

    return (prediction ,ST3f_o_yFinal_residu,ST3f_o_cbFinal_residu,ST3f_o_crFinal_residu)  