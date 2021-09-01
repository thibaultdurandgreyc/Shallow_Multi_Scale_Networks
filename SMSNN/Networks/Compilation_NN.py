import tensorflow as tf
#from tensorflow.keras.losses import *
from tensorflow.keras import initializers
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.utils import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback

from various_functions.tensor_tf_data_fonctions import *
from various_functions.numpy_data_fonctions import *
from various_functions.custom_filters import *
from various_functions.tensor_tf_data_fonctions import *

from Networks.Loss_Constructor import * 
    
def compilation_MAIN_network(loss_perceptuelle:list,loss_style:list, loss_pixel:bool, ponderation_pixel:float, ponderation_perc:float, ponderation_style:float, end):
    '''
    MAIN Network graph compilation with chosen loss and weights.
    
    (During TRaining : Input : Ycbcr [0,1]
    Output : Ycbcr [0,1]
    TRUE Data Database where generators extracts ground truth : Ycbcr [0,1])
    
    * loss_perceptuelle, loss_style, loss_pixel (list) : list of layers to extract from the VGG to compute perceptual, style and pixel-wise losses
    * ponderation_pixel, ponderation_style,  ponderation_pixel (int) : weights for the global loss to be computed out of perceptual, style and pixel-wise losses
    * end (tensor) : shape [batch_size,width,height,3], output tensor to compute loss
    '''

    outputs,loss_list,weight_list=[],[],[]

    for i in loss_perceptuelle:
        vgg16 = VGG16(include_top=False, weights='imagenet', pooling="avg")
        vgg16.trainable = False
        loss_model = tf.keras.models.Model(inputs=vgg16.inputs, outputs=vgg16.layers[i].output)    
        loss_list.append(vgg_loss_ycbcr(loss_model) ) 
        weight_list.append(ponderation_perc)
        outputs.append(end)
                
    for j in loss_style:
        vgg16 = VGG16(include_top=False, weights='imagenet', pooling="avg")
        vgg16.trainable = False
        loss_model = tf.keras.models.Model(inputs=vgg16.inputs, outputs=vgg16.layers[j].output)               
        loss_list.append(style_loss_vgg_ycbcr(loss_model,sizex)) 
        weight_list.append(ponderation_style)
        outputs.append(end)
           
    if loss_pixel:
        loss_list.append(mse_loss1  )#(L2 ycbcr)  / l1_loss1 (L1 ycbcr)
        weight_list.append(ponderation_pixel)
        outputs.append(end)
    
    return(loss_list,outputs,weight_list)


def compilation_StyleTransfer(normalisation:bool,loss_perceptuelle:list, loss_style:list, loss_pixel:bool, ponderation_pixel:float, ponderation_perc:float, ponderation_style:float,end, style_gram=0):  #compilation_ST_network
    '''
    ST(y) / ST(ycbcr) / ST3 :  Network graph compilation with chosen loss and weights. using VGG16
    
    (During Training : Input : Ycbcr [0,1]
    Output : rgb [0,1]
    TRUE Data Database where generators extracts ground truth : Ycbcr [0,1])
    
    * normalisation (bool) : if True, normalize Output Y channel 
    * loss_perceptuelle, loss_style, loss_pixel (list) : list of layers to extract from the VGG to compute perceptual, style and pixel-wise losses
    * ponderation_pixel, ponderation_style,  ponderation_pixel (int) : weights for the global loss to be computed out of perceptual, style and pixel-wise losses
    * end (tensor) : shape [batch_size,width,height,3], output tensor to compute loss
    * style_gram (tensor) : gram matrix stacked along the batch size for computing style losses computed with gram_matrix()
    
    '''
    outputs,loss_list,weight_list=[],[],[]

    
    # weights between vgg output features depending on layer number
    weight_style_layers=[0 for i in range(19)]
    weight_style_layers[0:2]=[1 for i in range(3)]
    weight_style_layers[3:5]=[1 for i in range(3)]
    weight_style_layers[6:9]=[1 for i in range(4)]
    weight_style_layers[10:13]=[1 for  i in range(4)]
    weight_style_layers[14:]=[1 for i in range(5)]
    
    for j in loss_perceptuelle:
        vgg16 = VGG16(include_top=False, weights='imagenet', pooling="avg")
        vgg16.trainable = False
        loss_model = tf.keras.models.Model(inputs=vgg16.inputs, outputs=vgg16.layers[j].output)    
        loss_list.append(vgg_loss_rgb(loss_model,normalisation))
        weight_list.append((ponderation_perc)/len(loss_perceptuelle))  
        outputs.append(end)

    for j in loss_style:
        vgg16 = VGG16(include_top=False, weights='imagenet', pooling="avg")
        vgg16.trainable = False
        loss_model_style = tf.keras.models.Model(inputs=vgg16.inputs, outputs=vgg16.layers[j].output)
        loss_list.append(style_transfert_loss_vgg_rgb(loss_model_style,j,style_gram,normalisation)) 
        weight_list.append((ponderation_style*weight_style_layers[j])/(len(loss_style)))
        outputs.append(end)
    

    if loss_pixel:
        loss_list.append(mse_loss1_rgb_col)  #(L2)             # mse_loss1_rgb_col L2 on colors   # mse_loss1_rgb L2 on whole tensor
        weight_list.append(ponderation_pixel)
        #loss_list.insert(0,l1_loss1)  #(L1)  
        outputs.append(end)

    return(loss_list,outputs,weight_list)
    


def compilation_ColTransfer(normalisation:bool,loss_perceptuelle:list, loss_style:list, loss_pixel:bool, ponderation_pixel:float, ponderation_perc:float, ponderation_style:float,end, hist_style,clusters): 
    '''
    ST(ycol) : Network graph compilation with chosen loss and weights. 
    [Perceptual losses from concatenation of grayscale images
    Histogram losses]
    
    * normalisation (bool) : if True, normalize Output Y channel 
    * loss_perceptuelle, loss_style, loss_pixel (list) : list of layers to extract from the VGG to compute perceptual, style and pixel-wise losses
    * ponderation_pixel, ponderation_style,  ponderation_pixel (int) : weights for the global loss to be computed out of perceptual, style and pixel-wise losses
    * end (tensor) : shape [batch_size,width,height,3], output tensor to compute loss
    * hist_style (tensor) : histogram of the style used for training computed with histogram_2d()
    * clusters (tensor) : shape [2,n_bins]
    '''
    outputs,loss_list,weight_list=[],[],[]

    # weights between vgg output features depending on layer number
    weight_style_layers=[0 for i in range(19)]
    weight_style_layers[0:2]=[1 for i in range(3)]
    weight_style_layers[3:5]=[1 for i in range(3)]
    weight_style_layers[6:9]=[1 for i in range(4)]
    weight_style_layers[10:13]=[1 for  i in range(4)]
    weight_style_layers[14:]=[1 for i in range(5)]
        
    for j in loss_perceptuelle:
        vgg16 = VGG16(include_top=False, weights='imagenet', pooling="avg")
        vgg16.trainable = False
        loss_model = tf.keras.models.Model(inputs=vgg16.inputs, outputs=vgg16.layers[j].output)    
        loss_list.append(vgg_loss_rgb_mean(loss_model))
        weight_list.append((ponderation_perc)/len(loss_perceptuelle))  
        outputs.append(end)
    

    for j in loss_style:
        vgg16 = VGG16(include_top=False, weights='imagenet', pooling="avg")
        vgg16.trainable = False
        loss_model_style = tf.keras.models.Model(inputs=vgg16.inputs, outputs=vgg16.layers[j].output)
        loss_list.append(col_loss_histogram(hist_style)) 
        weight_list.append((ponderation_style*weight_style_layers[j])/(len(loss_style)))
        outputs.append(end)

    #loss_pixel:
    '''
    loss_list.append(mse_loss1_rgb_soloY)  #(L2)           
    weight_list.append(ponderation_pixel)
    outputs.append(end)
    '''
    # Histogram
    loss_list.append(col_loss_histogram(hist_style,clusters)) 
    weight_list.append(1.)
    outputs.append(end)
    return(loss_list,outputs,weight_list)





