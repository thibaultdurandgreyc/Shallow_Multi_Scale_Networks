import tensorflow as tf
from tensorflow.keras.losses import *
from tensorflow.keras.models import *
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import *
from various_functions.tensor_tf_data_fonctions import *

from tensorflow.keras import backend as K

# -----------------------------------------------------------------------------
# ---------------------------- PIXEL LOSS  ------------------------------------
# -----------------------------------------------------------------------------

def PSNR(y_true, y_pred):
    """
    Compute PSNR between Y_true (Ycbcr) and Y_pred (Ycbcr) 
    """
    return tf.image.psnr(y_true,y_pred,1)

def l1_loss1(y_true,y_pred,ponderation_pixel):
    '''
    Mean Absolute Error between Y_true (Ycbcr) & Y_pred (Ycbcr)
    ''' 
    return (ponderation_pixel*(tf.keras.losses.MAE(tf.expand_dims(y_true, axis=0),tf.expand_dims(y_pred, axis=0))))
        
def mse_loss1(y_true,y_pred):
    """
    MSE between Y_true (Ycbcr) & Y_pred (Ycbcr)
    """
    return ((tf.keras.losses.MSE(tf.expand_dims(y_true, axis=0),tf.expand_dims(y_pred, axis=0))))

def mse_loss1_rgb(y_true,y_pred):
    """
    MSE with Y_true ( Ycbcr -> rgb) & Y_pred (rgb)
    y_true [0,1] tensor
    y_pred [0,1] tensor
    """
    y_true = tensor_ycbcr2rgb(y_true)/255.
    return ((tf.keras.losses.MSE(tf.expand_dims(y_true, axis=0),tf.expand_dims(y_pred, axis=0))))

def mse_loss1_rgb_col(y_true,y_pred):
    """
    MSE with Y_true ( Ycbcr )  & Y_pred  ( rgb-> ycbcr) / Only cb,cr
    """
    y_pred = tf_rgb2ycbcr(y_pred)/255.
    
    y_c_pred,cb_c_pred,cr_c_pred=tf.split(y_pred, 3 , axis=-1)
    y_c_true,cb_c_true,cr_c_true=tf.split(y_true, 3 , axis=-1)    
    
    return ((tf.keras.losses.MSE(tf.expand_dims(cb_c_pred, axis=0),tf.expand_dims(cb_c_true, axis=0)))  +   (tf.keras.losses.MSE(tf.expand_dims(cr_c_pred, axis=0),tf.expand_dims(cr_c_true, axis=0))))
   
def mse_loss1_rgb_soloY(y_true,y_pred):
    """
    MSE with Y_true (Ycbcr) & Y_pred  (rgb-> ycbcr) / Only Y
    """
    y_pred = tf_rgb2ycbcr(y_pred)/255.
    y_c_pred,cb_c_pred,cr_c_pred=tf.split(y_pred, 3 , axis=-1)
    y_c_true,cb_c_true,cr_c_true=tf.split(y_true, 3 , axis=-1)    
    
    return (tf.keras.losses.MSE(tf.expand_dims(y_c_pred, axis=0),tf.expand_dims(y_c_true, axis=0))  )



# -----------------------------------------------------------------------------
# ---------------------------- CONTENT/PERCEPTUAL LOSS  ----------------------------------
# -----------------------------------------------------------------------------

def normalize_with_moments(x, epsilon=1e-5, sigma=0.2): 
    '''
    Function for Y normalization applied to RGB [0,1] tensor (mean 0, std deviation 0.2) . Output is converted in RGb.
    
        Used
        - for branch St(y) : style normalized ( then Gram matrices are changed and applied to each tensor before computing loss)
    
    (Used for having the same kinf of magnitude of high frequency details in terms of energy, no matter the style used)
    
    *  x (input tensor) : [0,1] rgb
    '''
    ycbcr = tf_rgb2ycbcr(x)/255.

    mean_y, variance_y = tf.nn.moments(x[:,:,:,0],axes=[0 ,1 ,2])
    mean_cb, variance_cb = tf.nn.moments(x[:,:,:,1],axes=[0 ,1 ,2])
    mean_cr, variance_cr = tf.nn.moments(x[:,:,:,2],axes=[0 ,1 ,2])

    #new_mean=0.5
    y_normed = sigma*( (ycbcr[:,:,:,0] - mean_y) / (tf.sqrt(variance_y + epsilon)) ) #+ new_mean
    cb_normed = ycbcr[:,:,:,1]
    cr_normed = ycbcr[:,:,:,2] 
    
    out = tf.stack([y_normed,cb_normed,cr_normed],axis=-1)
    
    out=tensor_ycbcr2rgb(out)/255.
    return out


def vgg_loss_ycbcr(loss_model):
    """
    Content Loss for Y_true & Y_pred  (Ycbcr) / Used in :
        - SR
        - (COL)
        
    * loss_model_style (keras API classifior model) : typically VGG 16
    """
    def loss(y_true,y_pred):
        return tf.keras.backend.mean(tf.keras.backend.square(loss_model(preprocess_input(tensor_ycbcr2rgb(y_true))) - loss_model(preprocess_input(tensor_ycbcr2rgb(y_pred)))))
    return(loss)
 
    
def vgg_loss_rgb_mean(loss_model): 
    """
    Content Loss for Y_true & Y_pred (Ycbcr) build with the concatention of mean(rgb) 3 times  / Used in :
        - ST(col) mean(rgb) 
        
    * loss_model_style (keras API classifior model) : typically VGG 16
    """
    def loss(y_true,y_pred):

        y_true_rgb = tensor_ycbcr2rgb(y_true)
        
        y_true_rgb = tf.expand_dims(tf.reduce_mean(y_true_rgb,axis=-1),axis=-1)
        y_pred = tf.expand_dims(tf.reduce_mean(y_pred,axis=-1),axis=-1)
        
        y_true_rgb=tf.concat([y_true_rgb,y_true_rgb,y_true_rgb],axis=-1)
        y_pred=tf.concat([y_pred,y_pred,y_pred],axis=-1)
        
        vgg_pred = loss_model(preprocess_input((y_pred*256.)))
        vgg_true = loss_model(preprocess_input(y_true_rgb) )
        
        return (tf.keras.backend.mean(tf.keras.backend.square( vgg_true - vgg_pred  )))/(vgg_pred.shape[1]*vgg_pred.shape[2]*vgg_pred.shape[3]) # /(vgg_pred.shape[1]*vgg_pred.shape[2]*vgg_pred.shape[3]) same normalization jonshon
        
    return(loss)

def vgg_loss_rgb(loss_model,normalize): 
    """
    Content Loss for Y_true & Y_pred (Ycbcr)  / Used in :
        - ST(y) (WITH normalization ; for controlling the magnitude of learned high frequency details no matter the style used)
        - ST3 (NO normalization ; like in classic style transfert)
    
    [ Not used in ST(cb,cr) because hist matching seems better than gram matrix for color transfer ]
    
    * loss_model_style (keras API classifior model) : typically VGG 16
    * normalize (bool) : if True : normalize y channel with 'normalize_with_moments()'
    """
    def loss(y_true,y_pred):

        y_true_rgb = tensor_ycbcr2rgb(y_true)
        if normalize: # normalize Y channel -- For St (Y) branch
            y_true=normalize_with_moments(y_true_rgb/255.)*255. 
            y_pred=normalize_with_moments(y_pred)
        else: # No normalization 
            y_true=y_true_rgb
            y_pred=y_pred
        vgg_pred = loss_model(preprocess_input((y_pred*255.)))
        vgg_true = loss_model(preprocess_input(y_true) )
        return (tf.keras.backend.mean(tf.keras.backend.square( vgg_true - vgg_pred  )))/(vgg_pred.shape[1]*vgg_pred.shape[2]*vgg_pred.shape[3]) 
        
    return(loss)


# -----------------------------------------------------------------------------
# ---------------------------- STYLE TRANSFERT LOSS  --------------------------
# -----------------------------------------------------------------------------

def gram_matrix_nobatch(x):
    permuted=tf.keras.backend.permute_dimensions(x[:,:,:], (2, 0, 1))
    features = tf.keras.backend.batch_flatten(permuted)
    nbre_pixel=features.shape[1]
    nre_channel= features.shape[0]
    gram = tf.keras.backend.dot(features, tf.keras.backend.transpose(features))
    gram = gram  /  ( (nbre_pixel) * (nre_channel))   
    return gram

def gram_matrix(x,bs):
    l=[]
    for b in range(bs):
        l.append(gram_matrix_nobatch(x[b,:,:,:]))
    gram=tf.stack(l,axis=0)
    return gram


def style_transfert_loss_vgg_rgb(loss_model_style,style,style_gram,normalize):
    ''' 
    Style Loss between Style gram matrix (already computed and stored as 'style_gram') & Y_pred gram matrix  / Used in :
        - St(y) (with style normalization)
        - St3 (without normalization)
    
    * loss_model_style (keras API classifior model) : typically VGG 16
    * style (int): VGG layer number
    * normalize (bool) : if True : normalize y channel with 'normalize_with_moments()'
    * y_true : [0,1] tensor ycbcr
    * y_pred : [0,1] tensor rgb
    '''
    def loss(y_true,y_pred):
        if normalize:
            y_pred=normalize_with_moments(y_pred)
        
        bs = y_true.numpy().shape[0]
        y_pred_style = loss_model_style(preprocess_input((y_pred*255.)))  
        C = gram_matrix(y_pred_style,bs) 
        S = style_gram[style] 
        S_bs =[]
        for b in range(bs):
            S_bs.append(S)    # MONO STYLE
            #S_bs.append(S[b])  # MULTI STYLE
        S=tf.stack(S_bs,axis=0)
        l_style_loss = style_transfert_loss(S, C)

        return l_style_loss
    return(loss)

def style_transfert_loss(style, combination):
    '''
    MSE between 2 tensors
    '''
    return (tf.keras.backend.mean(tf.keras.backend.square(style - combination))) 



def style_loss_vgg_ycbcr(loss_model_style):
    '''
    Style Loss between Y_true gram matrix (already computed and stored as 'style_gram') & Y_pred gram matrix / May be used in : (but is not)
        - SR 
        - COL 
    * loss_model_style (keras API classifior model) : typically VGG 16
    * style (int): VGG layer number
    * y_true : [0,1] tensor ycbcr
    * y_pred : [0,1] tensor rgb
    '''
    def loss(y_true,y_pred):
        y_true_style = loss_model_style(preprocess_input(tensor_ycbcr2rgb(y_true)))
        y_pred_style = loss_model_style(preprocess_input(y_pred*255.))
        
        bs = y_true.numpy().shape[0]
        gram_y_pred = gram_matrix(y_pred_style,bs) 
        gram_y_true = gram_matrix(y_true_style,bs) 
        
        l_style_loss = style_transfert_loss(gram_y_true, gram_y_pred)
        return l_style_loss
    return(loss)


# -----------------------------------------------------------------------------
# ---------------------------- HISTOGRAM LOSS      --------------------------
# -----------------------------------------------------------------------------


def histogram_2d(tensor, clusters, row_num, col_num): 
        '''
        Compute 2d tensor histogram 
        
        * clusters (tensor) : shape (2,n) with n bins (int)
        * row_num (int), col_num (int) : size of the input tensor
        * tensor (tensor) : shape (n_row,n_col,2)   ( [0,1] cb,cr  )
        
        returns 
        ** normalized histogram
        
        '''
        sigm = 6 # sigma should bo smaller for more bins (6 for 120 bins)
        int_input = tensor *256.
        int_input = tf.transpose(int_input)
        
        
        factor=6
        int_input=int_input[:,::factor] # take only 1 pixel out of 4
        tensor=tensor[::factor,:]
        row_num=tensor.shape[0]
        
        # --- Compute Tf distance --- Dij=||Ci-Xj||**2=||Ki||**2 - ||Xj||**2 + 2<KI;Xj>
        bins=clusters.shape[1]
        l2_centers=clusters*clusters # (2,bins)

        l2_centers=tf.reduce_sum(l2_centers,axis=0) # (bins)
        l2_centers = tf.tile([l2_centers],[row_num,1])
        l2_centers=tf.transpose(l2_centers)
        
        l2_pixels = tensor*tensor * 256**2
        l2_pixels=tf.reduce_sum(l2_pixels,axis=1)
        l2_pixels = tf.tile([l2_pixels],[bins,1])
        l2_pixels = tf.cast(l2_pixels,tf.float32)
        
        # produit scalaire Pixels Centres
        produit_scalaire = tf.linalg.matmul(tf.transpose(clusters),(int_input))
        # Dij (i,centres, j pixels) = ||Ki-Xj||**2         
        D= l2_centers +  l2_pixels  -2*produit_scalaire

        x = tf.math.exp((-tf.abs(D))/(2*sigm**2)) 
        x = tf.math.reduce_sum(x,axis=1) # sum pixels for each cluster i
        x=x/tf.math.reduce_sum(x) #normalization

        return x
    
    
def col_loss_histogram(style_hist,clusters):
    '''
    Histogram Loss for Color Transfer matching Style Hist with Output Hist / Used in :
        - St(Cb,Cr) (over style tr loss)
    
    * style_hist (tensor histogram ) : histogram of style image built once with 'histogram_2d()'
    * clusters (tensor) : shape (2,n) with n bins (int)
    '''
    def loss(y_true,y_pred):
        pred_hist_ycbcr = tf_rgb2ycbcr(y_pred)

        tbstacked=[]
        bs=y_pred.shape[0]
        for b in range(bs):
            pred_hist = tf.reshape(pred_hist_ycbcr[b,:,:,1:3]/256.,(pred_hist_ycbcr.shape[1]*pred_hist_ycbcr.shape[2],2))
            pred_hist = histogram_2d(pred_hist,clusters , pred_hist.shape[0],pred_hist.shape[1])   
            tbstacked.append(pred_hist)
        pred_hist=tf.stack(tbstacked,axis=0)
        #return tf.keras.backend.mean(tf.keras.backend.square(style_hist  - pred_hist )) #+ y_pred*0   #https://stackoverflow.com/questions/54575515/keras-custom-loss-pool-values-into-histograms
        return tf.keras.losses.MAE(style_hist,pred_hist)
    return(loss)
    
