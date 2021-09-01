from __future__ import division,print_function
import numpy as np
import os
import scipy
import cv2
import tensorflow as tf
import math

# --- Energy() functions ---
def added_feature_PN(tensor_feature,nombre_patch_test,taille, enP, enN, fftP, fftN):
    '''
    In tensorflow, compute L2 norm ('energy') of the positive part and the negative part of a batch of feature. Adds these values to the specific statistics
    1 features output gives 4 statistics updates.
    
    * tensor_feature (tensor) : size [nombre_patch_test,width,height]
    [ nombre_patch_test (int) : size of the batch
      taille (int) : size of the patches (same size for the batch patches)]
    * enP, enN, fftP, fftN (int) : statistics to update
    
    Returns
    ** enP, enN, fftP, fftN (int) : updated statistics
    '''
    
    # 1. extraction of P,N features
    P=tf.reshape(tensor_feature,(nombre_patch_test,taille,taille,1) )
    N = tf.reshape( - tensor_feature,(nombre_patch_test,taille,taille,1) )
    N = tf.nn.relu(N)
    P = tf.nn.relu(P)
    
    # Add to statistics, energy & FFT (P for positive, N for Negative)
    enP+=tf.norm((P),ord=1).numpy() 
    enN+=tf.norm((N),ord=1).numpy() 
    fftP+=(tf.math.reduce_mean(tf.math.abs(tf.signal.fft2d(tf.cast(P,tf.complex64))))).numpy()
    fftN+=(tf.math.reduce_mean(tf.math.abs(tf.signal.fft2d(tf.cast(N,tf.complex64))))).numpy()
    return(enP,enN,fftP,fftN) 
    
# --- Data_preprocessing functions ---    
def decoupe_image(im:np.array,offsetX:int,offsetY:int,fen:int)->np.array:
    """
    Generate image section from a full size image ( grayscale ). 
    
    * im (numpy array): full numpy array
    * offsetX (int): offset from the upper left corner along X
    * offsetY (int): offset form the upper left corner along Y
    * fen (int): size of the section image
    Returns
    ** coupe (int): numpy image section
    """
    coupe = im[offsetX:offsetX+fen,offsetY:offsetY+fen] 
    return coupe   

def name_decoupe(numero:int,image_nom:str,offsetX:int,offsetY:int,fen:int)->str:
    """
    Generate the name of the associated image depending on the section
    
    * image_nom (str): name of the original image
    * offsetX (int): offset from the upper left corner along X
    * offsetY (int): offset form the upper left corner along Y
    * fen (int): size of the section image
    
    Return
    ** name (str): string name of the image section
    """
    #name = f'{numero}_{image_nom.replace(".png","")}_{fen}_{offsetX}_{offsetY}.png'
    numero=numero_into_thousands(numero,7)
    name = f'{numero}.png'
    return name

def numero_into_thousands(number,t):
    '''
    When creating patches, modify str names for patches to have unique names
    
    * number (str) : original number data name
    * t (int) : maximum size of patch name
    
    Returns
    ** number (str) : new str patch name 
    '''
    taille=len(str(number))
    taille_desired=t
    for zero in range(taille_desired - taille):
        number = '0'+str(number)
    return(number)

def test_variance(image:np,seuil:int): 
    """
    Test if image variance is high enough.
    * image (array) : numpy array to test
    * seuil (int) : minimum variance required
    Returns validation (bool), if True, image variance is above seuil
    """
    validation=False
    if (np.var(image)>seuil):
        validation=True
    return(validation)   
    
def is_greyimage(im):
    """
    Test if image variance is grey.
    * image (array) : numpy array to test
    Returns validation (bool), if True, image is grey
    """
    x = abs(im[:,:,0]-im[:,:,1])
    y = np.linalg.norm(x)
    if y==0:
        return True
    else:
        return False
    
def downsampling(im:np.array,R:int)->np.array:
    """
    Generate downgraded image section from HR npy array with bicubic downsampling
    * im (numpy array) : 1 channel numpy array
    * R (int) : downscale factor
    Returns : LR : 3 channels numpy array
    """
    #gaussian_filter(im, sigma=1)
    imL=cv2.resize(im,(int(im.shape[1]/R),int(im.shape[0]/R)), interpolation = cv2.INTER_CUBIC)  
    return(imL)

def downsampling_bilinear(im:np.array,R:int)->np.array:
    """
    Generate downgraded image section from HR npy array with bilinear downsampling
    * im (numpy array) : 1 channel numpy array
    * R (int) : downscale factor
    Returns : LR : 3 channels numpy array
    """
    imL=cv2.resize(im,(int(im.shape[1]/R),int(im.shape[0]/R)), interpolation = cv2.INTER_LINEAR)  
    return(imL)

def upsampling(im:np.array,R:int)->np.array:
    """
    Generate upgraded image section from LR npy array with bicubic upsampling
    * im (numpy array) : 1 channel numpy array
    * R (int) : downscale factor
    Returns : imL : 3 channels numpy array
    """
    imL = cv2.resize(im,(int(im.shape[1]*R),int(im.shape[0]*R)), interpolation = cv2.INTER_CUBIC)
    return(imL)

def upsampling3D(im:np.array,R:int)->np.array:
    """
    Generate upgraded image section from LR npy array with bicubic upsampling
    * im (numpy array) : 3 channels numpy array
    * R (int) : downscale factor
    Returns : imL : 3 channels numpy array
    """
    imL = cv2.resize(im,(int(im.shape[1]*R),int(im.shape[0]*R),3), interpolation = cv2.INTER_CUBIC)
    return(imL)
    
def upsampling_bilinear(im:np.array,R:int)->np.array:
    """
    Generate upgraded image section from LR npy array with bilinear upsampling
    * im (numpy array) : 3 channels numpy array
    * R (int) : downscale factor
    Returns : imL : 3 channels numpy array
    """
    imL = cv2.resize(im,(int(im.shape[1]*R),int(im.shape[0]*R)), interpolation = cv2.INTER_LINEAR)
    return(imL)

def grad_field(im:np.array):
    '''
    Computes gradient of a numpy array
    '''
    dx, dy = np.gradient(im.astype('float32'))
    return(dx,dy)

def rescale_255(patch:np):
    '''
    Rescales numpy array between 0;255
    '''
    out = np.uint8((patch-patch.min()) * (1/(patch.max()-patch.min())*255))
    return(out)

def rescale_255_maxrange(patch:np):
    '''
    Rescales numpy array between 0;patch.max()
    '''
    out = np.uint8((patch) * (1/(patch.max()-patch.min())*patch.max()-patch.min()))
    return(out)
    
    
def sigmoid_delta(x,delta=5):
    out= ( 1/(1 + np.exp((-x+ delta)*10)) ) 
    return(out)  
    
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def sigmoid_centree(x):
  return (sigmoid(x) -0.5)

def sigmoid_numpy(x):
    '''
    Sigmoid function for numpy array
    '''
    return np.vectorize(sigmoid)(x)

def tanh_numpy(x):
    '''
    tanh function for numpy array
    '''
    return np.vectorize(np.tanh)(x)

def linear_saving(x):
    """
    [-1,1] -> [0,255] no rescaling
    """
    return(122.5*(x+1))
     
def linear_reverse(x):
    """
    [0,255] -> [-1,1] no rescaling
    """
    return((x/122,5)-1)

def crop_center(img,cropx,cropy):
    """
    Crop img numpy 
    
    * img (numpy array) : [width,height,n]  with n = 1 or n = 3
    * cropx (int) : x size for the output
    * cropy (int): y size for the output
    
    Returns
    ** img (numpy array) : cropped array
    """
    x = img.shape[0]
    y = img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    if len(img.shape)==3: # 3 channels
        new_img=np.zeros((cropx,cropy,img.shape[2]))
        for channel in range(img.shape[2]):
            new_img[:,:,channel]=img[startx:startx+cropx , starty:starty+cropy,channel]
        return new_img
    else: # if 2 channels
        return img[startx:startx+cropx , starty:starty+cropy]

  
# --- Opening and Converts Image in numpy ---
def Ouverture_img(image,Upfactor):
    '''
    Open image as numpy array in ycbcr
    * Image  (str): path where to find the image
    * Upfactor (int): upsampling factor to apply
    Returns : im_full (npy) ycbcr
    im_Y (npy) y
    '''
    im_uint8 = cv2.imread(image)#.astype(np.float64)
    if type(im_uint8)!=type(None):
        if is_greyimage(im_uint8):
            im_uint8 = im_uint8[:,:,0]
        if len(im_uint8.shape)>2:
            #im_ycbcr = BGR2YCbCr(im_uint8)
            im_ycbcr = cv2.cvtColor(im_uint8, cv2.COLOR_BGR2YCrCb)
            im_full=im_ycbcr
            cr = im_full[:,:,1].copy()
            cb = im_full[:,:,2].copy()            
            im_full[:,:,1] = cb
            im_full[:,:,2] = cr
            im_Y = im_ycbcr[:,:,0]
        else:
            im_Y = im_uint8
            im_full=im_Y # TODO : problem with grayscale array
        return(im_full,im_Y)
    else:
        return(None,None)
 
def Ouverture_img_lab(image,Upfactor):
    '''
    Open image as numpy array in lab
    * Image  (str): path where to find the image
    * Upfactor (int): upsampling factor to apply
    Returns : im_full (npy) Lab
    im_Y (npy) L
    '''
    im_uint8 = cv2.imread(image)#.astype(np.float64)
    if type(im_uint8)!=type(None):
        if is_greyimage(im_uint8):
            im_uint8 = im_uint8[:,:,0]
        if len(im_uint8.shape)>2:
            im_lab = cv2.cvtColor(im_uint8, cv2.COLOR_BGR2LAB)
            im_full=im_lab
            im_Y = im_lab[:,:,0]
        else:
            im_Y = im_uint8
            im_full=im_Y #pb pour les images grayscale
        return(im_full,im_Y)
    else:
        return(None,None)
        
def Ouverture_img_rgb(image,Upfactor):
    '''
    Open image as numpy array in rgb
    * Image  (str): path where to find the image
    * Upfactor (int): upsampling factor to apply
    Returns : im_rgb (npy)
    '''
    im_uint8 = cv2.imread(image)#.astype(np.float64)
    if is_greyimage(im_uint8):
        im_rgb = im_uint8[:,:,0]
    im_rgb = cv2.cvtColor(im_uint8, cv2.COLOR_BGR2RGB)
    return(im_rgb)

def RGB2Ycbcr_numpy(r,g,b): 
    '''
    convert in numpy
    [0,1] rgb -> [0,1] ycbcr
    '''
    new_feat = np.zeros((r.shape[0],r.shape[1],3))
    new_feat[:,:,0]=r
    new_feat[:,:,1]=g
    new_feat[:,:,2]=b
    new_feat = cv2.cvtColor((new_feat*255.).astype(np.uint8), cv2.COLOR_RGB2BGR) # RGB -> BGR 
    new_feat = cv2.cvtColor(new_feat, cv2.COLOR_BGR2YCrCb) # BGR -> YcrCb
    new_feat_input = np.zeros((new_feat.shape[0],new_feat.shape[1],3))  #YCrCb -> YCbCr         
    new_feat_input[:,:,0]=new_feat[:,:,0]
    new_feat_input[:,:,1]=new_feat[:,:,2]
    new_feat_input[:,:,2]=new_feat[:,:,1]
    Y = new_feat_input[:,:,0].reshape(new_feat.shape[0],new_feat.shape[1])/255.
    cb = new_feat_input[:,:,1].reshape(new_feat.shape[0],new_feat.shape[1])/255.
    cr = new_feat_input[:,:,2].reshape(new_feat.shape[0],new_feat.shape[1])/255.
    return(Y,cb,cr)     


def RGB2YCbCr(im):
    '''
    convert in numpy
    [0,255] rgb -> [0,255]  ycbcr
    '''
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def YCBCbCr2RGB(im):  
    """
    convert in numpy
    [0,255] ycbcr -> [0,255]  rgb
    """
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return (rgb) 


# OTHER --
def lissage_mask(mask,moyenne,sigma):
    '''
    Used to smooth binary mask {0,1} through a convolution with a kernel. Adds a noise
    * mask (numpy array) : binary input mask {0,1}
    * moyenne : size of the kernel to smooth with
    * sigma : standart deviation of the noise to add
    Returns : mask (numpy array) smoothed and noisy (we may use sigma=0)
    '''
    oness = np.ones((moyenne,moyenne))
    moyenne_locale =(scipy.signal.convolve2d(mask,oness,mode="same") /moyenne**2  ) #*mask permet de ne faire la moyenne que sur le mask de base
    moyenne_locale =(scipy.signal.convolve2d(moyenne_locale,oness,mode="same") /moyenne**2  ) #*mask permet de ne faire la moyenne que sur le mask de base
    noise = np.random.normal(0,sigma,size=(mask.shape))*mask
    return(np.abs(moyenne_locale + noise))
    

def openImg_tf_format(folder,nom,taille_output,cropp):
    '''
    Open a png,jpg,tif or jpeg image as tensor and crop it to (taille_output,taille_output,3)
    '''
    try:
        out = tf.keras.preprocessing.image.load_img(path=os.path.join(folder,str(nom)+".png"))  
    except FileNotFoundError:
        try:
            out = tf.keras.preprocessing.image.load_img(path=os.path.join(folder,str(nom)+".jpg"))  
        except FileNotFoundError:
            try:
                out = tf.keras.preprocessing.image.load_img(path=os.path.join(folder,str(nom)+".tif"))  
            except FileNotFoundError:
                out = tf.keras.preprocessing.image.load_img(path=os.path.join(folder,str(nom)+".jpeg"))
                
    out = tf.keras.preprocessing.image.img_to_array(out)
    if cropp:
        out=crop_center(out,taille_output,taille_output)
    return(out)
