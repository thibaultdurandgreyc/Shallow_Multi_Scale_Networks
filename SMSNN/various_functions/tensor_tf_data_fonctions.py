from __future__ import division
import tensorflow as tf

# Convert ---
def tensor_ycbcr2rgb(x):
    '''
    convert tensor
    [0,1] ycbcr -> [0,255]  rgb
    
    * x (tf) : input tensor
    Returns
    true_tf_tensor (tf) : tensor ready to be fed as input for keras.application model like VGG16. (after preprocessing)
    '''
    y, u, v = tf.split(x, 3, axis=-1)  # correct range https://github.com/tensorflow/tensorflow/issues/37067
    target_u_min, target_u_max = -0.43601035 , 0.43601035
    target_v_min, target_v_max = -0.61497538, 0.61497538
                
    u = u * (target_u_max - target_u_min) + target_u_min
    v = v * (target_v_max - target_v_min) + target_v_min
    preprocessed_yuv_images = tf.concat([y, u, v], axis=-1)
    rgb_tensor_images = tf.image.yuv_to_rgb(preprocessed_yuv_images)

    true_tf_tensor= rgb_tensor_images*255. # preprocess (from keras.application) takes [0,255] rgb input
    return(true_tf_tensor)

def tf_rgb2ycbcr(rgb):
    """
    convert tensor
    [0,1] rgb -> [0,255]  ycbcr
    """
    rgb=rgb*255.
    r, g, b = tf.unstack(rgb, 3, axis=3)

    y = r * 0.299 + g * 0.587 + b * 0.114
    cb = r * -0.1687 - g * 0.3313 + b * 0.5
    cr = r * 0.5 - g * 0.4187 - b * 0.0813

    cb += 128
    cr += 128

    ycbcr = tf.stack((y, cb, cr), axis=3)
    return ycbcr


def tf_ycbcr2rgb(ycbcr):
    """
    convert tensor
    [0,255] ycbcr -> [0,255]  rgb
    """
    y, cb, cr = tf.unstack(ycbcr, 3, axis=3)

    cb -= 128
    cr -= 128

    r = y * 1. + cb * 0. + cr * 1.402
    g = y * 1. - cb * 0.34414 - cr * 0.71414
    b = y * 1. + cb * 1.772 + cr * 0.

    rgb = tf.stack((r, g, b), axis=3)
    return rgb

# Preprocessing / Deprocessing tensors ---
def crop_center_tf(img,cropx,cropy):
    """
    Crop img tf 
    
    * img (tf) : [width,height,n]  with n = 1 or n = 3
    * cropx (int) : x size for the output
    * cropy (int): y size for the output
    
    Returns
    ** img (tf) : cropped tensor
    """
    x = img.shape[1]
    y = img.shape[2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    if len(img.shape)==4: # 3 channels
        return img[:,startx:startx+cropx , starty:starty+cropy,:]
    else: #  2 channels
        return img[:,startx:startx+cropx , starty:starty+cropy]
    
