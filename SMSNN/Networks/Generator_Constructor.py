from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os  
import tensorflow as tf
from various_functions.tensor_tf_data_fonctions import *
from various_functions.numpy_data_fonctions import *

# 0.Testing generators ---
def create_dataset_tf(img_folder,taille_output, channels, nombre_patch_test):
    '''
    Stacks images from a folder into an array [ batch, W, H , 3]
    
    * img_folder (str) : data folder 
    * taille_output (int) : common size of extracted data (used for patches of the same size during test)
    * channels (int) : number of channels of the data (1 for grayscale or 3 for rgb/ycbcr)
    * nombre_patch_test (int) : number of patches to extract on wich test will be perform
    
    '''
    name=[]
    tf_img_data_array=[]  
    compteur=0
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            if compteur<nombre_patch_test:
                image= os.path.join(img_folder,dir1, file)
                image = tf.io.read_file(image)
                image = tf.io.decode_png(image, channels=channels)
                image = tf.image.resize(image, (taille_output,taille_output))
                image = tf.cast(image / 255., tf.float32)
                tf_img_data_array.append(image)
                name.append(file)
            compteur+=1
    return tf.stack(tf_img_data_array, axis=0),name 

def open_all_batches(main_network:str,input_lr:bool,taille:int,border:int,nombre_patch_test:int,root:str):
    '''
    Opens Test patches from DIV2K database in tensors conatining all testing data ; out of a testing images folder.
    
    * main_network (str) : nature of the MAIN NN ('SR', 'DENOISING' or 'BLURRING')
    * input_lr (bool) : if True : input data should be loaded before not interpolated data 
    * nombre_patch_test (int) : number of patches to extract on wich test will be perform
    * border (int) : size of the border 
    * root (str) : path to the folder / Patchs/... where data are saved
    '''
    if input_lr:
        lr_tf = create_dataset_tf((os.path.join(os.path.join(root,"Patchs"),"test_"+str(taille)+"/LR_ycbcr")),int(taille/R)+2*border, 3, nombre_patch_test)[0]
    else:
        lr_tf = create_dataset_tf((os.path.join(os.path.join(root,"Patchs"),"test_"+str(taille)+"/LR_bilinear_ycbcr")),taille+2*border, 3, nombre_patch_test)[0]
    
    bicubic_ycbcr_tf = create_dataset_tf((os.path.join(os.path.join(root,"Patchs"),"test_"+str(taille)+"/LR_bilinear_ycbcr")),taille+2*border, 3, nombre_patch_test)[0]
    bicubic_y_tf = create_dataset_tf((os.path.join(os.path.join(root,"Patchs"),"test_"+str(taille)+"/LR_bilinear")),taille+2*border, 1, nombre_patch_test)[0]
    
    true_tuple = create_dataset_tf((os.path.join(os.path.join(root,"Patchs"),"test_"+str(taille)+"/HR_ycbcr")),taille, 3, nombre_patch_test)
    true_tf=true_tuple[0]
    
    true_tuple_nocrop = create_dataset_tf((os.path.join(os.path.join(root,"Patchs"),"test_"+str(taille)+"/HR_ycbcr_nocrop")),taille+2*border, 3, nombre_patch_test)
    true_tf_nocrop=true_tuple_nocrop[0]
    
    noms_tf = true_tuple[1]
    
    return(lr_tf,bicubic_ycbcr_tf,bicubic_y_tf,true_tf,noms_tf,true_tf_nocrop)

def Generator_test_patch(main_network:str, size_output:int, border:int, root:str): 
    """
    Creates Testing generator out of testing image folders where test images are saved. 
    
    * main_network (str) : nature of the MAIN NN ('SR', 'DENOISING' or 'BLURRING')
    * size_output (int) : common size of extracted data (used for patches of the same size during test)
    * border (int) : size of the border 
    * root (str) : path to the folder / Patchs/... where data are saved

    """
    if main_network=="SR":
        input_path_data, output_path_data = "/LR_bilinear_ycbcr" , "/HR_ycbcr" # SR  ; bicubic -> HR
        size_input=size_output+2*border
    elif main_network in ["DENOISING","BLURRING"]:
        input_path_data, output_path_data = "/HR_ycbcr_nocrop" , "/HR_ycbcr" # DENOISING  ; noisy HR -> HR  /// BLURRING  ; blurred HR -> HR
        size_input=size_output+2*border
    elif main_network=="SR_EDSR":
        input_path_data, output_path_data = "/LR_ycbcr" , "/HR_ycbcr" # SR  ; LR -> HR (trained)
        size_input = int((size_output)/4)+2*border
    datagen_test = ImageDataGenerator(rescale=1./255)   
    
    # Input Batch of 1 (to be noised if denoising task)
    INPUT = datagen_test.flow_from_directory(os.path.join(os.path.join(root,"Patchs"),"test_"+str(size_output)+str(input_path_data)),target_size = (size_input,size_input),class_mode = None,color_mode="rgb",batch_size = 1,shuffle=True, seed=320)
    # Output Batch of 1
    OUTPUT = datagen_test.flow_from_directory(os.path.join(os.path.join(root,"Patchs"),"test_"+str(size_output)+str(output_path_data)),target_size = (size_output,size_output),class_mode = None,color_mode="rgb",batch_size = 1,shuffle=True, seed=320)
    return(INPUT,OUTPUT)
    
# 2. Tensorflow Preprocessing in generators ---
def noise(array,sigma_noise_blur): 
    """
    Adds random noise to each image in the tensor 'array'
    
    * array (tensor) : shape [batch_size, width, height,3] in the general cases
                        or shape [ width, height,3] if testing
                        [0,1] ycbcr (noise on y channel)
                        
    Returns
    ** Generator
    """
   
    noise = tf.random.normal(shape=tf.shape(array), mean=0.0, stddev=sigma_noise_blur, dtype=tf.float32)
    try: # batch - training
        array[:,:,:,0] = array[:,:,:,0] + noise[:,:,:,0]
    except IndexError : # array - testing
        array[:,:,0] = array[:,:,0] + noise[:,:,0]
    array = tf.clip_by_value(array, 0.0, 1.0)
    return np.clip(array, 0.0, 1.0)

def gaussian_blur(img, kernel_size=11, sigma=5):
    '''
    Applies a gaussian blur on a specific tensor, for each element of the batch
    '''
    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel
    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],padding='SAME', data_format='NHWC')

# 2. Validation/training generators
def Generator_patch(main_network:str, data_role:str, size_type_input : tuple, size_type_output:tuple, taille:int,loss_pixel:bool, loss_perceptuelle:list,loss_style:list, nombre_class:int,border:int,root:str,batch_size :int, sigma_noise_blur:float,shuffle=False):
    """
    Creates Validation generator out of validation image folders where images are saved. 
    
    * main_network (str) : nature of the MAIN NN ('SR', 'DENOISING' or 'BLURRING')
    * data_role (str) : either 'test' 'validation' or 'training' for loading specific folder
    * size_output (int) : common size of extracted data (used for patches of the same size during test)
    * border (int) : size of the border 
    * sigma_noise_blur (int) : std deviation for building   a.  noisy data   b. blurry data (if main_network is 'BLURRING' or 'DENOISING')
    * root (str) : path to the folder / Patchs/... where data are saved
    * loss_pixel, loss_perceptuelle,loss_style (bools) : booleans, for loading the data as many times as needed during loss computation
    
    Returns
    ** Generator
    """
    if main_network=="SR":
        input_path_data, output_path_data = "/LR_bilinear_ycbcr" , "/HR_ycbcr" # SR  ; bicubic -> HR
    elif main_network in ["DENOISING","BLURRING" ]:
        input_path_data, output_path_data = "/HR_ycbcr_nocrop" , "/HR_ycbcr" # DENOISING  ; noisy HR -> HR  /// BLURRING  ; blurred HR -> HR
    elif main_network=="SR_EDSR":
        input_path_data, output_path_data = "/LR_ycbcr" , "/HR_ycbcr" # SR  ; LR -> HR (trained)
        size_inp_1,size_inp_2 = int((size_type_input[0]-2*border)/4),int((size_type_input[1]-2*border)/4)
        size_type_input = ( size_inp_1,size_inp_2 )

    datagen_test = ImageDataGenerator(rescale=1/255.)  

    INPUT = datagen_test.flow_from_directory(os.path.join(os.path.join(root,"Patchs"),str(data_role)+"_"+str(taille)+str(input_path_data)),target_size = size_type_input,class_mode = None,color_mode="rgb",batch_size = batch_size,shuffle=shuffle, seed=320)
    OUTPUT = datagen_test.flow_from_directory(os.path.join(os.path.join(root,"Patchs"),str(data_role)+"_"+str(taille)+str(output_path_data)),target_size = size_type_output,class_mode = None,color_mode="rgb",batch_size = batch_size,shuffle=shuffle, seed=320)

        
    if loss_pixel:
        nombre_composante_loss = (len(loss_perceptuelle)+len(loss_style)+1)
            
    else:
        nombre_composante_loss = ( len(loss_perceptuelle)+len(loss_style)   )
    
    while True:  
        # Input Batch
        INPUTi =  INPUT.next()  
        if main_network=="DENOISING":
            INPUTi = noise(INPUTi,sigma_noise_blur)
        elif main_network =="BLURRING":
            INPUTi = gaussian_blur(INPUTi, kernel_size=11, sigma=sigma_noise_blur)
        # Output Batch
        OUTPUTi = OUTPUT.next()
        
        # Yielding as many batches as needed
        yield [INPUTi[0:batch_size]],[OUTPUTi[0:batch_size] for nbre in range (nombre_composante_loss)]
   

            