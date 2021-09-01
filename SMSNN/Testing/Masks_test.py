from os import listdir
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorboard import *

from various_functions.tensor_tf_data_fonctions import *
from various_functions.rapports_pdf_fonctions import *
from various_functions.numpy_data_fonctions import *
from various_functions.directory_management import *
from various_functions.custom_filters import *

# Architectures
from Networks.MAIN.architecture_MAIN import *
from Networks.STr_3.architecture_Str_3 import *
from Networks.STr_col.architecture_Str_col import *
from Networks.STr_y.architecture_Str_y import *
from Networks.Generator_Constructor import *
from Networks.Compilation_NN import *
from Networks.Loss_Constructor import *
from Networks.model_management import *

from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input


import matplotlib.pyplot as plt
import numpy as np
#from scipy.misc import imresize
import cv2
import imageio

import os  
import tensorflow as tf

import ISR #https://github.com/idealo/image-super-resolution
from ISR.models import RDN,RRDN
from codes_externes.model import get_G, get_D  #https://modelzoo.co/model/srgan
#Generateur_SRGAN imported from network.py

def zoom(img):
    '''
    Zoom localizations and weights per chosen styles for different DIV2K training and testing images
    '''    
        
    if img=="duck.png":  #  
        inp_x,inp_x_fin,inp_y,inp_y_fin=0,160,0,160   
        
    if img=="lena.png":  #   #474.png
        inp_x,inp_x_fin,inp_y,inp_y_fin=0,400,0,400   

    if img=="papillon.png":  #   #474.png
        inp_x,inp_x_fin,inp_y,inp_y_fin=0,256,0,256   
        
    if img=="singe.png":  #   #474.png
        inp_x,inp_x_fin,inp_y,inp_y_fin=20,280,20,280   
        
    if img=="enfant.png":  #   #474.png
        inp_x,inp_x_fin,inp_y,inp_y_fin=0,280,0,280   
        
    if img=="visage.png":  #   #FFHQ 10193.png
        inp_x,inp_x_fin,inp_y,inp_y_fin=200,500,300,600   

    if img=="zebre.png":  #   #FFHQ 10193.png
        inp_x,inp_x_fin,inp_y,inp_y_fin=50,250,50,250   
        
    if img=="visage2.png":  #   #FFHQ 10193.png
        inp_x,inp_x_fin,inp_y,inp_y_fin=200,500,300,600   
        
    if img=="dindon.png":  #   #BSD 100 80.png
        inp_x,inp_x_fin,inp_y,inp_y_fin=100,300,60,260
    
    if img=="graine.png":  #   #BSD 100 84.png
        inp_x,inp_x_fin,inp_y,inp_y_fin=0,320,0,480
    # TRAIN DATA
    
    if img=="indien.png":  #   0359.png de div2K
        inp_x,inp_x_fin,inp_y,inp_y_fin=300,800,800,1200
    
    if img=="ours.png":  #  0659.png
        inp_x,inp_x_fin,inp_y,inp_y_fin=100,800,600,1200 
    
    if img=="old.png":  #   #474.png
        inp_x,inp_x_fin,inp_y,inp_y_fin=400,900,200,700   
        
    if img=="table.png":  #   0922.png 
        inp_x,inp_x_fin,inp_y,inp_y_fin=40,220,40,220
    '''
    # TEST DATA
    if img=="0972.png":  #   HOMME D AFFAIRE
        inp_x,inp_x_fin,inp_y,inp_y_fin=28,150,28,150
            
    if img=="0910.png":  #  HOMME D AFFAIRE
        inp_x,inp_x_fin,inp_y,inp_y_fin=70,240,70,240
     
    if img=="0787.png":  #   BIBLE EXEMPLE 1 PAPIER
        ponderation_inter_styles =  [0.54,0.65,0.22]
        
    if img=="0759.png":  #   CHIEN QUI DORT EXEMPLE 2 PAPIER 
        ponderation_inter_styles =  [0.87,1]

    if img=="0728.png":  #   JEUNE laine
        ponderation_inter_styles =  [0.3,0.55]
    '''
    if img=="livre.png":  #   0787.png de DIV2K    y_bis = [30]     x_bis = [0]     delta=210
        inp_x,inp_x_fin,inp_y,inp_y_fin=530,1030,560,1060   
       
    if img=="chien.png":  #   #0.759 de div2K
        inp_x,inp_x_fin,inp_y,inp_y_fin=360,860,800,1200
    
    if img=="baby.png":  #   CHIEN QUI DORT EXEMPLE 2 PAPIER 
        inp_x,inp_x_fin,inp_y,inp_y_fin=40,400,40,400

    if img=="cookie.png":  #   les COOKIES
        inp_x,inp_x_fin,inp_y,inp_y_fin=40,400,40,400
        
    return(inp_x,inp_x_fin,inp_y,inp_y_fin)

def get_list(available_styles_path:str,start_name_style:str,save_rep:str):
    """
    Gives the complete list intersection between available styles and trained styles
    """
    all_styles= [x for x in os.listdir(available_styles_path) if os.path.isdir(os.path.join(available_styles_path,x)) ]
    all_styles=sorted(all_styles)
    trained = [x.replace(str(start_name_style)+"_","").replace(".h5","") for x in os.listdir(os.path.join(save_rep,"final_saved")) if x.endswith(".h5") and x.startswith(start_name_style)]
    trained=sorted(trained)
    names = [i.replace(start_name_style,"") for i in trained]
    
    # Intersection between lists
    not_learned=[ element for element in all_styles if element not in names]
    all_styles_sr = list(set(all_styles).intersection(set(names)))
    trained=[start_name_style+"_"+str(i) for i in all_styles_sr]
    names = [i.replace(start_name_style,"") for i in all_styles_sr]    
    return(trained,names)
    
def Composition(main_network:str,ponderation_features:list, border:int,filtres_sr:int,filtres_st:int,filtres_col:int,filtres_st3:int, kernel:int,save_rep:str,
                ouverture:float,BN_init:bool,BN_fin:bool,DOG_init:bool, DOG_fin:bool,  sigma_noise_blur:float,benchmark_folder:str,  w_h,w_v, w_h_s,w_v_s):
    

    """
    Build different Image outputs out of MAIN MODEL(either 'SR' 'DENOISING' or 'BLURRING') and BRANCHE MODELS ('St(y)','St(ycbcr)', 'St(col)' or 'St3')
    
    * main_network (str) : nature of the MAIN NN ('SR', 'DENOISING' or 'BLURRING')
    * save_rep (str) : path to the folder where are stored all the models 
    * benchmark_folder (str) : path where are stored all the benchmark data 
    
    * filtres_sr, filtres_st, filtres_col, filtres_st3 (int) : number of filters used per convolution for the 4 branch NN
    * kernel (int ) : kernel size per convolution 
    * ouverture (int) : difference between the 2 variance defining the high frequency DoG put at the end of the St(y) & St(ycbcr) branches
    * BN_init,BN_fin,DOG_init, DOG_fin (bools) : if True, adds BN layers or DoG layers
    
    * border (int) : size of the border 
    * sigma_noise_blur (int) : std deviation for building   a.  noisy data   b. blurry data (if main_network is 'BLURRING' or 'DENOISING')
    * w_h,w_v, w_h_s,w_v_s (tensors) : stores 1D Gaussian filters for building DoGs
    
    Returns
    ** None (Save Image in rep_save/final_saved/...)
    
    """ 
    # 0. Image list
    img_list = [x for x in os.listdir(os.path.join(benchmark_folder,"BENCHMARK_VISUEL/Masks")) if x.endswith(".png")]
    
    # 1. LOADING MODELS
    # MAIN NN ---
    tf.keras.backend.clear_session()
    model_identite=tf.keras.models.load_model(os.path.join(os.path.join(save_rep,"final_saved"),"model_MAIN.h5"),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False)
    new_model  = MAIN_network_none(filtres=filtres_sr, ponderation_features=ponderation_features,kernel=kernel,w_h=w_h,w_v=w_v, BN_init=BN_init, BN_fin=BN_fin, DOG_init=DOG_init, DOG_fin=DOG_fin)
    new_model =copy_parameters(model=new_model,model_patch=model_identite)
    
    # I. St(y) residual branches ---
    Trained_STy,Names_STy = get_list(available_styles_path = os.path.join(benchmark_folder,"SELECTED_STYLES"),start_name_style = "model_St(y)",save_rep=save_rep)
    list_modele_STY=[]# chargement des graphs
    for count,style_from_all in enumerate(Trained_STy):
        tf.keras.backend.clear_session()
        model_st=tf.keras.models.load_model(os.path.join(os.path.join(save_rep,"final_saved"),str(style_from_all)+".h5"),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False)
        list_modele_STY.append(model_st)
        
    # II. St(y,cb,cr) residual branches ---
    Trained_STycbcr,Names_STycbcr = get_list(available_styles_path = os.path.join(benchmark_folder,"SELECTED_STYLES"),start_name_style = "model_St(y)_col",save_rep=save_rep)
    list_modele_STYcbcr=[]# chargement des graphs
    for count,style_from_all in enumerate(Trained_STycbcr):
        tf.keras.backend.clear_session()
        model_st=tf.keras.models.load_model(os.path.join(os.path.join(save_rep,"final_saved"),str(style_from_all)+".h5"),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False)
        list_modele_STYcbcr.append(model_st)

    # III. St(col) residual branches ---
    Trained_STcol,Names_STcol = get_list(available_styles_path = os.path.join(benchmark_folder,"SELECTED_COLORS"),start_name_style = "model_ST(col)",save_rep=save_rep)
    list_modele_STcol=[]# chargement des graphs
    for count,style_from_all in enumerate(Trained_STcol):
        tf.keras.backend.clear_session()
        model_st=tf.keras.models.load_model(os.path.join(os.path.join(save_rep,"final_saved"),str(style_from_all)+".h5"),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False)
        list_modele_STcol.append(model_st)        
        
   # IV. St(col) residual branches ---
    Trained_ST3,Names_ST3 = get_list(available_styles_path = os.path.join(benchmark_folder,"SELECTED_STYLES_ST3"),start_name_style = "model_ST3",save_rep=save_rep)
    list_modele_ST3=[]# chargement des graphs
    for count,style_from_all in enumerate(Trained_ST3):
        tf.keras.backend.clear_session()
        model_st=tf.keras.models.load_model(os.path.join(os.path.join(save_rep,"final_saved"),str(style_from_all)+".h5"),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False)
        list_modele_ST3.append(model_st)  
    
    
    # INFEREING ON EACH IMAGE
    for img in img_list:
        print("image :"+str(img))
        # Folder for specific image
        rep_img_save = os.path.join(save_rep,"final_saved")
        rep_base_img = os.path.join(rep_img_save,img)           
        ensure_dir(rep_base_img)
        
        Comparison=os.path.join(rep_base_img,"Other_models")    
        ensure_dir(Comparison)
        
        Resdiual_outputs=os.path.join(rep_base_img,"Resdiual_outputs")    
        ensure_dir(Resdiual_outputs)
        
        Random_combinations=os.path.join(rep_base_img,"Random_combinations")    
        ensure_dir(Random_combinations)
        
        # zoom specific
        x_m,x_max,y_m,y_max = zoom(img)
        print(x_m,x_max,y_m,y_max)
        print(x_m/4,x_max/4,y_m/4,y_max/4)
        nom_mask=[x.replace(".png","").replace("mask_","") for x in listdir(os.path.join(benchmark_folder,"BENCHMARK_VISUEL/Masks/masks_array/"+str(img.replace(".png",""))+"/")) if x.endswith(".png") ] 
        
        # Data Preparation & Input
        LR = Ouverture_img(os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(benchmark_folder,"BENCHMARK_VISUEL/Masks"),"LR"),img.replace(".png","_LR.png")))),1)[0]
        
        LR=LR[int(x_m/4):int(x_max/4),int(y_m/4):int(y_max/4),:]/256.
        HR=Ouverture_img(os.path.join(os.path.join(os.path.join(os.path.join(benchmark_folder,"BENCHMARK_VISUEL/Masks"),img))),1)[0]
        HR=HR[x_m:x_max,y_m:y_max,:]/255.
        
        
        if main_network=="SR":
            INPUT = upsampling(LR,4) 
        elif main_network=="DENOISING":
            INPUT = HR.copy()
            INPUT[:,:,0]+=np.random.normal(0,sigma_noise_blur,(HR.shape[0],HR.shape[1]))
        elif main_network=="BLURRING":
            INPUT = HR.copy()
            INPUT_y = gaussian_filter(INPUT[:,:,0], sigma=sigma_noise_blur)
            INPUT[:,:,0]=INPUT_y
        INPUT=np.expand_dims(INPUT,axis=0)
        TRUE=np.expand_dims(HR,axis=0)
        
        # 1. Data        
        LR=cv2.imread(os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(benchmark_folder,"BENCHMARK_VISUEL/Masks"),"LR"),img.replace(".png","_LR.png")))))
        LR=cv2.cvtColor(LR, cv2.COLOR_BGR2RGB)
        LR=LR[int(x_m/4):int(x_max/4),int(y_m/4):int(y_max/4),:] 
        
        bilinear_global = upsampling(LR,4) 
        
        HR=cv2.imread(os.path.join(os.path.join(os.path.join(os.path.join(benchmark_folder,"BENCHMARK_VISUEL/Masks"),img)))) 
        HR_rgb=cv2.cvtColor(HR, cv2.COLOR_BGR2RGB)
        HR=HR[x_m:x_max,y_m:y_max,:]
       
        LR_ycbcr=RGB2YCbCr(LR)
        HR_ycbcr=RGB2YCbCr(HR)
        interpolation=RGB2YCbCr(bilinear_global)
        interpolation_crop=crop_center(interpolation,interpolation.shape[0]-2*border,interpolation.shape[1]-2*border)
        
        '''
        # VDSR
        print("VDSR")
        tf.keras.backend.clear_session()
        model_patch=(os.path.join(benchmark_folder,"MODEL/vdsr.h5"))       
        new_model = vdsr_sofa(sizex=interpolation.shape[0]-2*border,sizey=interpolation.shape[1]-2*border)         
        new_model.load_weights(model_patch)       
        vdsr_out=new_model.predict([(interpolation[:,:,0]/255.).reshape(1,interpolation.shape[0],interpolation.shape[1],1)])               
        vdsr_out=vdsr_out.reshape(vdsr_out.shape[1],vdsr_out.shape[2],1)
        vdsr_out_ycbcr = np.concatenate([vdsr_out,(interpolation_crop[:,:,1].reshape(interpolation_crop.shape[0],interpolation_crop.shape[1],1))/255.,(interpolation_crop[:,:,2].reshape(interpolation_crop.shape[0],interpolation_crop.shape[1],1))/255.],axis=-1)
        vdsr_out_ycbcr=vdsr_out_ycbcr.reshape(vdsr_out_ycbcr.shape[0],vdsr_out_ycbcr.shape[1],3)*255.
        imageio.imwrite(os.path.join(Comparison,str(img)+str("_vdsr.png")),YCBCbCr2RGB(vdsr_out_ycbcr).astype(np.uint8))
  
        # SRCNN
        print("SRCNN")
        tf.keras.backend.clear_session()
        model_patch=(os.path.join(benchmark_folder,"MODEL/srcnn.h5"))       
        new_model = srcnn_sofa(sizex=interpolation.shape[0]-2*border,sizey=interpolation.shape[1]-2*border)         
        new_model.load_weights(model_patch)       
        srcnn_out=new_model.predict([(interpolation[:,:,0]/255.).reshape(1,interpolation.shape[0],interpolation.shape[1],1)])               
        srcnn_out=srcnn_out.reshape(srcnn_out.shape[1],srcnn_out.shape[2],1)
        srcnn_out_ycbcr = np.concatenate([srcnn_out,(interpolation_crop[:,:,1].reshape(interpolation_crop.shape[0],interpolation_crop.shape[1],1))/255.,(interpolation_crop[:,:,2].reshape(interpolation_crop.shape[0],interpolation_crop.shape[1],1))/255.],axis=-1)
        srcnn_out_ycbcr=srcnn_out_ycbcr.reshape(srcnn_out_ycbcr.shape[0],srcnn_out_ycbcr.shape[1],3)*255.
        imageio.imwrite(os.path.join(Comparison,str(img)+str("_srcnn.png")),YCBCbCr2RGB(srcnn_out_ycbcr).astype(np.uint8))
        
        # SRFEAT
        print("SRFEAT")
        srfeat=generator_srfeat(is_train=False, use_bn=True)  
        srfeat.load_weights(os.path.join(benchmark_folder,"MODEL/srfeat-20.h5")) 
        pred_rgb_gan_srfeat=srfeat.predict((LR_ycbcr.reshape(1,LR.shape[0],LR.shape[1],3))) # Git hub uses Ycbcr
        pred_rgb_gan_srfeat=pred_rgb_gan_srfeat.reshape(pred_rgb_gan_srfeat.shape[1],pred_rgb_gan_srfeat.shape[2],3)
        pred_rgb_gan_srfeat=crop_center(pred_rgb_gan_srfeat,pred_rgb_gan_srfeat.shape[0]-2*border,pred_rgb_gan_srfeat.shape[1]-2*border)
        pred_rgb_gan_srfeat=np.clip(pred_rgb_gan_srfeat,0,255)
        imageio.imwrite(os.path.join(Comparison,str(img)+str("_srfeat.png")),YCBCbCr2RGB(pred_rgb_gan_srfeat).astype(np.uint8))
        # ESRGAN
        print("ESRGAN")
        esrgan=generator_esrgan()  
        esrgan.load_weights(os.path.join(benchmark_folder,"MODEL/esrgan-10.h5"))
        pred_rgb_esrgan=esrgan.predict(LR_ycbcr.reshape(1,LR.shape[0],LR.shape[1],3)) # Git hub uses Ycbcr
        pred_rgb_esrgan=pred_rgb_esrgan.reshape(pred_rgb_esrgan.shape[1],pred_rgb_esrgan.shape[2],3)*255.
        pred_rgb_esrgan=crop_center(pred_rgb_esrgan,pred_rgb_esrgan.shape[0]-2*border,pred_rgb_esrgan.shape[1]-2*border)
        pred_rgb_esrgan=np.clip(pred_rgb_esrgan,0,255)
        imageio.imwrite(os.path.join(Comparison,str(img)+str("_esrgan.png")),YCBCbCr2RGB(pred_rgb_esrgan).astype(np.uint8))
        
        # RRDN
        print("RRDN")
        rrdn = RRDN(weights='gans') #psnr-large
        pred_rgb_rrdn=rrdn.predict(LR.reshape(LR.shape[0],LR.shape[1],3))
        pred_rgb_rrdn = pred_rgb_rrdn.reshape(pred_rgb_rrdn.shape[0],pred_rgb_rrdn.shape[1],3)
        pred_rgb_rrdn=crop_center(pred_rgb_rrdn,pred_rgb_rrdn.shape[0]-2*border,pred_rgb_rrdn.shape[1]-2*border)
        imageio.imwrite(os.path.join(Comparison,str(img)+str("_rdn.png")),pred_rgb_rrdn.astype(np.uint8))
        
        # EDSR
        print("EDSR")
        ed =  edsr_sofa(scale=4, num_res_blocks=16)
        ed.load_weights(os.path.join(benchmark_folder,"MODEL/weights_github_krasserm/weights-edsr/weights-edsr-16-x4/weights/edsr-16-x4/weights.h5"))
        inputt = tf.cast(LR.reshape(1,LR.shape[0],LR.shape[1],3),dtype=tf.float32)
        pred_rgb_edsr=ed.predict(inputt/255.)
        pred_rgb_edsr = pred_rgb_edsr.reshape(pred_rgb_edsr.shape[1],pred_rgb_edsr.shape[2],3)
        pred_rgb_edsr=crop_center(pred_rgb_edsr,pred_rgb_edsr.shape[0]-2*border,pred_rgb_edsr.shape[1]-2*border)
        pred_rgb_edsr=np.clip(pred_rgb_edsr,0,255)
        imageio.imwrite(os.path.join(Comparison,str(img)+str("_edsr.png")),pred_rgb_edsr.astype(np.uint8))
        pred_ycbcr_edsr = (RGB2YCbCr(pred_rgb_edsr)/255.).astype(np.float64)
        '''
        #  Masks Loading
        list_masks=[]
        for st in range(len(nom_mask)):
            index=nom_mask[st]
            mask=(cv2.imread(os.path.join(benchmark_folder,"BENCHMARK_VISUEL/Masks/masks_array/"+str(img.replace(".png",""))+"/mask_"+str(index)+".png"),cv2.IMREAD_GRAYSCALE)/255.)[x_m:x_max,y_m:y_max]
            imageio.imwrite(os.path.join(rep_base_img,str("mask_"+str(st)+".png")),(mask*255.).astype(np.uint8)) # save local mask
            
            list_mask=[]
            #mask =lissage_mask(mask,12,0.04)
            mask=crop_center(mask,mask.shape[0]-2*border,mask.shape[1]-2*border)
            list_mask.append(mask)
            list_masks.append(list_mask) 
        
        # MAIN MODEL
        MAIN_feature = new_model.predict([INPUT]) 
        MAIN = MAIN_feature.reshape(MAIN_feature.shape[1],MAIN_feature.shape[2],3)

        # Residual Outputs --- St(y) ---
        list_residu_STy=[]
        print(Trained_STy, "y")
        for count,model_STy in enumerate(Trained_STy):
            model_STy_h5=tf.keras.models.load_model(os.path.join(os.path.join(save_rep,"final_saved"),model_STy+".h5"),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False)
            model_STy_h5 = model_STy_h5.predict([MAIN_feature]) 
            feature_STy = model_STy_h5.reshape(model_STy_h5.shape[1],model_STy_h5.shape[2],1)
            list_residu_STy.append(feature_STy) 
            residu_STy_display=feature_STy.copy()
            residu_STy_display[:,:,0] =( ((residu_STy_display[:,:,0] + 1) ) *255.  ) - 127
            imageio.imwrite(os.path.join(Resdiual_outputs,str(img.replace(".png",""))+"_"+str(model_STy)+str(".png")),(residu_STy_display.reshape(residu_STy_display.shape[0],residu_STy_display.shape[1],1)).astype(np.uint8)) 
        
        # Residual Outputs --- St(y,cbcr) ---
        list_residu_STycbcr_y,list_residu_STycbcr_cb,list_residu_STycbcr_cr=[],[],[]
        print(Trained_STycbcr, "ycbcr")
        for count,model_STycbcr in enumerate(Trained_STycbcr):
            model_STycbcr_h5=tf.keras.models.load_model(os.path.join(os.path.join(save_rep,"final_saved"),model_STycbcr+".h5"),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False)
            model_STycbcr_h5 = model_STycbcr_h5.predict([INPUT]) 
            
            feature_STycbcr_y = model_STycbcr_h5[0].reshape(model_STycbcr_h5[0].shape[1],model_STycbcr_h5[0].shape[2])
            feature_STycbcr_cb = model_STycbcr_h5[1].reshape(model_STycbcr_h5[1].shape[1],model_STycbcr_h5[1].shape[2])
            feature_STycbcr_cr = model_STycbcr_h5[2].reshape(model_STycbcr_h5[2].shape[1],model_STycbcr_h5[2].shape[2])

            list_residu_STycbcr_y.append(feature_STycbcr_y) 
            list_residu_STycbcr_cb.append(feature_STycbcr_cb) 
            list_residu_STycbcr_cr.append(feature_STycbcr_cr) 
            
            residu_STycbcr_display=feature_STycbcr_y.copy()
            residu_STycbcr_display =( ((residu_STycbcr_display + 1) ) *255.  ) - 127
            imageio.imwrite(os.path.join(Resdiual_outputs,str(img.replace(".png",""))+"_"+str(model_STycbcr)+str(".png")),(residu_STycbcr_display.reshape(residu_STycbcr_display.shape[0],residu_STycbcr_display.shape[1],1)).astype(np.uint8)) 
            
        # Residual Outputs --- St(col) ---
        list_residu_STcol_cb,list_residu_STcol_cr=[],[]
        print(Trained_STcol, "stcol")
        for count,model_STcol in enumerate(Trained_STcol):
            model_STcol_h5=tf.keras.models.load_model(os.path.join(os.path.join(save_rep,"final_saved"),model_STcol+".h5"),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False)
            model_STcol_h5 = model_STcol_h5.predict([MAIN_feature]) 
            feature_STcol_cb = model_STcol_h5[0].reshape(model_STcol_h5[0].shape[1],model_STcol_h5[0].shape[2])
            feature_STcol_cr = model_STcol_h5[1].reshape(model_STcol_h5[1].shape[1],model_STcol_h5[1].shape[2])
            list_residu_STcol_cb.append(feature_STcol_cb) 
            list_residu_STcol_cr.append(feature_STcol_cr) 
            residu_STcol_cb_display,residu_STcol_cr_display=feature_STcol_cb.copy(),feature_STcol_cr.copy()
            residu_STcol_cb_display,residu_STcol_cr_display =( ((residu_STcol_cb_display + 1) ) *255.  ) - 127  , ( ((residu_STcol_cr_display + 1) ) *255.  ) - 127 
            imageio.imwrite(os.path.join(Resdiual_outputs,str(img.replace(".png",""))+"_"+str(model_STcol)+str("_cb_.png")),(residu_STcol_cb_display.reshape(residu_STcol_cb_display.shape[0],residu_STcol_cb_display.shape[1],1)).astype(np.uint8)) 
            imageio.imwrite(os.path.join(Resdiual_outputs,str(img.replace(".png",""))+"_"+str(model_STcol)+str("_cr_.png")),(residu_STcol_cr_display.reshape(residu_STcol_cb_display.shape[0],residu_STcol_cb_display.shape[1],1)).astype(np.uint8)) 
        
        # Residual Outputs --- St3 ---
        list_residu_ST3_y,list_residu_ST3_cb,list_residu_ST3_cr=[],[],[]
        print(Trained_ST3 ,"st3")
        for count,model_ST3 in enumerate(Trained_ST3):
            model_ST3_h5=tf.keras.models.load_model(os.path.join(os.path.join(save_rep,"final_saved"),model_ST3+".h5"),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False)
            model_ST3_h5 = model_ST3_h5.predict([MAIN_feature]) 

            feature_ST3_y = model_ST3_h5[0].reshape(model_ST3_h5[0].shape[1],model_ST3_h5[0].shape[2])
            
            feature_ST3_cb = model_ST3_h5[1].reshape(model_ST3_h5[1].shape[1],model_ST3_h5[1].shape[2])
            feature_ST3_cr = model_ST3_h5[2].reshape(model_ST3_h5[2].shape[1],model_ST3_h5[2].shape[2])
            list_residu_ST3_y.append(feature_ST3_y) 
            list_residu_ST3_cb.append(feature_ST3_cb) 
            list_residu_ST3_cr.append(feature_ST3_cr) 
            residu_ST3_y_display,residu_ST3_cb_display,residu_ST3_cr_display=feature_ST3_y.copy(),feature_ST3_cb.copy(),feature_ST3_cr.copy()
            residu_ST3_y_display,residu_ST3_cb_display,residu_ST3_cr_display =( ((residu_ST3_y_display + 1) ) *255.  ) - 127,( ((residu_ST3_cb_display + 1) ) *255.  ) - 127,( ((residu_ST3_cr_display + 1) ) *255.  ) - 127
            imageio.imwrite(os.path.join(Resdiual_outputs,str(img.replace(".png",""))+"_"+str(model_ST3)+str("_y_.png")),(residu_ST3_y_display.reshape(residu_ST3_y_display.shape[0],residu_ST3_y_display.shape[1],1)).astype(np.uint8))             
            imageio.imwrite(os.path.join(Resdiual_outputs,str(img.replace(".png",""))+"_"+str(model_ST3)+str("_cb_.png")),(residu_ST3_cb_display.reshape(residu_ST3_cb_display.shape[0],residu_ST3_cb_display.shape[1],1)).astype(np.uint8))             
            imageio.imwrite(os.path.join(Resdiual_outputs,str(img.replace(".png",""))+"_"+str(model_ST3)+str("_cr_.png")),(residu_ST3_cr_display.reshape(residu_ST3_cr_display.shape[0],residu_ST3_cr_display.shape[1],1)).astype(np.uint8))             
        
        
        # 2. Making 'tries' random outputs out of the branch features
        tries=20
        for random_tries in range(tries):
            txt=""
            output=MAIN.copy()*255. 
            for number_mask in range(len(list_masks)): 
                txt=str(txt)+"___Mask:"+str(nom_mask[number_mask])
                
                # Y : from St(y), St(ycbcr) or ST3  / Cb,cr from St(ycbcr), St(col) or ST3
                # 5 possibilities : St(y) ;  St(y) & St(col)  ; St(ycbcr)  ; St(col)  ; ST3
                rdm = np.random.randint(0,6,1)[0]
                #print(rdm)
                
                if rdm==1 or rdm==2: #St(y) / St(y) & St(col)
                    random_style = np.random.randint(0,len(list_modele_STY),1)[0]
                    ponderation_random_style = np.random.randint(10,60,1)/100.
    
                    txt=str(txt)+"_Sty:"+str(Names_STy[random_style])+";"+str(ponderation_random_style)[1:5]
                    
                    f=list_residu_STy[random_style].copy()*255.
                    f=f.reshape(f.shape[0],f.shape[1])  

                    if rdm==2: #St(y) &St(col)
                        random_col = np.random.randint(0,len(list_modele_STcol),1)[0]
                        txt=str(txt)+"_Stcol:"+str(Names_STcol[random_col])
                        
                    for u in range(len(list_masks[number_mask])):
                        f=list_masks[number_mask][u]*f*ponderation_random_style
                        output[:,:,0]+=f 
                        if rdm==2: #St(y) &St(col)
                            cb=list_residu_STcol_cb[random_col].copy()*255.
                            cr=list_residu_STcol_cr[random_col].copy()*255.
                            
                            output[:,:,1]=list_masks[number_mask][u]*cb
                            output[:,:,2]=list_masks[number_mask][u]*cr
                    
                if rdm==3: #St(ycbcr) 3
                    random_style = np.random.randint(0,len(list_modele_STYcbcr),1)[0]
                    ponderation_random_style = np.random.randint(10,60,1)/100.
                    txt=str(txt)+"_Stycbcr:"+str(Names_STycbcr[random_style])+";"+str(ponderation_random_style)[1:5]
                        
                    ponderation_random_style = np.random.randint(10,60,1)/100.
                    f=list_residu_STycbcr_y[random_style].copy()*255.
                    f=f.reshape(f.shape[0],f.shape[1])  
                    cb=list_residu_STycbcr_cb[random_style].copy()*255.
                    cr=list_residu_STycbcr_cr[random_style].copy()*255.
                    f=crop_center(f,f.shape[0]-2*border,f.shape[1]-2*border) #TODO voir s'il faut vraiment crop ou changer l'architecture // ou série
                    cb=crop_center(cb,cb.shape[0]-2*border,cb.shape[1]-2*border)
                    cr=crop_center(cr,cr.shape[0]-2*border,cr.shape[1]-2*border)

                    for u in range(len(list_masks[number_mask])):
                        f=list_masks[number_mask][u]*f*ponderation_random_style
                        output[:,:,0]+=f
                        output[:,:,1]=list_masks[number_mask][u]*cb
                        output[:,:,2]=list_masks[number_mask][u]*cr
                    
                if rdm==4: #St(col)
                    random_col = np.random.randint(0,len(list_modele_STcol),1)[0]
                    txt=str(txt)+"_Stcol:"+str(Names_STcol[random_col])
                    
                    cb=list_residu_STcol_cb[random_col].copy()*255.
                    cr=list_residu_STcol_cr[random_col].copy()*255.
                    for u in range(len(list_masks[number_mask])):
                        output[:,:,1]=list_masks[number_mask][u]*cb
                        output[:,:,2]=list_masks[number_mask][u]*cr
                            
                if rdm==50: #St3 5 /// bug sur les modèles 'None' TODO. input 350 -> 352 en sortie auto encodeur de la branche en série / pour st(y) cropping mais en entrée
                    random_st3 = np.random.randint(0,len(list_modele_ST3),1)[0]
                    txt=str(txt)+"_St3:"+str(Names_ST3[random_st3])
                    y=list_residu_ST3_y[random_st3].copy()*255.
                    cb=list_residu_ST3_cb[random_st3].copy()*255.
                    cr=list_residu_ST3_cr[random_st3].copy()*255.
                    for u in range(len(list_masks[number_mask])):
                        output[:,:,0]=list_masks[number_mask][u]*y
                        output[:,:,1]=list_masks[number_mask][u]*cb
                        output[:,:,2]=list_masks[number_mask][u]*cr

            output=np.clip(output,0,255)
            imageio.imwrite(os.path.join(Random_combinations,str(txt+".png")),YCBCbCr2RGB((output)).astype(np.uint8))  
        
        # Saving imgs
        imageio.imwrite(os.path.join(rep_base_img,str("LR.png")),LR.astype(np.uint8))
        imageio.imwrite(os.path.join(rep_base_img,str("SR.png")),(YCBCbCr2RGB((MAIN*255.))).astype(np.uint8))
        imageio.imwrite(os.path.join(rep_base_img,str("TRUE.png")),HR.astype(np.uint8))
        imageio.imwrite(os.path.join(rep_base_img,str("Bic.png")),(YCBCbCr2RGB((interpolation_crop))).astype(np.uint8))
    
    
