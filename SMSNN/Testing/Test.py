from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorboard import *
from various_functions.tensor_tf_data_fonctions import *
from various_functions.rapports_pdf_fonctions import *
from various_functions.numpy_data_fonctions import *
from Networks.model_management import *
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

from Testing.PDF_reports import *
import numpy as np
import imageio

import itertools
import os  
import pandas as pd   
import tensorflow as tf

import tensorflow_probability as tfp

from scipy.ndimage.filters import gaussian_filter #blur

def Energy(model,main_network:str,root:str,save_rep:str,taille:int,nombre_class:int, border:int,nombre_patch_test:int,BN_init:bool, BN_fin:bool, DOG_init:bool,  DOG_fin:bool,sigma_noise_blur:float):
    """
    Compute different statistics for the MAIN MODEL 'model' on 'nombre_patch_test' test patches.
    
    * Lines FFT : means of the means of the FFT coefficients (L2 Norms of the spectre) ; (+) is what is added ; (-) is what is withdrawn
    ** Lines ENERGIE : means of the L2 Norms per patch ; (+) is what is added ; (-) is what is withdrawn
    
    --
    * model (keras API model) : MAIN model to test  already loaded
    * main_network (str) : nature of the MAIN NN ('SR', 'DENOISING' or 'BLURRING')
    * save_rep (str) : path of the energy folder where to save results
    * root(str) :  path where the model is saved
    
    * filtres_sr, taille, nombre_class, border, nombre_patch_test (int) : MAIN NN parameters and number of patches for the test to load
    * BN_init,BN_fin,DOG_init, DOG_fin (bools) : if True, adds BN layers or DoG layers
    * sigma_noise_blur (int) : std deviation for building   a.  noisy data   b. blurry data (if main_network is 'BLURRING' or 'DENOISING')
    
    Returns
    None (but save results i save_rep folder)
    """
    entree_lr_ycbcr,bicubic,bicubic_y,true,noms_tf, true_tf_nocrop  = open_all_batches(input_lr=False,main_network=main_network,taille=taille,border=border, nombre_patch_test=nombre_patch_test, root=root )    
    entree_y = tf.reshape(entree_lr_ycbcr[:,:,:,0],(nombre_patch_test,entree_lr_ycbcr.shape[1],entree_lr_ycbcr.shape[2],1))
    entree_y = crop_center_tf(entree_y,taille,taille)
    bicubic_crop = crop_center_tf(bicubic_y,taille,taille)
    true = tf.reshape(true[:,:,:,0],(nombre_patch_test,true.shape[1],true.shape[2],1)) 
        
    largeur=nombre_class 
    
    if main_network=="SR":
        INPUT_tf = entree_lr_ycbcr
    elif main_network=="DENOISING":
        INPUT_tf = true_tf_nocrop.numpy().copy()
        INPUT_tf[:,:,:,0]+=np.random.normal(0,sigma_noise_blur,(INPUT_tf.shape[1],INPUT_tf.shape[2]))
        
    elif main_network=="BLURRING":
        INPUT_tf = true_tf_nocrop.numpy().copy()
        INPUT_y = gaussian_filter(true_tf_nocrop[:,:,:,0], sigma=sigma_noise_blur)
        INPUT_tf[:,:,:,0]=INPUT_y
   
    # -------------------------------------------------------------------#
    # Extraction des modèles entrainés  ------------------------------------------#
    # -------------------------------------------------------------------# 
    SRm_i_y,SRm_i_cb,SRm_i_cr,SRm_i_yDoG,SRm_i_cbDoG,SRm_i_crDoG,SRm_i_yDoGBn,SRm_i_cbDoGBn,SRm_i_crDoGBn,SRm_o_y,SRm_o_yDoG,SRm_o_yDoGBn,SRm_o_yFinal=extract_layers_MAIN(model,nombre_class,DOG_init,DOG_fin,BN_init,BN_fin)
    # I. Withrawn/Added energy/fft coeffs in features --------------------#
    print("ENERGIE (L1) & FFT COEFFICIENTS PER FEATURES")
    # -------------------------------------------------------------------#
    # Inititalisation Statistiques  ------------------------------------------#
    # -------------------------------------------------------------------# 
    if DOG_init:
        SRenP_i_yDoG ,SRenP_i_cbDoG ,SRenP_i_crDoG ,SRfftP_i_yDoG ,SRfftP_i_cbDoG ,SRfftP_i_crDoG = [[ 0 for i in range (largeur)] for j in range(6)]  
        SRenN_i_yDoG ,SRenN_i_cbDoG ,SRenN_i_crDoG ,SRfftN_i_yDoG ,SRfftN_i_cbDoG ,SRfftN_i_crDoG = [[ 0 for i in range (largeur)] for j in range(6)]  
    if BN_init :
        SRenP_i_yDoGBn ,SRenP_i_cbDoGBn ,SRenP_i_crDoGBn ,SRfftP_i_yDoGBn ,SRfftP_i_cbDoGBn ,SRfftP_i_crDoGBn = [[ 0 for i in range (largeur)] for j in range(6)]  
        SRenN_i_yDoGBn ,SRenN_i_cbDoGBn ,SRenN_i_crDoGBn ,SRfftN_i_yDoGBn ,SRfftN_i_cbDoGBn ,SRfftN_i_crDoGBn = [[ 0 for i in range (largeur)] for j in range(6)]  
        
    SRenP_o_y ,SRfftP_o_y = [[ 0 for i in range (largeur)] for j in range(2)]        
    SRenN_o_y ,SRfftN_o_y = [[ 0 for i in range (largeur)] for j in range(2)]     
    
    if DOG_fin:
        SRenP_o_yDoG ,SRenN_o_yDoG ,SRfftP_o_yDoG ,SRfftN_o_yDoG = [[ 0 for i in range (largeur)] for j in range(4)]  
    if BN_fin:
        SRenP_o_yDoGBn ,SRenN_o_yDoGBn ,SRfftP_o_yDoGBn ,SRfftN_o_yDoGBn = [[ 0 for i in range (largeur)] for j in range(4)]  
    
    SRenP_o_yFinal ,SRenN_o_yFinal ,SRfftP_o_yFinal ,SRfftN_o_yFinal = [[ 0 for i in range (largeur)] for j in range(4)]  

    # Average Spectres
    FFT_OUTPUT_I = [np.zeros((nombre_patch_test,taille,taille)) for i in range(largeur)]
    FFT_SORTIE_I = [np.zeros((nombre_patch_test,taille,taille)) for i in range(largeur)]
    
    # Energie & FFt à calculer sur les inputs & Output
    #SR
    moyennes_HR_energie, moyennes_LR_energie, moyennes_BIC_energie, moyennes_OUT_energie, module_fft_patch_HR, module_fft_patch_LR, module_fft_patch_BIC, module_fft_patch_OUTPUT =[0 for j in range(8)]

    # Csv Energie
    list_columns=["Branche_(-)"+str(i) for i in range(nombre_class)]
    list_add=["Branche_(+)"+str(i) for i in range(nombre_class)]
    for j in range(1):
        list_add+=["Style_Y_(+)"+str(j),"Style_Cb_(+)"+str(j),"Style_Cr_(+)"+str(j)]
    for i in range(len(list_columns)):
        list_columns=np.insert(list_columns,2*i,list_add[i])
        
    df_psnr = pd.DataFrame(columns=list_columns)
    df_psnr.insert(0,"Energie",["INPUT","INPUT_FFT","","Y_Dog_en","Y_Dog_fft","Y_DogBn_en","Y_DogBn_fft","","Cb_Dog_en","Cb_Dog_fft","Cb_DogBn_en","Cb_DogBn_fft","","Cr_Dog_en","Cr_Dog_fft","Cr_DogBn_en","Cr_DogBn_fft","","Output_conv","Output_DOG","Output_final","","Output_conv_fft","Output_DOG_fft","Output_final_fft","","BICUBIC","BICUBIC_FFT","","OUTPUT","OUTPUT_FFT","","HR","HR_FFT"])
    
    # -------------------------------------------------------------------#
    # CALCULS DES FEATURES  --------------------------------------------#
    # -------------------------------------------------------------------#
    
    print("...")
    pred, SRf_i_yDoG,SRf_i_cbDoG,SRf_i_crDoG,SRf_i_yDoGBn,SRf_i_cbDoGBn,SRf_i_crDoGBn,SRf_o_y,SRf_o_yDoG,SRf_o_yDoGBn,SRf_o_yFinal = compute_features_MAIN(SRm_i_y,SRm_i_cb,SRm_i_cr,SRm_i_yDoG,SRm_i_cbDoG,SRm_i_crDoG,SRm_i_yDoGBn,SRm_i_cbDoGBn,SRm_i_crDoGBn,SRm_o_y,SRm_o_yDoG,SRm_o_yDoGBn,SRm_o_yFinal,
                                                                                                                                                           DOG_init,DOG_fin,BN_init,BN_fin, model, INPUT_tf ) 
    prediction=pred[:,:,:,0]
    print("computed")
    # -------------------------------------------------------------------#
    # CALCULS DES ENERGIES/FFT (+) (-) -------------------------------------#
    # -------------------------------------------------------------------#       
    for c in range (nombre_class):
        print('branche '+str(c))

        if DOG_init: 
            SRenP_i_yDoG[c],SRenN_i_yDoG[c],SRfftP_i_yDoG[c],SRfftN_i_yDoG[c]=added_feature_PN(SRf_i_yDoG[c],nombre_patch_test,taille+2*border,SRenP_i_yDoG[c],SRenN_i_yDoG[c],SRfftP_i_yDoG[c],SRfftN_i_yDoG[c])
            SRenP_i_cbDoG[c],SRenN_i_cbDoG[c],SRfftP_i_cbDoG[c],SRfftN_i_cbDoG[c]=added_feature_PN(SRf_i_cbDoG[c],nombre_patch_test,taille+2*border,SRenP_i_cbDoG[c],SRenN_i_cbDoG[c],SRfftP_i_cbDoG[c],SRfftN_i_cbDoG[c])
            SRenP_i_crDoG[c],SRenN_i_crDoG[c],SRfftP_i_crDoG[c],SRfftN_i_crDoG[c]=added_feature_PN(SRf_i_crDoG[c],nombre_patch_test,taille+2*border,SRenP_i_crDoG[c],SRenN_i_crDoG[c],SRfftP_i_crDoG[c],SRfftN_i_crDoG[c])
            
        if BN_init:
            SRenP_i_yDoGBn[c],SRenN_i_yDoGBn[c],SRfftP_i_yDoGBn[c],SRfftN_i_yDoGBn[c]=added_feature_PN(SRf_i_yDoGBn[c],nombre_patch_test,taille+2*border,SRenP_i_yDoGBn[c],SRenN_i_yDoGBn[c],SRfftP_i_yDoGBn[c],SRfftN_i_yDoGBn[c])
            SRenP_i_cbDoGBn[c],SRenN_i_cbDoGBn[c],SRfftP_i_cbDoGBn[c],SRfftN_i_cbDoGBn[c]=added_feature_PN(SRf_i_cbDoGBn[c],nombre_patch_test,taille+2*border,SRenP_i_cbDoGBn[c],SRenN_i_cbDoGBn[c],SRfftP_i_cbDoGBn[c],SRfftN_i_cbDoGBn[c])
            SRenP_i_crDoGBn[c],SRenN_i_crDoGBn[c],SRfftP_i_crDoGBn[c],SRfftN_i_crDoGBn[c]=added_feature_PN(SRf_i_crDoGBn[c],nombre_patch_test,taille+2*border,SRenP_i_crDoGBn[c],SRenN_i_crDoGBn[c],SRfftP_i_crDoGBn[c],SRfftN_i_crDoGBn[c])
            
        if DOG_fin:
            SRenP_o_yDoG[c],SRenN_o_yDoG[c],SRfftP_o_yDoG[c],SRfftN_o_yDoG[c]=added_feature_PN(SRf_o_yDoG[c],nombre_patch_test,taille,SRenP_o_yDoG[c],SRenN_o_yDoG[c],SRfftP_o_yDoG[c],SRfftN_o_yDoG[c])
            
        if BN_fin:
            SRenP_o_yDoGBn[c],SRenN_o_yDoGBn[c],SRfftP_o_yDoGBn[c],SRfftN_o_yDoGBn[c]=added_feature_PN(SRf_o_yDoGBn[c],nombre_patch_test,taille,SRenP_o_yDoGBn[c],SRenN_o_yDoGBn[c],SRfftP_o_yDoGBn[c],SRfftN_o_yDoGBn[c]) 
        
        SRenP_o_yFinal[c],SRenN_o_yFinal[c],SRfftP_o_yFinal[c],SRfftN_o_yFinal[c]=added_feature_PN(SRf_o_yFinal[c],nombre_patch_test,taille,SRenP_o_yFinal[c],SRenN_o_yFinal[c],SRfftP_o_yFinal[c],SRfftN_o_yFinal[c])
        SRenP_o_y[c],SRenN_o_y[c],SRfftP_o_y[c],SRfftN_o_y[c]=added_feature_PN(SRf_o_y[c],nombre_patch_test,taille,SRenP_o_y[c],SRenN_o_y[c],SRfftP_o_y[c],SRfftN_o_y[c])

        
        FFT_OUTPUT_I[c] += tf.math.reduce_mean(tf.math.abs(tf.signal.fft2d(tf.cast(SRf_o_y[c],tf.complex64))) )
        FFT_SORTIE_I[c] += tf.math.reduce_mean(tf.math.abs(tf.signal.fft2d(tf.cast(SRf_o_yFinal[c],tf.complex64))) )
 
    # Calcul FFt & Cn / spectres moyens Pour HR,BIC,LR et OUTPUT     
    # HR
    FFT_HR = tf.math.abs(tf.signal.fft2d(tf.cast(true,tf.complex64))).numpy()
    module_fft_patch_HR +=(tf.math.reduce_mean(tf.math.abs(tf.signal.fft2d(tf.cast(true,tf.complex64))))).numpy()
    moyennes_HR_energie+=tf.norm((true),ord=1).numpy()
    # LR
    FFT_LR= tf.math.abs(tf.signal.fft2d(tf.cast(entree_y,tf.complex64))).numpy()
    module_fft_patch_LR +=(tf.math.reduce_mean(tf.math.abs(tf.signal.fft2d(tf.cast(entree_y,tf.complex64))))).numpy()  
    moyennes_LR_energie+=tf.norm((entree_y),ord=1).numpy()
    # BIC
    FFT_BIC=  tf.math.abs(tf.signal.fft2d(tf.cast(bicubic_crop,tf.complex64))).numpy()
    module_fft_patch_BIC +=(tf.math.reduce_mean(tf.math.abs(tf.signal.fft2d(tf.cast(bicubic_crop,tf.complex64))))).numpy()   
    moyennes_BIC_energie+=tf.norm((bicubic_crop),ord=1).numpy()
    # OUTPUT
    FFT_OUTPUT= tf.math.abs(tf.signal.fft2d(tf.cast(prediction,tf.complex64))).numpy()
    module_fft_patch_OUTPUT +=(tf.math.reduce_mean(tf.math.abs(tf.signal.fft2d(tf.cast(prediction,tf.complex64))))).numpy()
    moyennes_OUT_energie+=tf.norm((prediction),ord=1).numpy()
    print("...")
    # CONSTRUCTION CSV ---------------
    
    # LIGNE 1 : INPUT (Energie & FFT)
    
    FFT_LR=FFT_LR/nombre_patch_test
    module_fft_patch_LR=module_fft_patch_LR/nombre_patch_test
    moyennes_LR_energie=moyennes_LR_energie/nombre_patch_test
    df_psnr.iloc[0,1] = moyennes_LR_energie
    df_psnr.iloc[1,1] = module_fft_patch_LR

    largeur2=nombre_class
        
    for c in range (largeur2):
        # CSV energie norme L1
        #(+)
        if DOG_init:
            df_psnr.iloc[3,2*c+1] = SRenP_i_yDoG[c]/nombre_patch_test
            df_psnr.iloc[4,2*c+1] = SRfftP_i_yDoG[c]/nombre_patch_test
        if BN_init :
            df_psnr.iloc[5,2*c+1] = SRenP_i_yDoGBn[c]/nombre_patch_test
            df_psnr.iloc[6,2*c+1] = SRfftP_i_yDoGBn[c]/nombre_patch_test
        if DOG_init:
            df_psnr.iloc[8,2*c+1] = SRenP_i_cbDoG[c]/nombre_patch_test
            df_psnr.iloc[9,2*c+1] = SRfftP_i_cbDoG[c]/nombre_patch_test
        if BN_init :
            df_psnr.iloc[10,2*c+1] = SRenP_i_cbDoGBn[c]/nombre_patch_test
            df_psnr.iloc[11,2*c+1] = SRfftP_i_cbDoGBn[c]/nombre_patch_test
        if DOG_init:
            df_psnr.iloc[13,2*c+1] = SRenP_i_crDoG[c]/nombre_patch_test
            df_psnr.iloc[14,2*c+1] = SRfftP_i_crDoG[c]/nombre_patch_test
        if BN_init :
            df_psnr.iloc[15,2*c+1] = SRenP_i_crDoGBn[c]/nombre_patch_test
            df_psnr.iloc[16,2*c+1] = SRfftP_i_crDoGBn[c]/nombre_patch_test
        
        
        df_psnr.iloc[18,2*c+1] = SRenP_o_y[c]/nombre_patch_test        #sortie convolutions
        if DOG_fin:
            df_psnr.iloc[19,2*c+1] = SRenP_o_yDoG[c]/nombre_patch_test       #sortie dog
        df_psnr.iloc[20,2*c+1] = SRenP_o_yFinal[c]/nombre_patch_test               #sortie finale
        
        
        df_psnr.iloc[22,2*c+1] =  SRfftP_o_y[c]/nombre_patch_test   #sortie convolutions
        if DOG_fin:
            df_psnr.iloc[23,2*c+1] =  SRfftP_o_yDoG[c]/nombre_patch_test  #sortie dog   
        df_psnr.iloc[24,2*c+1] =  SRfftP_o_yFinal[c]/nombre_patch_test          #sortie finale
        
        #(-)
        if DOG_init:
            df_psnr.iloc[3,2*c+2] = SRenN_i_yDoG[c]/nombre_patch_test
            df_psnr.iloc[4,2*c+2] = SRfftN_i_yDoG[c]/nombre_patch_test
        if BN_init :
            df_psnr.iloc[5,2*c+2] = SRenN_i_yDoGBn[c]/nombre_patch_test
            df_psnr.iloc[6,2*c+2] = SRfftN_i_yDoGBn[c]/nombre_patch_test
        if DOG_init:
            df_psnr.iloc[8,2*c+2] = SRenN_i_cbDoG[c]/nombre_patch_test
            df_psnr.iloc[9,2*c+2] = SRfftN_i_cbDoG[c]/nombre_patch_test
        if BN_init :
            df_psnr.iloc[10,2*c+2] = SRenN_i_cbDoGBn[c]/nombre_patch_test
            df_psnr.iloc[11,2*c+2] = SRfftN_i_cbDoGBn[c]/nombre_patch_test
        if DOG_init:
            df_psnr.iloc[13,2*c+2] = SRenN_i_crDoG[c]/nombre_patch_test
            df_psnr.iloc[14,2*c+2] = SRfftN_i_crDoG[c]/nombre_patch_test
        if BN_init :
            df_psnr.iloc[15,2*c+2] = SRenN_i_crDoGBn[c]/nombre_patch_test
            df_psnr.iloc[16,2*c+2] = SRfftN_i_crDoGBn[c]/nombre_patch_test
        
        
        df_psnr.iloc[18,2*c+2] = SRenN_o_y[c]/nombre_patch_test        #sortie convolutions
        if DOG_fin:
            df_psnr.iloc[19,2*c+2] = SRenN_o_yDoG[c]/nombre_patch_test       #sortie dog
        df_psnr.iloc[20,2*c+2] = SRenN_o_yFinal[c]/nombre_patch_test               #sortie finale
        
        
        df_psnr.iloc[22,2*c+2] =  SRfftN_o_y[c]/nombre_patch_test   #sortie convolutions
        if DOG_fin:
            df_psnr.iloc[23,2*c+2] =  SRfftN_o_yDoG[c]/nombre_patch_test  #sortie dog   
        df_psnr.iloc[24,2*c+2] =  SRfftN_o_yFinal[c]/nombre_patch_test          #sortie finale
            
    FFT_BIC=FFT_BIC/nombre_patch_test
    module_fft_patch_BIC=module_fft_patch_BIC/nombre_patch_test
    moyennes_BIC_energie=moyennes_BIC_energie/nombre_patch_test
    df_psnr.iloc[26,1] = moyennes_BIC_energie
    df_psnr.iloc[27,1] = module_fft_patch_BIC
    
    # LIGNE  : OUTPUT (Energie & FFT)    
    FFT_OUTPUT=FFT_OUTPUT/nombre_patch_test
    module_fft_patch_LR=module_fft_patch_OUTPUT/nombre_patch_test
    moyennes_OUT_energie=moyennes_OUT_energie/nombre_patch_test
    df_psnr.iloc[29,1] = moyennes_OUT_energie
    df_psnr.iloc[30,1] = module_fft_patch_OUTPUT

    # LIGNE  : HR (Energie & FFT)    
    FFT_HR=FFT_HR/nombre_patch_test
    module_fft_patch_HR=module_fft_patch_HR/nombre_patch_test
    moyennes_HR_energie=moyennes_HR_energie/nombre_patch_test
    df_psnr.iloc[32,1] = moyennes_HR_energie
    df_psnr.iloc[33,1] = module_fft_patch_HR       
    
    #FFT Outputs (+)&(-)
    for c in range(nombre_class-1):
        FFT_OUTPUT_I[c]=FFT_OUTPUT_I[c]/nombre_patch_test
    np.save(os.path.join(save_rep,"FFT_moyen_OUTPUT_"+str(c)+".npy"),FFT_OUTPUT_I[c])
      
    df_psnr.to_csv(os.path.join(save_rep,"energie_"+".csv"), sep='\t') 
    np.save(os.path.join(save_rep,"FFT_moyen_HR.npy"),FFT_HR)
    np.save(os.path.join(save_rep,"FFT_moyen_BIC.npy"),FFT_BIC)
    np.save(os.path.join(save_rep,"FFT_moyen_LR.npy"),FFT_LR)
    np.save(os.path.join(save_rep,"FFT_moyen_OUTPUT.npy"),FFT_OUTPUT)
    

    # II . Paerson Correlation coefficients ----------------------------#
    print(" Paerson Correlation coefficients")
    # For each intermediate feature, compute all the 2-2 pearson correlation coefficients as Pearson([1,W,H,C],[1,W,H,C], batch size being reduced to its mean)
    l=nombre_class
        
    one_combinations_cor = [np.reshape(np.array(i), (l)) for i in itertools.product([0, 1], repeat = l) if sum(np.reshape(np.array(i),(l)))==2 ] 
    nombre_combinaison_cor = len(one_combinations_cor) 
    
    
    list_features_SR = [SRf_o_yFinal , SRf_o_y]
    noms_features=["Sortie_branche", "Sortie_convolution_preDoG_BN"]
    
    if DOG_fin:
        list_features_SR.append(SRf_o_yDoG)
        noms_features.append("Sortie_convolution_Dog")

    if BN_fin:
        list_features_SR.append(SRf_o_yDoGBn)
        noms_features.append("Sortie_convolution_DogBn")
        
    for features in range(len(list_features_SR)):
        vec = list_features_SR[features]
        vecteurs = tf.concat(vec,axis=-1) 
        covariance_matrix = np.ones((l,l))
        for combinaison in range (nombre_combinaison_cor):
            index_combinaisons = np.where(one_combinations_cor[combinaison]==1)[0]
            x=index_combinaisons[0]
            y=index_combinaisons[1]
            A = tf.math.reduce_mean(tf.reshape(vecteurs[:,:,:,x],(nombre_patch_test,taille,taille,1)),axis=0)
            B = tf.math.reduce_mean(tf.reshape(vecteurs[:,:,:,y],(nombre_patch_test,taille,taille,1)),axis=0)
            Paerson = tfp.stats.correlation(A,B,event_axis=None,sample_axis=None)
            covariance_matrix[x,y] =  covariance_matrix[y,x] = Paerson
            
            # determinant
            determinant = tf.linalg.det(covariance_matrix)
            #np.save(os.path.join(save_rep,"covariance_matrix.np"),covariance_matrix) 
            df_cov = pd.DataFrame(covariance_matrix)
            df_cov.to_csv(os.path.join(save_rep,"covariance_matrix_"+str(noms_features[features])+".csv"),index=False)
    
    
    
    # III. PSNR combinations 

    one_combinations = [np.reshape(np.array(i), (l)) for i in itertools.product([0, 1], repeat = l) if sum(np.reshape(np.array(i), (l)))<12 ]#if sum(np.reshape(np.array(i), (l)))==1 or sum(np.reshape(np.array(i), (l)))==0] 
    nombre_vecteurs = [one_combinations[i].sum() for i in range(len(one_combinations))]    
    ordre_croissant = np.argsort(nombre_vecteurs)
    one_combinations = [one_combinations[ordre_croissant[i]] for i in range(len(ordre_croissant))]
    nombre_vecteurs_croissant = np.sort(nombre_vecteurs)
    nombre_combinaison = len(one_combinations)
    
    df_2 = pd.DataFrame(columns=[(str(one_combinations[i]))+"_"+str(nombre_vecteurs_croissant[i]) for i in range(nombre_combinaison)])
    df_2.insert(0,"séries_patch_test",["serie_"+str(nombre_patch_test)])

    vec = [tf.reshape(SRf_o_yFinal[i],(nombre_patch_test,taille,taille,1)) for i in range(len(SRf_o_yFinal))]
    vecteurs = tf.concat(vec,axis=-1)

    
    for combinaison in range (nombre_combinaison):
        nbre_vecteurs = sum(one_combinations[combinaison])
        output=bicubic_crop
        for chan in range(l):
            output += tf.reshape(vecteurs[:,:,:,chan],(nombre_patch_test,taille,taille,1)) *one_combinations[combinaison][chan] 
        output=tf.clip_by_value(output,0,1) 
        psnr = tf.image.psnr(output,true,1)
        psnr_mean = tf.math.reduce_mean(psnr)
        df_2.iloc[0,combinaison+1] = psnr_mean.numpy()
        
    df_2.to_csv(os.path.join(save_rep,"psnr_"+".csv"), sep='\t')   
  
def process_test_controle_MAIN(model,main_network:str, root:str,save_rep:str, save_rep_energie:str,taille:int,nombre_class:int, border:int,BN_init:bool, BN_fin:bool, DOG_fin:bool, DOG_init:bool,nombre_patch_test:int,sigma_noise_blur:float):
    """
    Evaluation of testing patchs (qualitatively) for the MAIN NN 'model'. Generates visual reports. 
    """
    paquet = nombre_patch_test//30
    
    # Generator
    gen=Generator_test_patch(main_network=main_network, size_output=taille,  border=border, root=root)
    multiple = int(nombre_patch_test/25)

    SRm_i_y,SRm_i_cb,SRm_i_cr,SRm_i_yDoG,SRm_i_cbDoG,SRm_i_crDoG,SRm_i_yDoGBn,SRm_i_cbDoGBn,SRm_i_crDoGBn,SRm_o_y,SRm_o_yDoG,SRm_o_yDoGBn,SRm_o_yFinal=extract_layers_MAIN(model,nombre_class,DOG_init,DOG_fin,BN_init,BN_fin)
    nombre_boucle = nombre_patch_test // paquet
    image_debut_boucle = 0  
    

    PSNR_outputs_1 = np.zeros((nombre_patch_test,nombre_class)) # Psnr table bic + Sr_i
    PSNR_outputs_2 = np.zeros((nombre_patch_test,nombre_class))    # Psnr table bic - Sr_i  

    for ind in range(nombre_boucle):
        for i in range(image_debut_boucle,(ind+1)*paquet): 
            if (i%multiple==1):
                # Patch Information ; loading INPUT (Bic,Blurry or Noisy), Ground truth, Name
                nom =gen[0].filenames[i].split("/")[-1].replace(".png","")
                entree_ycbcr = gen[0][i].reshape(taille+2*border,taille+2*border,3)
                INPUT_tf=gen[0][i]
                TRUE_tf= gen[1][i]
                if main_network=="SR":
                    INPUT_tf = INPUT_tf
                elif main_network=="DENOISING":
                    INPUT_tf = INPUT_tf.copy()
                    INPUT_tf[:,:,:,0]+=np.random.normal(0,sigma_noise_blur,(INPUT_tf.shape[1],INPUT_tf.shape[2]))        
                elif main_network=="BLURRING":
                    INPUT_tf = INPUT_tf.copy()
                    INPUT_y = gaussian_filter(INPUT_tf[:,:,:,0], sigma=sigma_noise_blur)
                    INPUT_tf[:,:,:,0]=INPUT_y
                
                entree_y , entree_cb , entree_cr = entree_ycbcr[:,:,0],entree_ycbcr[:,:,1],entree_ycbcr[:,:,2]        
                entree_y = entree_y[border:entree_y.shape[0]-border,border:entree_y.shape[1]-border]
                entree_ycbcr=entree_ycbcr[border:entree_ycbcr.shape[0]-border,border:entree_ycbcr.shape[1]-border]
                
                
                true_y=TRUE_tf[0,:,:,0].reshape(TRUE_tf.shape[1],TRUE_tf.shape[2])
                true_ycbcr=TRUE_tf[0,:,:,:].reshape(TRUE_tf.shape[1],TRUE_tf.shape[2],3)
                
                prediction_ycbcr, SRf_i_yDoG,SRf_i_cbDoG,SRf_i_crDoG,SRf_i_yDoGBn,SRf_i_cbDoGBn,SRf_i_crDoGBn,SRf_o_y,SRf_o_yDoG,SRf_o_yDoGBn,SRf_o_yFinal = compute_features_MAIN(SRm_i_y,SRm_i_cb,SRm_i_cr,SRm_i_yDoG,SRm_i_cbDoG,SRm_i_crDoG,SRm_i_yDoGBn,SRm_i_cbDoGBn,SRm_i_crDoGBn,SRm_o_y,SRm_o_yDoG,SRm_o_yDoGBn,SRm_o_yFinal,
                                                                                                                                                                                                          DOG_init,DOG_fin,BN_init,BN_fin, model, INPUT_tf ) 
                # PSNR
                mse_reseau,mse_bic = np.mean((true_ycbcr-prediction_ycbcr)**2) , np.mean((true_y-entree_y)**2)
                
                # PDF Reports
                tbadded=TRUE_tf[0,:,:,0]-INPUT_tf[0,border:INPUT_tf.shape[0]-border-1,border:INPUT_tf.shape[1]-border,0]
                added=prediction_ycbcr[0,:,:,0]-INPUT_tf[0,border:INPUT_tf.shape[0]-border-1,border:INPUT_tf.shape[1]-border,0]
            
                list_tensor_input= [INPUT_tf[0,:,:,0],INPUT_tf[0,:,:,:], TRUE_tf[0,:,:,0],TRUE_tf[0,:,:,:],prediction_ycbcr[0,:,:,0],prediction_ycbcr[0,:,:,:]]
                list_tensor_intermediate_features=[tbadded,added]
                        
                name_tensor_input=["Input_y","Input","True_y","True","Output_y","Output"]
                name_tensor_intermediate_features=["To be added","Added"]
                
                # PDF Report
                Report_features_MAIN(x=0,y=0,dx=taille,SRf_o_y=SRf_o_y,SRf_o_yDoG=SRf_o_yDoG,SRf_o_yDoGBn=SRf_o_yDoGBn,SRf_o_yFinal=SRf_o_yFinal,input_=INPUT_tf,prediction=prediction_ycbcr,BN_init=BN_init,BN_fin=BN_fin,DOG_init=DOG_init, DOG_fin=DOG_fin,taille = INPUT_tf.shape[1] ,nombre_class = nombre_class , nom = nom ,save_rep = save_rep ,root_folder = root)
               
                Patch_report_features(list_tensor_input, list_tensor_intermediate_features, name_tensor_input, name_tensor_intermediate_features,mse_reseau,mse_bic,save_rep=save_rep,taille=taille,nombre_class=nombre_class,root=root,border=border,nom=nom,ecart=22,numero_patch=i)
                
                Patch_report_MAIN(prediction_ycbcr[0,:,:,:],INPUT_tf[0,border:INPUT_tf.shape[0]-border-1,border:INPUT_tf.shape[1]-border,0],SRf_o_yFinal,save_rep=save_rep,nombre_class=nombre_class,root_folder=root,nom=nom)
                
            
        image_debut_boucle  = (ind+1)*paquet+1 
        Patch_report_features
    
def statistique_branch_MAIN(model,nombre_class,DOG_init,DOG_fin,BN_init,BN_fin,root_rep):
    '''
    Load Batch_Normalization weights from specific layers for the MAIN NN.
    '''
    with open(os.path.join(root_rep,"_PoidsConv.txt"), "w") as text_file:
        text_file.write(" -------  SR NETWORK -------"+"\n")
        for i in range(nombre_class): 
            # SR NETWORK
            if BN_init:
                text_file.write("1. Gains BatchNormS Init (y,cb,cr) -------"+"\n")
                
                weights_init_y = model.get_layer("SRm_i_yDoGBn"+str(i)).get_weights()
                weights_init_cb = model.get_layer("SRm_i_cbDoGBn"+str(i)).get_weights()
                weights_init_cr = model.get_layer("SRm_i_crDoGBn"+str(i)).get_weights()
                text_file.write("Gamma (gain) --- BN_y:"+ str(weights_init_y[0])[1:4]+" cb:"+ str(weights_init_cb[0])[1:4]+" cr:"+ str(weights_init_cr[0])[1:4]+"\n")
                text_file.write("Beta (gain) --- BN_y:"+ str(weights_init_y[1])[1:4]+" cb:"+ str(weights_init_cb[1])[1:4]+" cr:"+ str(weights_init_cr[1])[1:4]+"\n")
                text_file.write("Mean --- BN_y:"+ str(weights_init_y[2])[1:4]+" cb:"+ str(weights_init_cb[2])[1:4]+" cr:"+ str(weights_init_cr[2])[1:4]+"\n")
                text_file.write("Variance --- BN_y:"+ str(weights_init_y[3])[1:4]+" cb:"+ str(weights_init_cb[3])[1:4]+" cr:"+ str(weights_init_cr[3])[1:4]+"\n")

            text_file.write("2. Gains BatchNormS Intermediaires (MOYENNE) -------"+"\n")
            weights_bn_conv1 = model.get_layer("R_Bn_0_channel"+str(i)).get_weights()
            weights_bn_conv2 = model.get_layer("R_Bn_1_channel"+str(i)).get_weights()
            weights_bn_conv3 = model.get_layer("R_Bn_2_channel"+str(i)).get_weights()
            #weights_bn_conv4 = model.get_layer("R_Bn_3_channel"+str(i)).get_weights()
            text_file.write("Gamma (gain) --- BN_convo1:"+ str(np.mean(weights_bn_conv1[0]))[1:4]+" BN_convo2:"+ str(np.mean(weights_bn_conv2[0]))[1:4]+" BN_convo3:"+ str(np.mean(weights_bn_conv3[0]))[1:4]+"\n") #+"\n"+" BN_convo4:"+ str(np.mean(weights_bn_conv4[0]))[1:4]+
            text_file.write("Beta (gain) --- BN_convo1:"+ str(np.mean(weights_bn_conv1[1]))[1:4]+" BN_convo2:"+ str(np.mean(weights_bn_conv2[1]))[1:4]+" BN_convo3:"+ str(np.mean(weights_bn_conv3[1]))[1:4]+"\n")
            text_file.write("Mean --- BN_convo1:"+ str(np.mean(weights_bn_conv1[2]))[1:4]+" BN_convo2:"+ str(np.mean(weights_bn_conv2[2]))[1:4]+" BN_convo3:"+ str(np.mean(weights_bn_conv3[2]))[1:4]+"\n")
            text_file.write("Variance --- BN_convo1:"+ str(np.mean(weights_bn_conv1[3]))[1:4]+" BN_convo2:"+ str(np.mean(weights_bn_conv2[3]))[1:4]+" BN_convo3:"+ str(np.mean(weights_bn_conv3[3]))[1:4]+"\n")
                
            if BN_fin:
                text_file.write("3. Gains BatchNorm Fin (y) -------"+"\n")
                weights_fin = model.get_layer("SRm_o_yDoGBn"+str(i)).get_weights()
                text_file.write("Gamma (gain) --- BN_y:"+ str(weights_fin[0])[1:4]+"\n")
                text_file.write("Beta (gain) --- BN_y:"+ str(weights_fin[1])[1:4]+"\n")
                text_file.write("Mean --- BN_y:"+ str(weights_fin[2])[1:4]+"\n")
                text_file.write("Variance --- BN_y:"+ str(weights_fin[3])[1:4]+"\n")
            text_file.write("-------------------------"+"\n")   

def Save_gray_list(list_tensor_gray:list,list_name_gray:list,rep_save:str):
    '''
    Save list of (W,H,1) tensor features into a specific folder /rep_save
    '''
    compteur=0
    for tensor in (list_tensor_gray):
        imageio.imwrite(os.path.join(rep_save,str(list_name_gray)+"_"+str(compteur)+".png"),rescale_255(tensor[0,:,:,0]*255.).astype(np.uint8))
        compteur+=1

def Save_tensor(list_tensor:list,list_name:list,rep_save:str):
    '''
    Save list of (W,H,3) tensors into a specific folder /rep_save
    '''
    compteur=0
    for tensor in (list_tensor):
        try:
            imageio.imwrite(os.path.join(rep_save,"_"+str(list_name[compteur]+".png")),(YCBCbCr2RGB(tensor[0,:,:,:]*255.)).astype(np.uint8))
        except IndexError: #gray
            imageio.imwrite(os.path.join(rep_save,"_"+str(list_name[compteur]+".png")),(rescale_255(tensor[0,:,:,0]*255.)).astype(np.uint8))
            
        compteur+=1
        
        
        
def Image_benchmark_MAIN(model_identite,main_network:str,ponderation_features:list, nombre_class:int,border:int, sigma_noise_blur:float,filtres:int, nombre_kernel:int,save_rep:str,root_folder:str, BN_init:bool,BN_fin:bool,DOG_init:bool, DOG_fin:bool,  FG_sigma_init:int,FG_sigma_puissance:int,benchmark_folder:str, w_h,w_v):
    """
    Test on Benchmark Images for MAIN Network 
    """ 
    rep_images = os.path.join(benchmark_folder,"BENCHMARK_VISUEL/Rapports")
    liste_images = [x for x in os.listdir(rep_images) if x.endswith(".png")]
    x_m,y_m,x_max,y_max=300,300,900,900
    mse_reseau_moyen=0
    mse_bicubic_moyen=0
    x_zoom=257
    y_zoom=293
    dx_zoom=450
    R=4

    tf.keras.backend.clear_session()
    new_model = MAIN_network_none(nombre_class=nombre_class,filtres=filtres, ponderation_features=ponderation_features,kernel=nombre_kernel,w_h=w_h,w_v=w_v, BN_init=BN_init, BN_fin=BN_fin, DOG_init=DOG_init, DOG_fin=DOG_fin)  
    new_model =copy_parameters(model=new_model,model_patch=model_identite)


    nombre_image=len(liste_images)
    for img in liste_images :
        print("Infering image: "+str(img))

        # Data Preparation & Input
        img_array_LR = Ouverture_img(os.path.join(os.path.join(os.path.join(os.path.join(rep_images,"LR"),img.replace(".png","_LR.png")))),1)[0]#.astype(np.float64)
        img_array_LR=img_array_LR[int(x_m/R):int(x_max/R),int(y_m/R):int(y_max/R),:]/256.
        img_array_HR=Ouverture_img(os.path.join(os.path.join(os.path.join(rep_images,img))),1)[0]#.astype(np.float64)
        img_array_HR=img_array_HR[x_m:x_max,y_m:y_max,:]/256.
        
        if main_network=="SR":
            INPUT = upsampling(img_array_LR,R) 
        elif main_network=="DENOISING":
            INPUT = img_array_HR.copy()
            INPUT[:,:,0]+=np.random.normal(0,sigma_noise_blur,(img_array_HR.shape[0],img_array_HR.shape[1]))
        elif main_network=="BLURRING":
            INPUT = img_array_HR.copy()
            INPUT_y = gaussian_filter(INPUT[:,:,0], sigma=sigma_noise_blur)
            INPUT[:,:,0]=INPUT_y
        INPUT=np.expand_dims(INPUT,axis=0)
        TRUE=np.expand_dims(img_array_HR,axis=0)
        
        # Features Main NN
        SRm_i_y,SRm_i_cb,SRm_i_cr,SRm_i_yDoG,SRm_i_cbDoG,SRm_i_crDoG,SRm_i_yDoGBn,SRm_i_cbDoGBn,SRm_i_crDoGBn,SRm_o_y,SRm_o_yDoG,SRm_o_yDoGBn,SRm_o_yFinal = extract_layers_MAIN(new_model,nombre_class,DOG_init,DOG_fin,BN_init,BN_fin)
        
        prediction_ycbcr, SRf_i_yDoG,SRf_i_cbDoG,SRf_i_crDoG,SRf_i_yDoGBn,SRf_i_cbDoGBn,SRf_i_crDoGBn,SRf_o_y,SRf_o_yDoG,SRf_o_yDoGBn,SRf_o_yFinal = compute_features_MAIN(SRm_i_y,SRm_i_cb,SRm_i_cr,SRm_i_yDoG,SRm_i_cbDoG,SRm_i_crDoG,SRm_i_yDoGBn,SRm_i_cbDoGBn,SRm_i_crDoGBn,SRm_o_y,SRm_o_yDoG,SRm_o_yDoGBn,SRm_o_yFinal,
                                                                                                                                                                             DOG_init,DOG_fin,BN_init,BN_fin,new_model, INPUT )
        
        rep_img_save = os.path.join(os.path.join(os.path.join(save_rep,"rapports_benchmark"),str(img)))
        ensure_dir(rep_img_save)
        
        #  Pdf reports ---
        tbadded=TRUE[0,border:INPUT.shape[0]-border-1,border:INPUT.shape[1]-border,0]-INPUT[0,border:INPUT.shape[0]-border-1,border:INPUT.shape[1]-border,0]
        added=prediction_ycbcr[0,:,:,0]-INPUT[0,border:INPUT.shape[0]-border-1,border:INPUT.shape[1]-border,0]
                
        list_tensor_input= [INPUT[0,:,:,0],INPUT[0,:,:,:], TRUE[0,:,:,0],TRUE[0,:,:,:],prediction_ycbcr[0,:,:,0],prediction_ycbcr[0,:,:,:]]
        list_tensor_intermediate_features=[tbadded,added]
                
        name_tensor_input=["Input_y","Input","True_y","True","Output_y","Output"]
        name_tensor_intermediate_features=["To be added","Added"]
        
        Report_features_MAIN(x=x_zoom,y=y_zoom,dx=dx_zoom,SRf_o_y=SRf_o_y,SRf_o_yDoG=SRf_o_yDoG,SRf_o_yDoGBn=SRf_o_yDoGBn,SRf_o_yFinal=SRf_o_yFinal,input_=INPUT,prediction=prediction_ycbcr,BN_init=BN_init,BN_fin=BN_fin,DOG_init=DOG_init, DOG_fin=DOG_fin,taille = INPUT.shape[1]//4 ,nombre_class = nombre_class , nom = img ,save_rep = rep_img_save ,root_folder = root_folder)
        
        Benchmark_report(x=x_zoom,y=y_zoom,dx=dx_zoom,list_tensor_input=list_tensor_input, list_tensor_intermediate_features=list_tensor_intermediate_features, name_tensor_input=name_tensor_input,name_tensor_intermediate_features=name_tensor_intermediate_features,type_branch=main_network,taille=INPUT.shape[1],nombre_class=nombre_class,root=root_folder,border=border,save_rep=rep_img_save,nom=img)
        
        # Save Png ----
        Save_tensor(list_tensor=[INPUT,TRUE,prediction_ycbcr],list_name=["input","true","prediction"],rep_save=rep_img_save)
        Save_gray_list(list_tensor_gray=SRf_o_yFinal,list_name_gray="Output_branch_",rep_save=rep_img_save)
        


#  BRANCH
def statistique_branch_Sty(model,nombre_class,DOG_init,DOG_fin,BN_init,BN_fin,root_rep):
    '''
    Load Batch_Normalization weights from specific layers for STy branch.
    '''

    with open(os.path.join(root_rep,"_PoidsConv.txt"), "w") as text_file:
        text_file.write(" -------  ST NETWORK -------"+"\n")
        if BN_init:
            text_file.write("1. Gains BatchNormS Init (y,cb,cr) -------"+"\n")
            weights_init_y = model.get_layer("STm_i_yDoGBn").get_weights()
            weights_init_cb = model.get_layer("STm_i_cbDoGBn").get_weights()
            weights_init_cr = model.get_layer("STm_i_crDoGBn").get_weights()
            text_file.write("Gamma (gain) --- BN_y:"+ str(weights_init_y[0])[1:4]+" cb:"+ str(weights_init_cb[0])[1:4]+" cr:"+ str(weights_init_cr[0])[1:4]+"\n")
            text_file.write("Beta (gain) --- BN_y:"+ str(weights_init_y[1])[1:4]+" cb:"+ str(weights_init_cb[1])[1:4]+" cr:"+ str(weights_init_cr[1])[1:4]+"\n")
            text_file.write("Mean --- BN_y:"+ str(weights_init_y[2])[1:4]+" cb:"+ str(weights_init_cb[2])[1:4]+" cr:"+ str(weights_init_cr[2])[1:4]+"\n")
            text_file.write("Variance --- BN_y:"+ str(weights_init_y[3])[1:4]+" cb:"+ str(weights_init_cb[3])[1:4]+" cr:"+ str(weights_init_cr[3])[1:4]+"\n")
                
        text_file.write("2. Gains BatchNormS Intermediaires (MOYENNE) -------"+"\n")
        weights_bn_conv1 = model.get_layer("T_Bn0_0").get_weights()
        weights_bn_conv2 = model.get_layer("T_Bn0_1").get_weights()
        weights_bn_conv3 = model.get_layer("T_Bn0_2").get_weights()
        weights_bn_conv4 = model.get_layer("T_Bn0_3").get_weights()
        text_file.write("Gamma (gain) --- BN_convo1:"+ str(np.mean(weights_bn_conv1[0]))[1:4]+" BN_convo2:"+ str(np.mean(weights_bn_conv2[0]))[1:4]+" BN_convo3:"+ str(np.mean(weights_bn_conv3[0]))[1:4]+"\n"+" BN_convo4:"+ str(np.mean(weights_bn_conv4[0]))[1:4]+"\n")
        text_file.write("Beta (gain) --- BN_convo1:"+ str(np.mean(weights_bn_conv1[1]))[1:4]+" BN_convo2:"+ str(np.mean(weights_bn_conv2[1]))[1:4]+" BN_convo3:"+ str(np.mean(weights_bn_conv3[1]))[1:4]+"\n"+" BN_convo4:"+ str(np.mean(weights_bn_conv4[1]))[1:4]+"\n")
        text_file.write("Mean --- BN_convo1:"+ str(np.mean(weights_bn_conv1[2]))[1:4]+" BN_convo2:"+ str(np.mean(weights_bn_conv2[2]))[1:4]+" BN_convo3:"+ str(np.mean(weights_bn_conv3[2]))[1:4]+"\n"+" BN_convo4:"+ str(np.mean(weights_bn_conv4[2]))[1:4]+"\n")
        text_file.write("Variance --- BN_convo1:"+ str(np.mean(weights_bn_conv1[3]))[1:4]+" BN_convo2:"+ str(np.mean(weights_bn_conv2[3]))[1:4]+" BN_convo3:"+ str(np.mean(weights_bn_conv3[3]))[1:4]+"\n"+" BN_convo4:"+ str(np.mean(weights_bn_conv4[3]))[1:4]+"\n")
                
        text_file.write("3. MODULE FB (y) -------"+"\n")
        weights_fin = model.get_layer("F_BN_outputi_y_style_fb").get_weights()
        text_file.write("Gamma (gain) --- BN_y:"+ str(weights_fin[0])[1:4]+"\n")
        text_file.write("Beta (gain) --- BN_y:"+ str(weights_fin[1])[1:4]+"\n")
        #text_file.write("Mean --- BN_y:"+ str(weights_fin[2])[1:4]+"\n")
        #text_file.write("Variance --- BN_y:"+ str(weights_fin[3])[1:4]+"\n")

    print(root_rep," statistiques saved")
    
def statistique_branch_Stcol(model,nombre_class,DOG_init,DOG_fin,BN_init,BN_fin,root_rep):
    '''
    Load Batch_Normalization weights from specific layers for STy branch.
    '''
    with open(os.path.join(root_rep,"_PoidsConv.txt"), "w") as text_file:
        text_file.write(" -------  COL NETWORK -------"+"\n")
        if BN_init:
            text_file.write("1. Gains BatchNormS Init (y,cb,cr) -------"+"\n")
            weights_init_y = model.get_layer("COLm_i_yDoGBn").get_weights()
            weights_init_cb = model.get_layer("COLm_i_cbDoGBn").get_weights()
            weights_init_cr = model.get_layer("COLm_i_crDoGBn").get_weights()
            text_file.write("Gamma (gain) --- BN_y:"+ str(weights_init_y[0])[1:4]+" cb:"+ str(weights_init_cb[0])[1:4]+" cr:"+ str(weights_init_cr[0])[1:4]+"\n")
            text_file.write("Beta (gain) --- BN_y:"+ str(weights_init_y[1])[1:4]+" cb:"+ str(weights_init_cb[1])[1:4]+" cr:"+ str(weights_init_cr[1])[1:4]+"\n")
            text_file.write("Mean --- BN_y:"+ str(weights_init_y[2])[1:4]+" cb:"+ str(weights_init_cb[2])[1:4]+" cr:"+ str(weights_init_cr[2])[1:4]+"\n")
            text_file.write("Variance --- BN_y:"+ str(weights_init_y[3])[1:4]+" cb:"+ str(weights_init_cb[3])[1:4]+" cr:"+ str(weights_init_cr[3])[1:4]+"\n")
                
        text_file.write("2. Gains BatchNormS Intermediaires (MOYENNE) -------"+"\n")
        weights_bn_conv1 = model.get_layer("T_Bn0_0").get_weights()
        weights_bn_conv2 = model.get_layer("T_Bn0_1").get_weights()
        weights_bn_conv3 = model.get_layer("T_Bn0_2").get_weights()
        weights_bn_conv4 = model.get_layer("T_Bn0_3").get_weights()
        text_file.write("Gamma (gain) --- BN_convo1:"+ str(np.mean(weights_bn_conv1[0]))[1:4]+" BN_convo2:"+ str(np.mean(weights_bn_conv2[0]))[1:4]+" BN_convo3:"+ str(np.mean(weights_bn_conv3[0]))[1:4]+"\n"+" BN_convo4:"+ str(np.mean(weights_bn_conv4[0]))[1:4]+"\n")
        text_file.write("Beta (gain) --- BN_convo1:"+ str(np.mean(weights_bn_conv1[1]))[1:4]+" BN_convo2:"+ str(np.mean(weights_bn_conv2[1]))[1:4]+" BN_convo3:"+ str(np.mean(weights_bn_conv3[1]))[1:4]+"\n"+" BN_convo4:"+ str(np.mean(weights_bn_conv4[1]))[1:4]+"\n")
        text_file.write("Mean --- BN_convo1:"+ str(np.mean(weights_bn_conv1[2]))[1:4]+" BN_convo2:"+ str(np.mean(weights_bn_conv2[2]))[1:4]+" BN_convo3:"+ str(np.mean(weights_bn_conv3[2]))[1:4]+"\n"+" BN_convo4:"+ str(np.mean(weights_bn_conv4[2]))[1:4]+"\n")
        text_file.write("Variance --- BN_convo1:"+ str(np.mean(weights_bn_conv1[3]))[1:4]+" BN_convo2:"+ str(np.mean(weights_bn_conv2[3]))[1:4]+" BN_convo3:"+ str(np.mean(weights_bn_conv3[3]))[1:4]+"\n"+" BN_convo4:"+ str(np.mean(weights_bn_conv4[3]))[1:4]+"\n")
                
        text_file.write("NO MODULE FBéta -------"+"\n")
        
    print(root_rep," statistiques saved")

def statistique_branch_St3(model,nombre_class,DOG_init,DOG_fin,BN_init,BN_fin,root_rep):
    '''
    Load Batch_Normalization weights from specific layers for STy branch.
    '''

    with open(os.path.join(root_rep,"_PoidsConv.txt"), "w") as text_file:
        text_file.write(" -------  COL NETWORK -------"+"\n")
        if BN_init:
            text_file.write("1. Gains BatchNormS Init (y,cb,cr) -------"+"\n")
            weights_init_y = model.get_layer("ST3m_i_yDoGBn").get_weights()
            weights_init_cb = model.get_layer("ST3m_i_cbDoGBn").get_weights()
            weights_init_cr = model.get_layer("ST3m_i_crDoGBn").get_weights()
            text_file.write("Gamma (gain) --- BN_y:"+ str(weights_init_y[0])[1:4]+" cb:"+ str(weights_init_cb[0])[1:4]+" cr:"+ str(weights_init_cr[0])[1:4]+"\n")
            text_file.write("Beta (gain) --- BN_y:"+ str(weights_init_y[1])[1:4]+" cb:"+ str(weights_init_cb[1])[1:4]+" cr:"+ str(weights_init_cr[1])[1:4]+"\n")
            text_file.write("Mean --- BN_y:"+ str(weights_init_y[2])[1:4]+" cb:"+ str(weights_init_cb[2])[1:4]+" cr:"+ str(weights_init_cr[2])[1:4]+"\n")
            text_file.write("Variance --- BN_y:"+ str(weights_init_y[3])[1:4]+" cb:"+ str(weights_init_cb[3])[1:4]+" cr:"+ str(weights_init_cr[3])[1:4]+"\n")
                
        text_file.write("2. Gains BatchNormS Intermediaires (MOYENNE) -------"+"\n")
        weights_bn_conv1 = model.get_layer("T_Bn0_0").get_weights()
        weights_bn_conv2 = model.get_layer("T_Bn0_1").get_weights()
        weights_bn_conv3 = model.get_layer("T_Bn0_2").get_weights()
        weights_bn_conv4 = model.get_layer("T_Bn0_3").get_weights()
        text_file.write("Gamma (gain) --- BN_convo1:"+ str(np.mean(weights_bn_conv1[0]))[1:4]+" BN_convo2:"+ str(np.mean(weights_bn_conv2[0]))[1:4]+" BN_convo3:"+ str(np.mean(weights_bn_conv3[0]))[1:4]+"\n"+" BN_convo4:"+ str(np.mean(weights_bn_conv4[0]))[1:4]+"\n")
        text_file.write("Beta (gain) --- BN_convo1:"+ str(np.mean(weights_bn_conv1[1]))[1:4]+" BN_convo2:"+ str(np.mean(weights_bn_conv2[1]))[1:4]+" BN_convo3:"+ str(np.mean(weights_bn_conv3[1]))[1:4]+"\n"+" BN_convo4:"+ str(np.mean(weights_bn_conv4[1]))[1:4]+"\n")
        text_file.write("Mean --- BN_convo1:"+ str(np.mean(weights_bn_conv1[2]))[1:4]+" BN_convo2:"+ str(np.mean(weights_bn_conv2[2]))[1:4]+" BN_convo3:"+ str(np.mean(weights_bn_conv3[2]))[1:4]+"\n"+" BN_convo4:"+ str(np.mean(weights_bn_conv4[2]))[1:4]+"\n")
        text_file.write("Variance --- BN_convo1:"+ str(np.mean(weights_bn_conv1[3]))[1:4]+" BN_convo2:"+ str(np.mean(weights_bn_conv2[3]))[1:4]+" BN_convo3:"+ str(np.mean(weights_bn_conv3[3]))[1:4]+"\n"+" BN_convo4:"+ str(np.mean(weights_bn_conv4[3]))[1:4]+"\n")
                
        text_file.write("NO MODULE FBéta -------"+"\n")
        
    print(root_rep," statistiques saved")
    
def Image_benchmark_BRANCH(model_identite,main_network:str,type_branch:str,ponderation_features:list, nombre_class:int,border:int,filtres:int,filtres_branch:int, nombre_kernel:int,save_rep:str,root_folder:str,BN_init:bool,BN_fin:bool,DOG_init:bool, DOG_fin:bool,  sigma_noise_blur:float,FG_sigma_init:int,FG_sigma_puissance:int,benchmark_folder:str, cbcr_sty:bool, w_h,w_v, w_h_s,w_v_s,style,clusters,bins):
                                
    """
    Test on Benchmark Images for MAIN Network (composed by SR branches)
    """ 
    rep_images = os.path.join(benchmark_folder,"BENCHMARK_VISUEL/Rapports")
    liste_images = [x for x in os.listdir(rep_images) if x.endswith(".png")]
    x_m,y_m,x_max,y_max=300,300,900,900
    R=4
    tf.keras.backend.clear_session()

    # Change Input size
    if main_network=="SR_EDSR":
        new_model = model_identite
    else:
        new_model = MAIN_network_none(nombre_class=nombre_class,filtres=filtres, ponderation_features=ponderation_features,kernel=nombre_kernel,w_h=w_h,w_v=w_v, BN_init=BN_init, BN_fin=BN_fin, DOG_init=DOG_init, DOG_fin=DOG_fin)
    if type_branch=="Stycbcr":
        if cbcr_sty:
            new_model= STy_branch_col_none(model=new_model, main_network=main_network,filtres_branch=filtres_branch,border=border,DOG_init=DOG_init, DOG_fin=DOG_fin, BN_init=BN_init, BN_fin=BN_fin,nombre_kernel=nombre_kernel,w_h_s=w_h_s,w_v_s=w_v_s)
        else:
            new_model= STy_branch_none(model=new_model, main_network=main_network,filtres_branch=filtres_branch,border=border,DOG_init=DOG_init, DOG_fin=DOG_fin, BN_init=BN_init, BN_fin=BN_fin,nombre_kernel=nombre_kernel,w_h_s=w_h_s,w_v_s=w_v_s)
        
    if type_branch=="Stcol":
        new_model = STrcol_branch_none(model=new_model,main_network=main_network,filtres_branch=filtres_branch, border=border,DOG_init=DOG_init, DOG_fin=DOG_fin, BN_init=BN_init, BN_fin=BN_fin,nombre_kernel=nombre_kernel,w_h_s=w_h_s,w_v_s=w_v_s)
    
    if type_branch=="St3":
        new_model = ST3_branch_none(model=new_model,main_network=main_network,filtres_branch=filtres_branch, border=border,DOG_init=DOG_init, DOG_fin=DOG_fin, BN_init=BN_init, BN_fin=BN_fin,nombre_kernel=nombre_kernel,w_h_s=w_h_s,w_v_s=w_v_s)
    
    new_model =copy_parameters(model=new_model,model_patch=model_identite)
    nombre_image=len(liste_images)
    for img in liste_images :
        print("Traitement de l'image: "+str(img))
        
        # Data Preparation & Input
        img_array_LR = Ouverture_img(os.path.join(os.path.join(os.path.join(os.path.join(rep_images,"LR"),img.replace(".png","_LR.png")))),1)[0]#.astype(np.float64)
        img_array_LR=img_array_LR[int(x_m/R):int(x_max/R),int(y_m/R):int(y_max/R),:]/256.
        img_array_HR=Ouverture_img(os.path.join(os.path.join(os.path.join(rep_images,img))),1)[0]#.astype(np.float64)
        img_array_HR=img_array_HR[x_m:x_max,y_m:y_max,:]/256.
        
        if main_network=="SR":
            INPUT = upsampling(img_array_LR,R) 
        elif main_network=="DENOISING":
            INPUT = img_array_HR.copy()
            INPUT[:,:,0]+=np.random.normal(0,0.2,(img_array_HR.shape[0],img_array_HR.shape[1]))
        elif main_network=="BLURRING":
            INPUT = img_array_HR.copy()
            INPUT_y = gaussian_filter(INPUT[:,:,0], sigma=sigma_noise_blur)
            INPUT[:,:,0]=INPUT_y
        elif main_network=="SR_EDSR":
            INPUT=img_array_LR.copy()*255.
        INPUT=np.expand_dims(INPUT,axis=0)
        TRUE=np.expand_dims(img_array_HR,axis=0)
        
        # Extraction des features et Calculs inférence sur l'image
        if type_branch=="Stycbcr":
            STm_i_y,STm_i_cb,STm_i_cr, STm_i_yDoG,STm_i_cbDoG,STm_i_crDoG,   STm_i_yDoGBn,STm_i_cbDoGBn,STm_i_crDoGBn ,STm_o_y,STm_o_yDoG,STm_o_yDoGBn,STm_o_yFinal,STm_o_yFinal_residu = extract_layers__STr_y(new_model,BN_init,BN_fin,DOG_init,DOG_fin)
            prediction, STf_i_yDoG,STf_i_cbDoG,STf_i_crDoG,STf_i_yDoGBn,STf_i_cbDoGBn,STf_i_crDoGBn,STf_o_y,STf_o_yDoG,STf_o_yDoGBn,STf_o_yFinal = compute_features_Sty(STm_i_y,STm_i_cb,STm_i_cr,STm_i_yDoG,STm_i_cbDoG,STm_i_crDoG,STm_i_yDoGBn,STm_i_cbDoGBn,STm_i_crDoGBn,STm_o_y,STm_o_yDoG,STm_o_yDoGBn,STm_o_yFinal,DOG_init,DOG_fin,BN_init,BN_fin,new_model,INPUT ) 
            
        if type_branch=="Stcol":
            COLm_i_y,COLm_i_cb,COLm_i_cr, COLm_i_yDoG,COLm_i_cbDoG,COLm_i_crDoG,   COLm_i_yDoGBn,COLm_i_cbDoGBn,COLm_i_crDoGBn ,COLm_o_cb,COLm_o_cbDoG,COLm_o_cbDoGBn,COLm_o_cbFinal,COLm_o_cbFinalresidu,COLm_o_cr,COLm_o_crDoG,COLm_o_crDoGBn,COLm_o_crFinal,COLm_o_crFinalresidu=extract_layers_STr_col(new_model,BN_init,BN_fin,DOG_init,DOG_fin)  
            prediction ,COLf_o_cbFinalresidu,COLf_o_crFinalresidu=compute_features_Stcol(COLm_i_y,COLm_i_cb,COLm_i_cr, COLm_i_yDoG,COLm_i_cbDoG,COLm_i_crDoG,   COLm_i_yDoGBn,COLm_i_cbDoGBn,COLm_i_crDoGBn ,COLm_o_cb,COLm_o_cbDoG,COLm_o_cbDoGBn,COLm_o_cbFinal,COLm_o_cbFinalresidu,COLm_o_cr,COLm_o_crDoG,COLm_o_crDoGBn,COLm_o_crFinal,COLm_o_crFinalresidu,DOG_init,DOG_fin,BN_init,BN_fin,new_model,INPUT )

        if type_branch =="St3":
            ST3m_i_y,ST3m_i_cb,ST3m_i_cr, ST3m_i_yDoG,ST3m_i_cbDoG,ST3m_i_crDoG,   ST3m_i_yDoGBn,ST3m_i_cbDoGBn,ST3m_i_crDoGBn ,ST3m_o_y,ST3m_o_cb,ST3m_o_yDoG,ST3m_o_cbDoG,ST3m_o_cbDoGBn,ST3m_o_yDoGBn,ST3m_o_cbFinal,ST3m_o_yFinal,ST3m_o_cbFinalresidu,ST3m_o_yFinalresidu,ST3m_o_cr,ST3m_o_crDoG,ST3m_o_crDoGBn,ST3m_o_crFinal,ST3m_o_crFinalresidu = extract_layers_STr3(new_model,BN_init,BN_fin,DOG_init,DOG_fin)  
            prediction ,ST3f_o_yFinal_residu,ST3f_o_cbFinal_residu,ST3f_o_crFinal_residu=  compute_features_St3(ST3m_i_y,ST3m_i_cb,ST3m_i_cr, ST3m_i_yDoG,ST3m_i_cbDoG,ST3m_i_crDoG,   ST3m_i_yDoGBn,ST3m_i_cbDoGBn,ST3m_i_crDoGBn ,ST3m_o_y,ST3m_o_cb,ST3m_o_yDoG,ST3m_o_cbDoG,ST3m_o_cbDoGBn,ST3m_o_yDoGBn,ST3m_o_cbFinal,ST3m_o_yFinal,ST3m_o_cbFinalresidu,ST3m_o_yFinalresidu,ST3m_o_cr,ST3m_o_crDoG,ST3m_o_crDoGBn,ST3m_o_crFinal,ST3m_o_crFinalresidu,DOG_init,DOG_fin,BN_init,BN_fin,new_model,INPUT ) 
        
        y,cb,cr =RGB2Ycbcr_numpy(prediction[0,:,:,0],prediction[0,:,:,1],prediction[0,:,:,2])
        prediction_ycbcr= np.stack([y,cb,cr],axis=-1)
        
        # FOLDER & SAVING
        rep_img_save = os.path.join(os.path.join(os.path.join(save_rep,"rapports_benchmark"),str(img)))
        ensure_dir(rep_img_save)
        
        # TENSORS to display
        if main_network=="SR_EDSR": # adapting report if main model is already trained
            list_tensor_input,list_tensor_intermediate_features= [TRUE[0,:,:,0],TRUE[0,:,:,:],prediction_ycbcr[:,:,0],prediction_ycbcr,style],[]
            name_tensor_input,name_tensor_intermediate_features=["True_y","True","Output_y","Output","style_patch"],[]
            INPUT=INPUT/255.
        else:
            if type_branch=="Stycbcr":
                tbadded=TRUE[0,border:INPUT.shape[1]-border,border:INPUT.shape[2]-border,0]-INPUT[0,border:INPUT.shape[1]-border,border:INPUT.shape[2]-border,0]
                added=prediction_ycbcr[:,:,0].copy()-INPUT[0,border:INPUT.shape[1]-border,border:INPUT.shape[2]-border,0]
                
                
                style_02,style_05,style_08 = INPUT[0,border:INPUT.shape[1]-border,border:INPUT.shape[2]-border,0].copy(),INPUT[0,border:INPUT.shape[1]-border,border:INPUT.shape[2]-border,0].copy(),INPUT[0,border:INPUT.shape[1]-border,border:INPUT.shape[2]-border,0].copy()
                style_02 += 0.2*added.copy()
                style_05 += 0.5*added.copy()
                style_08 += 0.8*added.copy()

                list_tensor_input= [INPUT[0,:,:,0],INPUT[0,:,:,:], TRUE[0,:,:,0],TRUE[0,:,:,:],y,style_02,style_05,style_08,prediction_ycbcr,style]
                list_tensor_intermediate_features=[STf_o_y[0][:,:,0],STf_o_yDoG[0][:,:,0],tbadded,added]
                
                name_tensor_input=["Input_y","Input","True_y","True","Output_y","Output(0.2)","Output(0.5)","Output(0.8)","Output",'style_patch']
                name_tensor_intermediate_features=["Output_BRANCH","Output_BRANCH_DoG",'To be Added','Added']
                
            if type_branch=="Stcol":
                # hist _ pred
                cbcr=np.stack([prediction_ycbcr[:,:,1],prediction_ycbcr[:,:,2]],-1)
                cbcr = np.reshape(cbcr,(cbcr.shape[0]*cbcr.shape[1],2)).astype(np.float32)
                hist_out = histogram_2d(cbcr, clusters , cbcr.shape[0],cbcr.shape[1])       
                Save_Hist_1d_tf(hist_out,os.path.join(rep_img_save,str(img.replace(".png",""))+"__HIST1D_out__.png"),bins)
                Save_Hist_2d_tf(hist_out,os.path.join(rep_img_save,str(img.replace(".png",""))+"__HIST2D_out__.png"),bins)
                
                # hist _ true
                cbcr=np.stack([TRUE[:,:,:,1],TRUE[:,:,:,2]],-1)
                cbcr = np.reshape(cbcr,(cbcr.shape[1]*cbcr.shape[2],2)).astype(np.float32)
                hist_out = histogram_2d(cbcr, clusters , cbcr.shape[0],cbcr.shape[1])       
                Save_Hist_1d_tf(hist_out,os.path.join(rep_img_save,str(img.replace(".png",""))+"__HIST1D_true__.png"),bins)
                Save_Hist_2d_tf(hist_out,os.path.join(rep_img_save,str(img.replace(".png",""))+"__HIST2D_true__.png"),bins)
        
        
                list_tensor_input= [INPUT[0,:,:,0],INPUT[0,:,:,:], TRUE[0,:,:,0],TRUE[0,:,:,:],prediction_ycbcr,style]
                list_tensor_intermediate_features=[COLf_o_cbFinalresidu[0][0,:,:],COLf_o_crFinalresidu[0][0,:,:]]
                    
                name_tensor_input,name_tensor_intermediate_features=["Input_y","Input","True_y","True","Output",'style_patch'],["Output_cb","Output_cr"]
            
            if type_branch=="St3":
                list_tensor_input= [INPUT[0,:,:,0],INPUT[0,:,:,:], TRUE[0,:,:,0],TRUE[0,:,:,:],prediction_ycbcr,style]
                list_tensor_intermediate_features=[ST3f_o_yFinal_residu[0,:,:,0],ST3f_o_cbFinal_residu[0,:,:,0],ST3f_o_crFinal_residu[0,:,:,0]]
                    
                name_tensor_input,name_tensor_intermediate_features=["Input_y","Input","True_y","True","Output",'style_patch'],["Output_y","Output_cb","Output_cr"]
            
        # PDF Report
        Benchmark_report(200,300,350,list_tensor_input, list_tensor_intermediate_features, name_tensor_input, name_tensor_intermediate_features,
                                type_branch=type_branch,save_rep=rep_img_save,taille=TRUE.shape[1],nombre_class=nombre_class,root=root_folder,border=border, nom = img)

        # PNG files
        Save_tensor(list_tensor=[INPUT,TRUE,np.expand_dims(prediction_ycbcr,axis=0)],list_name=["input","true","prediction"],rep_save=rep_img_save)
        
        if type_branch=="Stycbcr":
            Save_tensor(list_tensor=[STf_o_yFinal],list_name=["Output_y"],rep_save=rep_img_save)
            
        if type_branch=="Stcol":
            Save_tensor(list_tensor=[COLf_o_cbFinalresidu,COLf_o_crFinalresidu],list_name=["Branch_output_cb_stcol","Branch_output_cr_stcol"],rep_save=rep_img_save)
        
        if type_branch =="St3":
            Save_tensor(list_tensor=[ST3f_o_yFinal_residu,ST3f_o_cbFinal_residu,ST3f_o_crFinal_residu],list_name=["Branch_output_y_st3","Branch_output_cb_st3","Branch_output_cr_st3"],rep_save=rep_img_save)
            
def process_test_controle_BRANCH(model,style_patch,clusters,bins,main_network:str, root:str,save_rep:str, save_rep_energie:str,taille:int,nombre_class:int, border:int,BN_init:bool, BN_fin:bool, DOG_fin:bool, DOG_init:bool,nombre_patch_test:int,type_branch:str,sigma_noise_blur:float):
    """
    Evaluation of testing patchs (qualitatively) for BRANCH NN 'model'. Generates visual reports. Same functions for all the branch NN.
    """
    paquet = nombre_patch_test//8
    
    gen=Generator_test_patch(main_network=main_network, size_output=taille,  border=border, root=root)
    multiple = int(nombre_patch_test/30)
        
    if type_branch=="Stycbcr":
        STm_i_y,STm_i_cb,STm_i_cr, STm_i_yDoG,STm_i_cbDoG,STm_i_crDoG,   STm_i_yDoGBn,STm_i_cbDoGBn,STm_i_crDoGBn ,STm_o_y,STm_o_yDoG,STm_o_yDoGBn,STm_o_yFinal,STm_o_yFinal_residu = extract_layers__STr_y(model,BN_init,BN_fin,DOG_init,DOG_fin)
    if type_branch=="Stcol":
        COLm_i_y,COLm_i_cb,COLm_i_cr, COLm_i_yDoG,COLm_i_cbDoG,COLm_i_crDoG,   COLm_i_yDoGBn,COLm_i_cbDoGBn,COLm_i_crDoGBn ,COLm_o_cb,COLm_o_cbDoG,COLm_o_cbDoGBn,COLm_o_cbFinal,COLm_o_cbFinalresidu,COLm_o_cr,COLm_o_crDoG,COLm_o_crDoGBn,COLm_o_crFinal,COLm_o_crFinalresidu=extract_layers_STr_col(model,BN_init,BN_fin,DOG_init,DOG_fin)  
    if type_branch =="St3":
        ST3m_i_y,ST3m_i_cb,ST3m_i_cr, ST3m_i_yDoG,ST3m_i_cbDoG,ST3m_i_crDoG,   ST3m_i_yDoGBn,ST3m_i_cbDoGBn,ST3m_i_crDoGBn ,ST3m_o_y,ST3m_o_cb,ST3m_o_yDoG,ST3m_o_cbDoG,ST3m_o_cbDoGBn,ST3m_o_yDoGBn,ST3m_o_cbFinal,ST3m_o_yFinal,ST3m_o_cbFinalresidu,ST3m_o_yFinalresidu,ST3m_o_cr,ST3m_o_crDoG,ST3m_o_crDoGBn,ST3m_o_crFinal,ST3m_o_crFinalresidu = extract_layers_STr3(model,BN_init,BN_fin,DOG_init,DOG_fin)  
            
    nombre_boucle = nombre_patch_test // paquet
    image_debut_boucle = 0  
    
    if main_network=="SR_EDSR":
        taille_i=int(taille/4)
    else:
        taille_i=taille
        
    for ind in range(nombre_boucle):
        for i in range(image_debut_boucle,(ind+1)*paquet): 

            if (i%multiple==1):
            
                nom =gen[0].filenames[i].split("/")[-1].replace(".png","")
                
                INPUT = gen[0][i].reshape(taille_i+2*border,taille_i+2*border,3)
                if main_network=="DENOISING":
                    INPUT[:,:,0]+=np.random.normal(0,sigma_noise_blur,(INPUT.shape[0],INPUT.shape[1]))            
                elif main_network=="BLURRING":
                    INPUT_y = gaussian_filter(INPUT[:,:,0], sigma=sigma_noise_blur)
                    INPUT[:,:,0]=INPUT_y
                    
                INPUT=np.expand_dims(INPUT,axis=0)
                TRUE_tf= gen[1][i]
    
                if main_network=="SR_EDSR":
                    INPUT=INPUT[:,border:INPUT.shape[0]-border-1,border:INPUT.shape[1]-border,:]*255.
                # Features computations
                if type_branch=="Stycbcr":    
                    prediction, STf_i_yDoG,STf_i_cbDoG,STf_i_crDoG,STf_i_yDoGBn,STf_i_cbDoGBn,STf_i_crDoGBn,STf_o_y,STf_o_yDoG,STf_o_yDoGBn,STf_o_yFinal = compute_features_Sty(STm_i_y,STm_i_cb,STm_i_cr,STm_i_yDoG,STm_i_cbDoG,STm_i_crDoG,STm_i_yDoGBn,STm_i_cbDoGBn,STm_i_crDoGBn,STm_o_y,STm_o_yDoG,STm_o_yDoGBn,STm_o_yFinal,DOG_init,DOG_fin,BN_init,BN_fin,model,INPUT ) 
                    
                if type_branch=="Stcol":
                    prediction ,COLf_o_cbFinalresidu,COLf_o_crFinalresidu=compute_features_Stcol(COLm_i_y,COLm_i_cb,COLm_i_cr, COLm_i_yDoG,COLm_i_cbDoG,COLm_i_crDoG,   COLm_i_yDoGBn,COLm_i_cbDoGBn,COLm_i_crDoGBn ,COLm_o_cb,COLm_o_cbDoG,COLm_o_cbDoGBn,COLm_o_cbFinal,COLm_o_cbFinalresidu,COLm_o_cr,COLm_o_crDoG,COLm_o_crDoGBn,COLm_o_crFinal,COLm_o_crFinalresidu,DOG_init,DOG_fin,BN_init,BN_fin,model,INPUT )
        
                if type_branch =="St3":        
                    prediction ,ST3f_o_yFinal_residu,ST3f_o_cbFinal_residu,ST3f_o_crFinal_residu=  compute_features_St3(ST3m_i_y,ST3m_i_cb,ST3m_i_cr, ST3m_i_yDoG,ST3m_i_cbDoG,ST3m_i_crDoG,   ST3m_i_yDoGBn,ST3m_i_cbDoGBn,ST3m_i_crDoGBn ,ST3m_o_y,ST3m_o_cb,ST3m_o_yDoG,ST3m_o_cbDoG,ST3m_o_cbDoGBn,ST3m_o_yDoGBn,ST3m_o_cbFinal,ST3m_o_yFinal,ST3m_o_cbFinalresidu,ST3m_o_yFinalresidu,ST3m_o_cr,ST3m_o_crDoG,ST3m_o_crDoGBn,ST3m_o_crFinal,ST3m_o_crFinalresidu,DOG_init,DOG_fin,BN_init,BN_fin,model,INPUT ) 
            
                y,cb,cr =RGB2Ycbcr_numpy(prediction[0,:,:,0],prediction[0,:,:,1],prediction[0,:,:,2])
                prediction_ycbcr= np.stack([y,cb,cr],axis=-1)
            
                # Tensor to display
                if main_network=="SR_EDSR": # adapting report if main model is already trained
                    list_tensor_input,list_tensor_intermediate_features= [TRUE_tf[0,:,:,0],TRUE_tf[0,:,:,:],prediction_ycbcr[:,:,0],prediction_ycbcr,style_patch],[]
                    name_tensor_input,name_tensor_intermediate_features=["True_y","True","Output_y","Output","style_patch"],[]
                    INPUT=INPUT/255.
                else:
                    if type_branch=="Stycbcr" :    
                        tbadded=TRUE_tf[0,:,:,0]-INPUT[0,border:INPUT.shape[0]-border-1,border:INPUT.shape[1]-border,0]
                        added=prediction_ycbcr[:,:,0]-INPUT[0,border:INPUT.shape[0]-border-1,border:INPUT.shape[1]-border,0]
                        
                        list_tensor_input= [INPUT[0,:,:,0],INPUT[0,:,:,:], TRUE_tf[0,:,:,0],TRUE_tf[0,:,:,:],STf_o_yFinal[0,:,:,0],prediction_ycbcr,style_patch]
                        list_tensor_intermediate_features=[STf_o_y[0,:,:,0],STf_o_yDoG[0,:,:,0],tbadded,added]
                            
                        name_tensor_input,name_tensor_intermediate_features=["Input_y","Input","True_y","True","Output_y","Output","style_patch"],["Output_BRANCH","Output_BRANCH_DoG","To be added","Added"]
                        
                    if type_branch=="Stcol":
                        list_tensor_input,list_tensor_intermediate_features= [INPUT[0,:,:,0],INPUT[0,:,:,:], TRUE_tf[0,:,:,0],TRUE_tf[0,:,:,:],prediction_ycbcr,style_patch],[cb,cr]
                        name_tensor_input, name_tensor_intermediate_features=["Input_y","Input","True_y","True","Output","style_patch"],["cb","cr"]
                        
                        # PNG histogram -- Out
                        cbcr=np.stack([prediction_ycbcr[:,:,1],prediction_ycbcr[:,:,2]],-1)
                        cbcr = np.reshape(cbcr,(cbcr.shape[0]*cbcr.shape[1],2)).astype(np.float32)
                        hist_out = histogram_2d(cbcr, clusters , cbcr.shape[0],cbcr.shape[1])       
                        Save_Hist_1d_tf(hist_out,os.path.join(save_rep,str(nom.replace(".png",""))+"__HIST1D_out__.png"),bins)
                        Save_Hist_2d_tf(hist_out,os.path.join(save_rep,str(nom.replace(".png",""))+"__HIST2D_out__.png"),bins)
                        
                        # hist _ true
                        cbcr=np.stack([TRUE_tf[:,:,:,1],TRUE_tf[:,:,:,2]],-1)
                        cbcr = np.reshape(cbcr,(cbcr.shape[1]*cbcr.shape[2],2)).astype(np.float32)
                        hist_out = histogram_2d(cbcr, clusters , cbcr.shape[0],cbcr.shape[1])       
                        Save_Hist_1d_tf(hist_out,os.path.join(save_rep,str(nom.replace(".png",""))+"__HIST1D_true__.png"),bins)
                        Save_Hist_2d_tf(hist_out,os.path.join(save_rep,str(nom.replace(".png",""))+"__HIST2D_true__.png"),bins)
                    
                    if type_branch =="St3":        
                        list_tensor_input= [INPUT[0,:,:,0],INPUT[0,:,:,:], TRUE_tf[0,:,:,0],TRUE_tf[0,:,:,:],prediction_ycbcr,style_patch]
                        list_tensor_intermediate_features=[prediction_ycbcr[:,:,0],prediction_ycbcr[:,:,1],prediction_ycbcr[:,:,2]]
                        name_tensor_input,name_tensor_intermediate_features=["Input_y","Input","True_y","True","Output","style_patch"],["y","cb","cr"]
                    
                    # PDF Report
                Patch_report_features(list_tensor_input, list_tensor_intermediate_features, name_tensor_input, name_tensor_intermediate_features,0,0,save_rep=save_rep,taille=taille,nombre_class=nombre_class,root=root,border=border,nom=nom,ecart=22,numero_patch=i)
                
        image_debut_boucle  = (ind+1)*paquet+1 

    
    
    