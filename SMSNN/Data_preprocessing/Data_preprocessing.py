import numpy as np
import os
import imageio
import random
from scipy.ndimage import gaussian_filter

from various_functions.tensor_tf_data_fonctions import *
from various_functions.rapports_pdf_fonctions import *
from various_functions.numpy_data_fonctions import *
from Networks.model_management import *
from various_functions.directory_management import *
from various_functions.custom_filters import *


def save_png_sr(im_HR_y:np.array,im_HR_ycbcr:np.array,im_LR_y:np.array,im_LR_bilinear_y:np.array,im_LR_ycbcr:np.array,im_LR_bilinear_ycbcr:np.array,patch_HR_ycbcr_nocrop:np.array,
                name:str, root_folder:str, bdd:str,size:int)->None:
    """
    Save patches data as Ycbcr image (.png) in specific folder (/train /validation /test).
    
    * im_HR_y, im_HR_ycbcr, im_LR_y, im_LR_bilinear_y,im_LR_ycbcr, im_LR_bilinear_ycbcr,patch_HR_ycbcr_nocrop (numpy arrays) corresponding to different patches to save in different folders and used afterwards for training MAIN NN or BRANCH NN.
    * root_folder (str) : root where all the patches are saved
    * bdd (str) : either 'train' 'validation' or 'test'
    * name (str) : name number of the patch
    * size (int) : size of the patches to be saved corresponding to the HR patch size associated to the patch, appeared in the folder
    
    Returns 
    None (Save patches)
    """ 
    # Folder definition for each patch type         
    root_HR_ycbcr=os.path.join(root_folder,str(bdd)+"_"+str(size)+"/HR_ycbcr/HR_ycbcr")
    root_HR_y=os.path.join(root_folder,str(bdd)+"_"+str(size)+"/HR_y/HR_y")
    root_LR_y=os.path.join(root_folder,str(bdd)+"_"+str(size)+"/LR/LR")
    root_LR_bilinear_y=os.path.join(root_folder,str(bdd)+"_"+str(size)+"/LR_bilinear/LR_bilinear")
    root_LR_ycbcr=os.path.join(root_folder,str(bdd)+"_"+str(size)+"/LR_ycbcr/LR_ycbcr")
    root_LR_bilinear_ycbcr=os.path.join(root_folder,str(bdd)+"_"+str(size)+"/LR_bilinear_ycbcr/LR_bilinear_ycbcr")
    root_HR_ycbcr_nocrop = os.path.join(root_folder,str(bdd)+"_"+str(size)+"/HR_ycbcr_nocrop/HR_ycbcr_nocrop")
        
    # Patches saving
    imageio.imwrite(root_HR_ycbcr+"/"+str(name),im_HR_ycbcr)
    imageio.imwrite(root_HR_y+"/"+str(name),im_HR_y)
    imageio.imwrite(root_LR_y+"/"+str(name),im_LR_y)
    imageio.imwrite(root_LR_bilinear_y+"/"+str(name),im_LR_bilinear_y)
    imageio.imwrite(root_LR_ycbcr+"/"+str(name),im_LR_ycbcr)
    imageio.imwrite(root_LR_bilinear_ycbcr+"/"+str(name),im_LR_bilinear_ycbcr)
    imageio.imwrite(root_HR_ycbcr_nocrop+"/"+str(name),patch_HR_ycbcr_nocrop)




def data_preparation_sr(type_upsampling:str,type_downsampling:str, bluring_downsampling:float, R:int,size:int,delta:int,root_folder:str,seuil:int,border:int):    
    """
    Build patches out of DIV2K image dataset (in 'image_folder).
    Sort patches into /train /valid & /test folders depending on image number
    Patches are then used to train MAIN NN, and BRANCH NN, cross validating models during training and test models after training.
    
    * root_folder (str) : path where patches will be saved
    * type_upsampling, type_downsampling (str) : either 'bicubic' or 'bilinear'(type_upsampling), "bicubic' 'bilinear' or 'automatic' (type_downsampling).
        if type_upsampling=='automatic' : then DIV2K bicubic dataset is used as bicubic interpolation / if not , interpolated data are build out of downsampling, blurring and upsampling, depending on 'bicubic' or 'bilinear down/up-sampling
    * bluring_downsampling(int) , if type_upsampling not 'automatic : blur used for building LR data
    * proportion (float)
    * R (int) : upsampling factor
    * delta (int) : overlap between patches
    * seuil(int) : minimum variance to have for a patch to be saved
    * border (int) : border added for some of the patches (overlap is border + delta)
    
    Returns 
    * compteur_save (int) : number of patches saved (square & in folders as .png)
    * seuil(int)
    """
    
    image_folder = os.path.join(root_folder,"External_Data/ORIGINAL_DATASET") # path where external dataset like DIV2K are saved
    print(image_folder)
    ensure_dir(os.path.join(root_folder,"Patchs"))
    for i in [os.path.join(root_folder,"Patchs"+"/test_"+str(size)),os.path.join(root_folder,"Patchs"+"/train_"+str(size)),os.path.join(root_folder,"Patchs"+"/validation_"+str(size))]:
        ensure_dir(i)
        print(i)
    for i in [os.path.join(root_folder,"Patchs"+"/test_"+str(size)+"/LR/LR"),os.path.join(root_folder,"Patchs"+"/test_"+str(size)+"/LR_bilinear/LR_bilinear"),os.path.join(root_folder,"Patchs"+"/test_"+str(size)+"/HR_y/HR_y"),os.path.join(root_folder,"Patchs"+"/train_"+str(size)+"/LR_bilinear/LR_bilinear"),os.path.join(root_folder,"Patchs"+"/train_"+str(size)+"/LR/LR"),os.path.join(root_folder,"Patchs"+"/train_"+str(size)+"/HR_y/HR_y"),os.path.join(root_folder,"Patchs"+"/validation_"+str(size)+"/LR/LR"),os.path.join(root_folder,"Patchs"+"/validation_"+str(size)+"/HR_y/HR_y"),os.path.join(root_folder,"Patchs"+"/validation_"+str(size)+"/LR_bilinear/LR_bilinear"),os.path.join(root_folder,"Patchs"+"/validation_"+str(size)+"/LR_bilinear_ycbcr/LR_bilinear_ycbcr"),os.path.join(root_folder,"Patchs"+"/validation_"+str(size)+"/LR_ycbcr/LR_ycbcr"),os.path.join(root_folder,"Patchs"+"/train_"+str(size)+"/LR_bilinear_ycbcr/LR_bilinear_ycbcr"),os.path.join(root_folder,"Patchs"+"/train_"+str(size)+"/LR_ycbcr/LR_ycbcr"),os.path.join(root_folder,"Patchs"+"/test_"+str(size)+"/LR_bilinear_ycbcr/LR_bilinear_ycbcr"),os.path.join(root_folder,"Patchs"+"/test_"+str(size)+"/LR_ycbcr/LR_ycbcr"),os.path.join(root_folder,"Patchs"+"/test_"+str(size)+"/HR_ycbcr/HR_ycbcr"),os.path.join(root_folder,"Patchs"+"/train_"+str(size)+"/HR_ycbcr/HR_ycbcr"),os.path.join(root_folder,"Patchs"+"/validation_"+str(size)+"/HR_ycbcr/HR_ycbcr")
                , os.path.join(root_folder,"Patchs"+"/test_"+str(size)+"/HR_ycbcr_nocrop/HR_ycbcr_nocrop"),os.path.join(root_folder,"Patchs"+"/train_"+str(size)+"/HR_ycbcr_nocrop/HR_ycbcr_nocrop"),os.path.join(root_folder,"Patchs"+"/validation_"+str(size)+"/HR_ycbcr_nocrop/HR_ycbcr_nocrop")]:
        ensure_dir(i)
        delete_all_png(i)
        #delete_all_png_string(i,"14445")
        print("Repertoire "+str(i)+ " nettoyé")
      
    liste_images_train_test = [x for x in os.listdir(os.path.join(image_folder,"DIV2K/DIV2K_train_HR")) if x.endswith(".png")]
    liste_images_validation = [x for x in os.listdir(os.path.join(image_folder,"DIV2K/DIV2K_valid_HR")) if x.endswith(".png")]
    liste_images = liste_images_train_test + liste_images_validation
    liste_images=liste_images

    overlap=size-random.randint(0,delta)    
    compteur_save=0
    compteur_not_square=0
    for  i, image_nom in enumerate(liste_images): 
        print(f'{image_nom} {i}/{len(liste_images)}')
        numb=int(image_nom[0:4])
        if numb < 650: # Train dataset 1-649
            bdd="train"
        elif numb > 649 and numb < 750: # Test Dataset 650 - 749
            bdd="test"
        else: # Validation Dataset 750 900
            bdd="validation"
        print(" -- "+str(bdd)+" -- ")
        if bdd=="test" or bdd=="train":
            im_HR=Ouverture_img(os.path.join(os.path.join(image_folder,"DIV2K/DIV2K_train_HR"),image_nom),R)
            if type_downsampling=="automatic":
                im_LR=Ouverture_img(os.path.join(os.path.join(image_folder,"DIV2K/DIV2K_train_LR_bicubic_X"+str(R)+"/DIV2K_train_LR_bicubic/X"+str(R)),image_nom.replace(".png","x"+str(R)+".png")),R) 
        else:
            try:
                im_HR=Ouverture_img(os.path.join(os.path.join(image_folder,"DIV2K/DIV2K_valid_HR"),image_nom),R)
                if type_downsampling=="automatic":
                    im_LR=Ouverture_img(os.path.join(os.path.join(image_folder,"DIV2K/DIV2K_valid_LR_bicubic_X"+str(R)+"/DIV2K_valid_LR_bicubic/X"+str(R)),image_nom.replace(".png","x"+str(R)+".png")),R) 
                e=(im_HR[0].shape)
                e=(im_LR[0].shape)
            except AttributeError:
                im_HR=Ouverture_img(os.path.join(os.path.join(image_folder,"DIV2K/DIV2K_train_HR"),image_nom),R)
                if type_downsampling=="automatic":
                    im_LR=Ouverture_img(os.path.join(os.path.join(image_folder,"DIV2K/DIV2K_train_LR_bicubic_X"+str(R)+"/DIV2K_train_LR_bicubic/X"+str(R)),image_nom.replace(".png","x"+str(R)+".png")),R) 
                e=(im_HR[0].shape)
                e=(im_LR[0].shape)
             
        if type_downsampling=="bicubic": #on peut rajouter un bruit blanc eventuellement /// on blur seulement Y et on downsample Y et les couleurs (sous enchantillonage)
            #HR_ycbcr_blur=gaussian_filter(im_HR[0], sigma=bluring_downsampling)
            HR_y_blur = gaussian_filter(im_HR[1], sigma=bluring_downsampling)
            im_Y_down =  downsampling(HR_y_blur,R).reshape(int(HR_y_blur.shape[0]/R),int(HR_y_blur.shape[1]/R),1)
            im_cb_down = downsampling(im_HR[0][:,:,1].reshape(im_HR[0].shape[0],im_HR[0].shape[1]),R).reshape(int(HR_y_blur.shape[0]/R),int(HR_y_blur.shape[1]/R),1)
            im_cr_down = downsampling(im_HR[0][:,:,2].reshape(im_HR[0].shape[0],im_HR[0].shape[1]),R).reshape(int(HR_y_blur.shape[0]/R),int(HR_y_blur.shape[1]/R),1)
            im_LR_down_y = im_Y_down.reshape(int(HR_y_blur.shape[0]/R),int(HR_y_blur.shape[1]/R))
            im_LR_down_ycbcr = np.concatenate([im_Y_down,im_cb_down,im_cr_down],axis=-1).reshape(int(HR_y_blur.shape[0]/R),int(HR_y_blur.shape[1]/R),3)
            im_LR = [im_LR_down_ycbcr,im_LR_down_y]
            
        elif type_downsampling=="bilinear":
            HR_ycbcr_blur=gaussian_filter(im_HR[0], sigma=bluring_downsampling)
            HR_y_blur=gaussian_filter(im_HR[1], sigma=bluring_downsampling)          
            im_LR=[downsampling_bilinear(HR_ycbcr_blur,R),downsampling_bilinear(HR_y_blur,R)]
            
        if type(im_HR[0])!=type(None):
            im_HR_ycbcr=im_HR[0]
            im_HR_y=im_HR[1]

            im_LR_y=im_LR[1]
            im_LR_ycbcr = im_LR[0]

            if type_upsampling=="bicubic":
                im_LR_bilinear_ycbcr = upsampling(im_LR_ycbcr,R) 
                im_LR_bilinear_y = upsampling(im_LR_y,R)  
            elif  type_upsampling=="bilinear":  
                im_LR_bilinear_ycbcr = upsampling_bilinear(im_LR_ycbcr,R) 
                im_LR_bilinear_y = upsampling_bilinear(im_LR_y,R) 
            
            
            #Préparation Patchs   
            for offsetX in range(size,im_HR_y.shape[0]-size,overlap):       
                for offsetY in range(size,im_HR_y.shape[1]-size,overlap):
                           
                    patch_HR_ycbcr_nocrop =  decoupe_image(im_HR_ycbcr,offsetX-border,offsetY-border,size+2*border) 
                    patch_HR_ycbcr = decoupe_image(im_HR_ycbcr,offsetX,offsetY,size) 
                    patch_HR_y = decoupe_image(im_HR_y,offsetX,offsetY,size) 
                        
                    if test_variance(patch_HR_ycbcr,seuil):
                        patch_LR_ycbcr = decoupe_image(im_LR_ycbcr,int(offsetX/R)-border,int(offsetY/R)-border,int(size/R)+2*border)  
                        patch_LR_bilinear_ycbcr = decoupe_image(im_LR_bilinear_ycbcr,offsetX-border,offsetY-border,size+2*border)   
                            
                        patch_LR_y = decoupe_image(im_LR_y,int(offsetX/R)-border,int(offsetY/R)-border,int(size/R)+2*border)   
                        patch_LR_bilinear_y = decoupe_image(im_LR_bilinear_y,offsetX-border,offsetY-border,size+2*border) 
                            
                        nom = name_decoupe(compteur_save,image_nom,offsetX,offsetY,size) 
                        compteur_save+=1
                        
                        if patch_LR_y.shape[0]==patch_LR_y.shape[1]:
                            save_png_sr(patch_HR_y,patch_HR_ycbcr,patch_LR_y,patch_LR_bilinear_y,patch_LR_ycbcr,patch_LR_bilinear_ycbcr,patch_HR_ycbcr_nocrop,nom,os.path.join(root_folder,"Patchs"),bdd,size)
                        else:
                            compteur_not_square+=1

            print(str(compteur_save)+" saved patches")
            print(str(compteur_not_square)+" patchs which are not square")
        else:
            print(f'  {image_nom} is corrupted, and ignored')
            
            
    return(compteur_save,seuil)

