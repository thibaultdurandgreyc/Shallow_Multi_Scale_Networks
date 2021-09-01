from various_functions.tensor_tf_data_fonctions import *
from various_functions.rapports_pdf_fonctions import *
from various_functions.numpy_data_fonctions import *
from Networks.model_management import *
from various_functions.directory_management import *
# Architectures

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import convolve2d, medfilt # convol2d et filtre median
from mpl_toolkits.axes_grid1 import ImageGrid
from math import log,log10

import cv2
import scipy.ndimage
import imageio
 
import PIL
from PIL import Image,ImageFont,ImageDraw

'''
Modules and functions for building PDF reports out of images and/or features.
'''
    
# 1.  BRANCH reports  
def Report_features_MAIN(x,y,dx,SRf_o_y,SRf_o_yDoG,SRf_o_yDoGBn,SRf_o_yFinal,input_,prediction,BN_init,BN_fin,DOG_init,DOG_fin,taille:int,nombre_class:int,nom:str,save_rep:str,root_folder:str):
    
    '''
    PDF report for MAIN NN, displaying intermediate features multiple branch outputs for a given input PATCH
    '''
   
    offset=45
    width = 9*taille
    height = taille *7 +6*offset      
    output = Image.new("RGB", (width, height), color='white') 
    ecart=40

    draw = ImageDraw.Draw(output)   
    font = ImageFont.truetype(os.path.join(root_folder,"font_file.ttf"), 10) 
    
    # Display features 
    patch_x=20
    patch_y=40

    # Branch Outputs
    total_energie=0
    energie_contribution = []
    for channel in range(0,nombre_class):
        feature = np.squeeze(SRf_o_y[channel],axis=0)
        feature = np.squeeze(feature,axis=-1)
        energy_feature =  np.linalg.norm(feature,ord=1)
        total_energie+=energy_feature
        energie_contribution.append(energy_feature)

    for contrib in range(nombre_class):    
        patch_x = plot_1line ("y_output"+str(contrib)+"_Energie:"+str(energie_contribution[contrib])[0:3]+":"+str(energie_contribution[contrib]*100/total_energie)[0:4]+"%",rescale_255((SRf_o_y[contrib][0,:,:,0]*255.+np.mean(input_[0,:,:,0])))[x:x+dx, y:y+dx],draw,output,patch_x,patch_y,font,taille,ecart,offset,color=False)     
    patch_x=20
    patch_y+= offset +taille    
    
    
    p_affichage = [1., 1.1, 1.3, 1.7, 2, 4, 8, 10, 10]
    if DOG_fin:   
        # Branch & Dog Outputs
        total_energie=0
        energie_contribution = []
        for channel in range(0,nombre_class):
            feature = np.squeeze(SRf_o_yDoG[channel],axis=0)
            feature = np.squeeze(feature,axis=-1)
            energy_feature =  np.linalg.norm(feature,ord=1)
            total_energie+=energy_feature
            energie_contribution.append(energy_feature)
        for contrib in range(nombre_class):        
            patch_x = plot_1line ("y_output_Dog_"+str(contrib)+"_Energie:"+str(energie_contribution[contrib])[0:3]+":"+str(energie_contribution[contrib]*100/total_energie)[0:4]+"%",rescale_255(SRf_o_yDoG[contrib][0,:,:,0]*255.+np.mean(input_[0,:,:,0])*p_affichage[contrib])[x:x+dx, y:y+dx],draw,output,patch_x,patch_y,font,taille,ecart,offset,color=False)     
        patch_x=20
        patch_y+= 3*offset +taille 
    
    if BN_fin:
        # Branch & Dog & BN Outputs
        total_energie=0
        energie_contribution = []
        for channel in range(0,nombre_class):
            feature = np.squeeze(SRf_o_yDoGBn[channel],axis=0)
            feature = np.squeeze(feature,axis=-1)
            energy_feature =  np.linalg.norm(feature,ord=1)
            total_energie+=energy_feature
            energie_contribution.append(energy_feature)
        for contrib in range(nombre_class):        
            patch_x = plot_1line ("y_output_BN_"+str(contrib)+"_Energie:"+str(energie_contribution[contrib])[0:3]+":"+str(energie_contribution[contrib]*100/total_energie)[0:4]+"%",rescale_255(SRf_o_yDoGBn[contrib][0,:,:,0]*255.+np.mean(input_[0,:,:,0])*p_affichage[contrib])[x:x+dx, y:y+dx],draw,output,patch_x,patch_y,font,taille,ecart,offset,color=False)     
        patch_x=20
        patch_y+= 3*offset +taille 
        
    # Branch Outputs (final)
    total_energie=0
    energie_contribution = []
    for channel in range(0,nombre_class):
        feature = np.squeeze(SRf_o_yFinal[channel],axis=0)
        feature = np.squeeze(feature,axis=-1)
        energy_feature =  np.linalg.norm(feature,ord=1)
        total_energie+=energy_feature
        energie_contribution.append(energy_feature)
    for contrib in range(nombre_class):  
        patch_x = plot_1line ("y_output_FINAL"+str(contrib)+"_Energie:"+str(energie_contribution[contrib])[0:3]+":"+str(energie_contribution[contrib]*100/total_energie)[0:4]+"%",rescale_255((SRf_o_yFinal[contrib][0,:,:,0]*255.+np.mean(input_[0,:,:,0]))*p_affichage[contrib])[x:x+dx, y:y+dx],draw,output,patch_x,patch_y,font,taille,ecart,offset,color=False)     
    
    patch_x=20
    patch_y+= 3*offset +taille 
    patch_x = plot_1line ("bic+sum",rescale_255(prediction[0,:,:,0]*255.)[x:x+dx, y:y+dx],draw,output,patch_x,patch_y,font,taille,ecart,offset,color=False)    
    
    output.save(os.path.join(save_rep,str(nom.replace(".png","_Rapport_image_features"))+".pdf"),"pdf")   
    

def Patch_report_MAIN(prediction,INPUT,SRf_o_yFinal,
                      nombre_class:int,nom:str,save_rep:str,root_folder:str):
    """
    PDF report for MAIN NN, displaying different combinations of the multi-scale branches for a given input PATCH
    """
    offset=40
    width = 14*prediction.shape[1]
    height = prediction.shape[1] *12 +6*offset    
    output = Image.new("RGB", (width, height), color='white') 
    ecart=11

    draw = ImageDraw.Draw(output)   
    font = ImageFont.truetype(os.path.join(root_folder,"font_file.ttf"), 18) 
    
    # Display features 
    patch_x=20
    patch_y=40
 
    # Display different branch combinations
    centrage=int(nombre_class/2) 
    ai_lf_hf=[list(np.linspace(1/(ai**(centrage)),1,centrage))+list(np.linspace(ai,(ai**(nombre_class-centrage)),nombre_class-centrage)) for ai in [0.5,0.66,0.8,0.91,0.96,1,1.05,1.1,1.25,1.5,2]]
    amplification=[0.1, 0.25, 0.5, 0.75,0.83, 1,1.25,1.5,2,4, 10]
    taille=len(ai_lf_hf)
    profondeur = len(amplification)
    transformation=[[ai_lf_hf[num][a]*amplification[amp] for a in range(nombre_class)]  for amp in range(profondeur) for num in range(taille)]
   
    patch_x=50
    patch_y+=25
    offset = 15
    caption = "Transformations sur les histogrammes des "+str(nombre_class)+" sorties. Re correspond au rapport energie apportée / energie apportée par la prédiction originale. Si Re>1, la transformation apporte plus d'energie que la solution naturelle. "
    draw.text((patch_x, patch_y-35),caption,(0,0,0),font=font)
    patch_y+=15
    tensor_displayed,colored=Prepare_Display_Tensor(prediction)
    patch_x=plot_1line("prediction",(tensor_displayed).astype(np.uint8),draw,output,patch_x,patch_y,font,tensor_displayed.shape[0],ecart,offset,color=colored)           
    
    total_energie=0
    energie_contribution = []
    for channel in range(0,nombre_class):
        feature = np.squeeze(SRf_o_yFinal[channel],axis=0)
        feature = np.squeeze(feature,axis=-1)
        energy_feature =  np.linalg.norm(feature,ord=1)
        total_energie+=energy_feature
        
    compteur=1
    for transfo in range(len(transformation)):
        total_energie_t=0
        pred_new = np.zeros((tensor_displayed.shape[0],tensor_displayed.shape[0],3))
        pred_new_y = np.zeros((tensor_displayed.shape[0],tensor_displayed.shape[0]))
        for channel in range(nombre_class): 
            feature = np.squeeze(SRf_o_yFinal[channel],axis=0)
            feature = np.squeeze(feature,axis=-1)
            y_i = feature * transformation[transfo][channel]
            pred_new_y+=y_i
            total_energie_t+=np.linalg.norm(y_i,ord=1)
       
        pred_new_y+=INPUT

        pred_new[:,:,0]=pred_new_y
        pred_new[:,:,1]=prediction[:,:,2]
        pred_new[:,:,2]=prediction[:,:,1]
        pred_new_rgb=cv2.cvtColor((pred_new*255.).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
                    
        patch_x=plot_1line("Re:"+str(total_energie_t/total_energie)[0:3],pred_new_rgb,draw,output,patch_x,patch_y,font,tensor_displayed.shape[0],ecart,offset,color=True)           
        if (compteur)%taille==0:
            patch_y+=34+prediction.shape[1]
            patch_x=50+ prediction.shape[1]+2*ecart
        compteur+=1
        
    caption = " - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    draw.text((patch_x, patch_y-30),caption,(0,0,0),font=font)
    
            
    output.save(os.path.join(save_rep,str(nom.replace(".png",""))+"_Combinations_nn_"+".pdf"),"pdf")  
    
def Patch_report_features(list_tensor_input:list, list_tensor_intermediate_features:list, name_tensor_input:list,name_tensor_intermediate_features:list,mse_reseau:float,mse_bic:float,taille:int,nombre_class:int,root:str,border:int,nom:str,save_rep:str,ecart:int, numero_patch:int):
    """
    PDF report for BRANCH NN, displaying intermediate features for a specific input test patch.
    """
    
    # Preparing last features out of feature
    width = taille*10+200
    height = taille * 4       
    output = Image.new("RGB", (width, height), color='white') 
            
    # Figure,Police & Taille
    draw = ImageDraw.Draw(output)   
    font = ImageFont.truetype(os.path.join(root,"font_file.ttf"), 12)

    # Ecriture dans le PDF
    patch_x=50
    patch_y=20
    offset = 30
    # Afficahge psnr - modèle - nom
    caption = "Test Patch number  : "+str(nom)+" MSE Bic :"+str(mse_bic)+" MSE Réseau :"+str(mse_reseau)
    draw.text((patch_x, patch_y),caption,(0,0,0),font=font)
    patch_y+=80
    patch_x=50
    compteur=0

    # line 1
    caption = "Input & Output patches : "+str(nom)
    draw.text((patch_x, patch_y),caption,(0,0,0),font=font)
    patch_y+=30
    patch_x=50
    compteur=0
    
    for tensor in list_tensor_input: 
        if name_tensor_input[compteur]=="style_patch":
            tensor_displayed=tensor*255.
            colored=True
        else:
            tensor_displayed,colored=Prepare_Display_Tensor(tensor)
        patch_x=plot_1line(str(name_tensor_input[compteur]),(tensor_displayed).astype(np.uint8),draw,output,patch_x,patch_y,font,tensor.shape[0],ecart,offset,color=colored)           
        compteur+=1
            
    #line 2
    patch_y+=150+tensor.shape[0]
    patch_x=50
    compteur=0
    
    for tensor in list_tensor_intermediate_features:   
        tensor_displayed,colored=Prepare_Display_Residual(tensor)
        patch_x=plot_1line(str(name_tensor_intermediate_features[compteur]),(tensor_displayed).astype(np.uint8),draw,output,patch_x,patch_y,font,tensor.shape[0],ecart,offset,color=colored)           
        compteur+=1

    patch_y+=250
    patch_x=50        
    caption = " - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    draw.text((patch_x, patch_y-30),caption,(0,0,0),font=font)
    
            
    output.save(os.path.join(save_rep,str(nom.replace(".png",""))+"_MAIN_nn_"+".pdf"),"pdf")  

# 2.  BRANCH/MAIN reports  
def Benchmark_report(x:int,y:int,dx:int,list_tensor_input:list, list_tensor_intermediate_features:list, name_tensor_input:list,name_tensor_intermediate_features:list,type_branch:str,taille:int,nombre_class:int,root:str,border:int,save_rep:str,nom:str):
    """
    PDF report for BRANCH NN & MAIN NN, displaying chosen features for a specific input image
    """
    
    # Preparing last features out of feature
    ecart=20
    width = taille*9+200
    height = taille * 5      
    output = Image.new("RGB", (width, height), color='white') 
            
    # Figure,Police & Taille
    draw = ImageDraw.Draw(output)   
    font = ImageFont.truetype(os.path.join(root,"font_file.ttf"), 15)

    # Ecriture dans le PDF
    patch_x=50
    patch_y=20
    offset = 30
    # Afficahge psnr - modèle - nom
    caption = str(type_branch)
    draw.text((patch_x, patch_y),caption,(0,0,0),font=font)
    patch_y+=80
    patch_x=50
    compteur=0

    # line 1
    caption = "Input & Output Images : "
    draw.text((patch_x, patch_y),caption,(0,0,0),font=font)
    patch_y+=30
    patch_x=50
    compteur=0
    
    for tensor in list_tensor_input:   
        if name_tensor_input[compteur]=="style_patch":
            tensor_displayed=tensor*255.
            colored=True
        else:
            tensor_displayed,colored=Prepare_Display_Tensor(tensor)
        patch_x=plot_1line(str(name_tensor_input[compteur]),(tensor_displayed).astype(np.uint8),draw,output,patch_x,patch_y,font,tensor.shape[0],ecart,offset,color=colored)           
        compteur+=1
            
    #line 2
    patch_y+=taille+50
    patch_x=50
    compteur=0
    
    for tensor in list_tensor_intermediate_features:   
        tensor_displayed,colored=Prepare_Display_Residual(tensor)
        patch_x=plot_1line(str(name_tensor_intermediate_features[compteur]),(tensor_displayed).astype(np.uint8),draw,output,patch_x,patch_y,font,tensor.shape[0],ecart,offset,color=colored)           
        compteur+=1
    
    # ZOOM ---
    # line 1
    caption = "Zoom : "
    draw.text((patch_x, patch_y),caption,(0,0,0),font=font)
    patch_y+=30+taille
    patch_x=50
    compteur=0
    
    for tensor in list_tensor_input:   
        if name_tensor_input[compteur]!="style_patch":
            tensor_displayed,colored=Prepare_Display_Tensor(tensor)
            patch_x=plot_1line(str(name_tensor_input[compteur]),(tensor_displayed[x:x+dx,y:y+dx]).astype(np.uint8),draw,output,patch_x,patch_y,font,tensor.shape[0],ecart,offset,color=colored)           
            compteur+=1
            
    #line 2
    patch_y+=taille+50
    patch_x=50
    compteur=0
    
    for tensor in list_tensor_intermediate_features:   
        tensor_displayed,colored=Prepare_Display_Residual(tensor)
        patch_x=plot_1line(str(name_tensor_intermediate_features[compteur]),(tensor_displayed[x:x+dx,y:y+dx]).astype(np.uint8),draw,output,patch_x,patch_y,font,tensor.shape[0],ecart,offset,color=colored)           
        compteur+=1
        
    patch_y+=250
    patch_x=50        
    caption = " - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    draw.text((patch_x, patch_y-30),caption,(0,0,0),font=font)
    
    output.save(os.path.join(save_rep,"_Benchmark_"+str(nom)+"_"+str(type_branch)+".pdf"),"pdf")  
    

    


    

# BROUILLONS -------------------------

def Rapport_choisi_controle_benchmark(prediction_rgb_SR,prediction_rgb_ST,true_rgb,lr,bicubic_rgb,folder,
                                      list_mask_style,list_style,style_img_list, style_chosen_index,      
                           taille_input:int,taille:int,nom:str,save_rep:str,root_folder:str,ponderation_inter_styles:list,
                           taille_full_x:int,taille_full_y:int):
    """

    """
    offset=40
    rep_base_img = os.path.join(save_rep,nom)           
    ensure_dir(rep_base_img)
    rep_base_img_features_comp = os.path.join(rep_base_img,"features")  
    ensure_dir(rep_base_img_features_comp)
    rep=os.path.join(rep_base_img,folder)      
    ensure_dir(rep)
    imageio.imwrite(os.path.join(rep_base_img_features_comp,str("LR.png")),lr.astype(np.uint8))
    imageio.imwrite(os.path.join(rep_base_img_features_comp,str("ST.png")),prediction_rgb_ST)
    imageio.imwrite(os.path.join(rep_base_img_features_comp,str("SR.png")),prediction_rgb_SR)
    imageio.imwrite(os.path.join(rep_base_img_features_comp,str("true.png")),true_rgb)
    imageio.imwrite(os.path.join(rep_base_img_features_comp,str("Bic.png")),bicubic_rgb)

    width = taille*(18)
    height = taille*(30) 
    output = Image.new("RGB", (width, height), color='white') 
    ecart=11
    color=True
    draw = ImageDraw.Draw(output)   
    font2 = ImageFont.truetype(os.path.join(root_folder,"font_file.ttf"), 15) 
    font4 = ImageFont.truetype(os.path.join(root_folder,"font_file.ttf"), 21) 
    font7 = ImageFont.truetype(os.path.join(root_folder,"font_file.ttf"), 33) 
    
    # 1. MODELE ETUDE ---  DIFFERENTES FEATURES y_dog_lr y_dog_lr_BN  
    patch_x=20
    patch_y=40

    # RGB
    patch_x=plot_1line("Haute Résolution",true_rgb,draw,output,patch_x,patch_y,font7,2*taille,ecart,offset,color=color)
    bmm=np.zeros((prediction_rgb_SR.shape[0],prediction_rgb_SR.shape[1]))
    
    for sti in range(len(list_mask_style)):
        bm=np.zeros((prediction_rgb_SR.shape[0],prediction_rgb_SR.shape[1]))
        for ma in range(len(list_mask_style[sti])):
            bm+=list_mask_style[sti][ma]
        patch_x=plot_1line("Mask_style_"+str(list_style[sti]),bm*255.,draw,output,patch_x,patch_y,font4,1*taille,5*ecart,offset,color=False)
        imageio.imwrite(os.path.join(rep_base_img_features_comp,str(sti)+str("_mask.png")),(bm*255.).astype(np.uint8))
        bmm+=bm
        
    patch_x=20 + 2*taille+ ecart  
    patch_y+=1*taille +2*offset
    for sti in range(len(list_mask_style)):
        st_img= style_img_list[style_chosen_index[sti]].reshape(style_img_list[style_chosen_index[sti]].shape[0],style_img_list[style_chosen_index[sti]].shape[1],3)
        patch_x=plot_1line(str(ponderation_inter_styles[sti]),(st_img).astype(np.uint8),draw,output,patch_x,patch_y,font4,1*taille,5*ecart,offset,color=True)
    patch_x=20
    patch_y+=2*taille
    
    # MEDIAN FILTER  
    patch_x = plot_1line("Bicubic, ",bicubic_rgb,draw,output,patch_x,patch_y,font4,2*taille,ecart,offset,color=color)  
    patch_x=plot_1line("Bloc SR :",prediction_rgb_SR,draw,output,patch_x,patch_y,font4,2*taille,ecart,offset,color=color)   # prediction_rgb_SR_plot       #prediction_rgb_SR_filtrethf
    patch_x = plot_1line("Bloc SR + masque lissé et bruité de style ",prediction_rgb_ST,draw,output,patch_x,patch_y,font4,2*taille,ecart,offset,color=color)  
    patch_x=plot_1line("Haute Résolution",true_rgb,draw,output,patch_x,patch_y,font4,2*taille,ecart,offset,color=color)
    
    patch_x=20
    patch_y+=3*taille
    
    # Multiples zooms
    delta=170
    y_bis = [x for x in range(0,taille_full_x-delta-10,int((taille_full_x-delta-10)/6))]
    x_bis = [x for x in range(0,taille_full_y-delta-10,int((taille_full_y-delta-10)/6))]
    c=0
    for x in x_bis:
        for y in y_bis:
           patch_x = plot_1line("Bicubic, ",bicubic_rgb[x:x+delta,y:y+delta,:],draw,output,patch_x,patch_y,font2,taille,ecart,offset,color=color)  
           patch_x=plot_1line("Sr, :",prediction_rgb_SR[x:x+delta,y:y+delta,:],draw,output,patch_x,patch_y,font2,taille,ecart,offset,color=color)  #prediction_rgb_SR_filtrethf
           patch_x = plot_1line("Mask_Y, ",prediction_rgb_ST[x:x+delta,y:y+delta,:],draw,output,patch_x,patch_y,font2,taille,ecart,offset,color=color)  
           patch_x=plot_1line("Haute Résolution",true_rgb[x:x+delta,y:y+delta,:],draw,output,patch_x,patch_y,font2,taille,ecart,offset,color=color)
           patch_x+=10
           if c%2==0:
               patch_x=20
               patch_y+=taille+15
           c+=1
    output.save(os.path.join(rep,str("_Rapport_mask"+"_"+str(list_style)+"_"+str(ponderation_inter_styles))+".pdf"),"pdf")     

'''
def Rapport_choisi_specialpapier_controle_benchmark(prediction_rgb_SR,prediction_rgb_ST,true,bicubic_rgb,list_mask_style,list_style,style_img_list, style_chosen_index,  
                        y_bis:int,x_bis:int,delta:int,local:int,img:str,
                           taille_input:int,taille:int,nom:str,save_rep:str,root_folder:str,
                           taille_full_x:int,taille_full_y:int):
    """

    """
    font2 = ImageFont.truetype(os.path.join(root_folder,"font_file.ttf"), 15) 
    font4 = ImageFont.truetype(os.path.join(root_folder,"font_file.ttf"), 21) 
    font7 = ImageFont.truetype(os.path.join(root_folder,"font_file.ttf"), 33) 
    
    
    offset=0
    rep_base_img = os.path.join(save_rep,nom)           
    ensure_dir(rep_base_img)
    rep_base_img_features_comp = os.path.join(rep_base_img,"features")  
    ensure_dir(rep_base_img_features_comp)
    papier=os.path.join(rep_base_img,"papier")      
    ensure_dir(papier)
     
    # RAPPORT Complet ------------------------------------------------------------------------
    ecart=3
    width = taille*(5)+ecart 
    height = taille*(5)+ecart 
    output = Image.new("RGB", (width, height), color='white') 
    color=True
    # Figure,Police & Taille
    draw = ImageDraw.Draw(output)   
    # Ecriture dans le PDF
    patch_x=0
    patch_y=0   
    
    # HR 
    patch_x=plot_1line("",true.astype(np.uint8),draw,output,patch_x,patch_y,font7,3*taille,ecart,offset,color=color)
    patch_x+=ecart
    # Masks & styles - linspace for infering y points depending on masks number
    list_mask_style=list_mask_style[0:3]
    y_coord = np.linspace(0,int(2*taille),len(list_mask_style))
    
    for sti in range(len(list_mask_style)):
        bm=np.zeros((list_mask_style[0][0].shape[0],list_mask_style[0][0].shape[1]))
        for ma in range(len(list_mask_style[sti])):
            bm+=list_mask_style[sti][ma]#[x_bis[0]:x_bis[0]+delta,y_bis[0]:y_bis[0]+delta]
        patch_y=int(y_coord[sti])
        imageio.imwrite(os.path.join(rep_base_img,str("Mask_"+str(sti)+".png")),(bm*255.).astype(np.uint8))
        patch_y=plot_1line("",(bm*255.).astype(np.uint8),draw,output,patch_x,patch_y,font4,int(1*taille-ecart),ecart,offset,color=False)
    
    patch_x+= int(taille+ecart) 
    for sti in range(len(list_mask_style)):
        st_img= style_img_list[style_chosen_index[sti]].reshape(style_img_list[style_chosen_index[sti]].shape[0],style_img_list[style_chosen_index[sti]].shape[1],3)
        patch_y=int(y_coord[sti])
        patch_y=plot_1line("",(st_img).astype(np.uint8),draw,output,patch_x,patch_y,font4,int(1*taille-ecart),ecart,offset,color=True)
        imageio.imwrite(os.path.join(rep_base_img,str("Style_"+str(sti)+".png")),st_img.astype(np.uint8))
        
    patch_x=0
    patch_y=int(3*taille + ecart)
    
    patch_x=plot_1line("",bicubic_rgb[x_bis[0]:x_bis[0]+delta,y_bis[0]:y_bis[0]+delta,:].astype(np.uint8),draw,output,patch_x,patch_y,font7,int(1.5*taille-ecart),ecart*2,offset,color=color)
    patch_x=plot_1line("",prediction_rgb_SR[x_bis[0]:x_bis[0]+delta,y_bis[0]:y_bis[0]+delta,:].astype(np.uint8),draw,output,patch_x,patch_y,font7,int(1.5*taille-ecart),ecart,offset,color=color)
    patch_x+=int(0.25*taille)
    patch_x=plot_1line("",prediction_rgb_ST[x_bis[0]:x_bis[0]+delta,y_bis[0]:y_bis[0]+delta,:].astype(np.uint8),draw,output,patch_x,patch_y,font7,int(1.5*taille),ecart,offset,color=color)
    
    output.save(os.path.join(rep_base_img,str(img)+str("_Rapport_.pdf")),"pdf") 
    
    # Saving
    imageio.imwrite(os.path.join(rep_base_img,str("BIC.png")),bicubic_rgb[x_bis[0]+2*local:x_bis[0]+delta-2*local,y_bis[0]+2*local:y_bis[0]+delta-2*local,:].astype(np.uint8))
    imageio.imwrite(os.path.join(rep_base_img,str("SR.png")),prediction_rgb_SR[x_bis[0]+2*local:x_bis[0]+delta-2*local,y_bis[0]+2*local:y_bis[0]+delta-2*local,:].astype(np.uint8))
    imageio.imwrite(os.path.join(rep_base_img,str("ST.png")),prediction_rgb_ST[x_bis[0]+2*local:x_bis[0]+delta-2*local,y_bis[0]+2*local:y_bis[0]+delta-2*local,:].astype(np.uint8))
    imageio.imwrite(os.path.join(rep_base_img,str("HR.png")),true[x_bis[0]+2*local:x_bis[0]+delta-2*local,y_bis[0]+2*local:y_bis[0]+delta-2*local,:].astype(np.uint8))
'''


def Rapport_aleatoire_controle_benchmark(prediction,bicubic,true,bicubic_rgb, list_style_out, true_rgb,entree_rgb,outputs,list_mask_style,list_style,style_img_list, style_chosen_index,  all_styles,      
                           taille_input:int,taille:int,nombre_class:int,nom:str,save_rep:str,root_folder:str,ponderation_inter_styles:list,
                           taille_full_x:int,taille_full_y:int):


    offset=40
    rep_base_img = os.path.join(save_rep,nom)           # REP POUR TOUS LES RAPPORTS DE LIMAGE (CHOISI ET RANDOM)
    ensure_dir(rep_base_img)
    rep_base_img_features_comp = os.path.join(rep_base_img,"features")  #rep pour HR, STYLE, STYLE + BIC ...
    ensure_dir(rep_base_img_features_comp)
    random=os.path.join(rep_base_img,"random")      # rep des rapports random
    ensure_dir(random)
    
    # SAVING FEATURES COMPLETE - SR+Bic, HR, INPUT, Et enfin styles
    prediction_rgb_SR=np.zeros((prediction.shape[0],prediction.shape[1],3))
    prediction_rgb_SR[:,:,0]=prediction[:,:,0] # SR + bicubic
    prediction_rgb_SR[:,:,1]=prediction[:,:,2]
    prediction_rgb_SR[:,:,2]=prediction[:,:,1]
    prediction_rgb_SR_plot=cv2.cvtColor((prediction_rgb_SR*255.).astype(np.uint8), cv2.COLOR_YCrCb2RGB) #la fction npy YCBCR2RGB donne la mm chose
    
    imageio.imwrite(os.path.join(rep_base_img_features_comp,str("Sr&Bic.png")),prediction_rgb_SR_plot)
    imageio.imwrite(os.path.join(rep_base_img_features_comp,str("true.png")),true)
    imageio.imwrite(os.path.join(rep_base_img_features_comp,str("Bic.png")),bicubic_rgb)

    for st in range(len(all_styles)): 
        s=list_style_out[st]
        s=s.reshape(s.shape[1],s.shape[2],3)
        # RGB style&bic
        prediction_rgb_style=np.zeros((prediction.shape[0],prediction.shape[1],3))
        prediction_rgb_style[:,:,0]=np.clip(s[:,:,0] + bicubic[:,:,0],0,1) # Style + bicubic
        prediction_rgb_style[:,:,1]=np.clip(s[:,:,2],0,1)
        prediction_rgb_style[:,:,2]=np.clip(s[:,:,1],0,1)
        prediction_rgb_style=cv2.cvtColor((prediction_rgb_style*255.).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        # RGB style
        prediction_rgb_styleO=np.zeros((prediction.shape[0],prediction.shape[1],3))
        prediction_rgb_styleO[:,:,0]=s[:,:,0] # Style 
        prediction_rgb_styleO[:,:,1]=s[:,:,2]
        prediction_rgb_styleO[:,:,2]=s[:,:,1]
        prediction_rgb_styleO=cv2.cvtColor((prediction_rgb_styleO*255.).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        # RGB style
        prediction_rgb_SRstyle=np.zeros((prediction.shape[0],prediction.shape[1],3))
        prediction_rgb_SRstyle[:,:,0]=np.clip(s[:,:,0]+prediction[:,:,0],0,1) # BIC + SRi + Style
        prediction_rgb_SRstyle[:,:,1]=np.clip(s[:,:,2],0,1)
        prediction_rgb_SRstyle[:,:,2]=np.clip(s[:,:,1],0,1)
        prediction_rgb_SRstyle=cv2.cvtColor((prediction_rgb_SRstyle*255.).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    
        imageio.imwrite(os.path.join(rep_base_img_features_comp,str("Style+Bic_"+str(all_styles[st])+".png")),prediction_rgb_style)
        imageio.imwrite(os.path.join(rep_base_img_features_comp,str("Style_"+str(all_styles[st])+".png")),prediction_rgb_styleO)
        imageio.imwrite(os.path.join(rep_base_img_features_comp,str("Style+SR_"+str(all_styles[st])+".png")),prediction_rgb_SRstyle)
    
    nbre_rapport_random=10
    print("Rapports styles aléatoires")
    for rapp in range(nbre_rapport_random):
        #print("Rapp aléa "+str(rapp))
        width = taille*(18)
        height = taille*(30) 
        output = Image.new("RGB", (width, height), color='white') 
        ecart=11
        color=True
        # Figure,Police & Taille
        draw = ImageDraw.Draw(output)   
        font2 = ImageFont.truetype(os.path.join(root_folder,"font_file.ttf"), 15) 
        font4 = ImageFont.truetype(os.path.join(root_folder,"font_file.ttf"), 21) 
        font7 = ImageFont.truetype(os.path.join(root_folder,"font_file.ttf"), 33) 
    
        # 1. MODELE ETUDE ---  DIFFERENTES FEATURES y_dog_lr y_dog_lr_BN  
        patch_x=20
        patch_y=40
    
        # RGB
        patch_x=plot_1line("Haute Résolution",true_rgb,draw,output,patch_x,patch_y,font7,2*taille,ecart,offset,color=color)
        bmm=np.zeros((prediction.shape[0],prediction.shape[1]))
        
        random_style_index=np.random.randint(0,len(list_style_out),len(list_mask_style))
        ponderation_random_styles=np.random.randint(15,80,len(list_mask_style))/100  
        
        for sti in range(len(list_mask_style)):
            bm=np.zeros((prediction.shape[0],prediction.shape[1]))
            for ma in range(len(list_mask_style[sti])):
                bm+=list_mask_style[sti][ma]
            patch_x=plot_1line("Mask_style"+str(list_style[sti]),bm*255.,draw,output,patch_x,patch_y,font4,1*taille,5*ecart,offset,color=False)
            bmm+=bm

        patch_x=20 + 2*taille+ ecart  
        patch_y+=1*taille +2*offset
        for sti in range(len(list_mask_style)):
            st_img= style_img_list[random_style_index[sti]].reshape(style_img_list[random_style_index[sti]].shape[0],style_img_list[random_style_index[sti]].shape[1],3)
            patch_x=plot_1line(str(ponderation_random_styles[sti]),(st_img).astype(np.uint8),draw,output,patch_x,patch_y,font4,1*taille,5*ecart,offset,color=True)
        patch_x=20
        patch_y+=2*taille
        
        # OUTPUT Avec Mask -  Y
        outputY_rand=np.zeros((prediction.shape[0],prediction.shape[1],3))
        outputY_rand[:,:,0]=prediction[:,:,0]
        outputY_rand[:,:,1]=bicubic[:,:,2]
        outputY_rand[:,:,2]=bicubic[:,:,1] 

        for st in range(len(list_style)): 

            index = random_style_index[st]
            style_rand=list_style_out[index]
            style_rand=style_rand.reshape(style_rand.shape[1],style_rand.shape[2],3)
            style_y_rand=style_rand[:,:,0].reshape(style_rand.shape[0],style_rand.shape[1])
            
            
            for u in range(len(list_mask_style[st])):
                style_y_rand=style_y_rand*ponderation_random_styles[u]*list_mask_style[st][u]
                outputY_rand[:,:,0]+=style_y_rand
    
        outputY_rand[:,:,0]=np.clip(outputY_rand[:,:,0],0,1)
        outputY_rand=cv2.cvtColor((outputY_rand*255.).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        
        patch_x = plot_1line("Bicubic, ",bicubic_rgb,draw,output,patch_x,patch_y,font4,2*taille,ecart,offset,color=color)  
        patch_x=plot_1line("Bloc SR :",prediction_rgb_SR_plot,draw,output,patch_x,patch_y,font4,2*taille,ecart,offset,color=color)  #prediction_rgb_SR_filtrethf
        patch_x = plot_1line("Bloc SR + masque lissé et bruité de style ",outputY_rand,draw,output,patch_x,patch_y,font4,2*taille,ecart,offset,color=color)  
        patch_x=plot_1line("Haute Résolution",true_rgb,draw,output,patch_x,patch_y,font4,2*taille,ecart,offset,color=color)
        patch_x=20
        patch_y+=3*taille
        
        # Multiples zooms
        delta=170
        y_bis = [x for x in range(0,taille_full_x-delta-10,int((taille_full_x-delta-10)/6))]
        x_bis = [x for x in range(0,taille_full_y-delta-10,int((taille_full_y-delta-10)/6))]
        c=0
        for x in x_bis:
            for y in y_bis:
               patch_x = plot_1line("Bicubic, ",bicubic_rgb[x:x+delta,y:y+delta,:],draw,output,patch_x,patch_y,font2,taille,ecart,offset,color=color)  
               patch_x=plot_1line("Sr, :",prediction_rgb_SR_plot[x:x+delta,y:y+delta,:],draw,output,patch_x,patch_y,font2,taille,ecart,offset,color=color)
               patch_x = plot_1line("Mask_Y, ",outputY_rand[x:x+delta,y:y+delta,:],draw,output,patch_x,patch_y,font2,taille,ecart,offset,color=color)  
               patch_x=plot_1line("Haute Résolution",true_rgb[x:x+delta,y:y+delta,:],draw,output,patch_x,patch_y,font2,taille,ecart,offset,color=color)
               patch_x+=10
               if c%2==0:
                   patch_x=20
                   patch_y+=taille+15
               c+=1
        list_nom_random = [all_styles[i] for i in random_style_index]
        output.save(os.path.join(random,str("_Rapport_mask"+"_"+str(list_nom_random)+"_"+str(ponderation_random_styles))+".pdf"),"pdf") 

# COL VGG ZHANG 
def Rapport_benchmark_col(x,y,dx,prediction,true,entree,bins_f_T,T_list,BN_init,BN_fin,DOG_init,DOG_fin,
                           taille:int,nombre_class:int,nom:str,save_rep:str,root_folder:str):
    
    # --------------------------------------------------------------------------
    # --------------------RAPPORT 2 --- IMAGE COMPLETE & ZOOM ------------------
    # --------------------------------------------------------------------------
    offset=50
    width = (taille+offset)*10 + 750
    height = (taille+2*offset) * 6
    output = Image.new("RGB", (width, height), color='white') 
    ecart=11
    offset=40
    color=True# toujours en rgb pour ce rapport
    # Figure,Police & Taille
    draw = ImageDraw.Draw(output)   
    draw = ImageDraw.Draw(output)   
    font = ImageFont.truetype(os.path.join(root_folder,"font_file.ttf"), 12) 
    font7 = ImageFont.truetype(os.path.join(root_folder,"font_file.ttf"), 15) 
    font10 = ImageFont.truetype(os.path.join(root_folder,"font_file.ttf"), 16) 
    # n. lignes : MODELEs Comparatifs
    patch_x=20
    patch_y=40
    # MODELE ETUDE ---  SORTIES GLOBALES EN RGB
    #patch_x =plot_1line("Input",entree,draw,output,patch_x,patch_y,font,taille,ecart,offset,color=color) 
    patch_x+=20
    patch_x = plot_1line("Y, ",entree,draw,output,patch_x,patch_y,font,taille,ecart,offset,color=False)  
    patch_x=plot_1line(" ycbcr, :",true,draw,output,patch_x,patch_y,font,taille,ecart,offset,color=color)
    # Out Decoded
    patch_x=plot_1line("out",prediction,draw,output,patch_x,patch_y,font,taille,ecart,offset,color=color)
    for j in range(len(T_list)):
        patch_x=plot_1line("out_T_"+str(T_list[j]),bins_f_T[j],draw,output,patch_x,patch_y,font,taille,ecart,offset,color=color)
        
    patch_y+=taille + 2*offset
    # VisionZoom
    entree_crop=entree[x:x+dx, y:y+dx]
    true_crop=true[x:x+dx, y:y+dx]
    prediction_crop=prediction[x:x+dx, y:y+dx]
    
    caption = " Modèle étudié "
    draw.text((patch_x, patch_y-offset),caption,(0,0,0),font=font10)
    patch_x=2*ecart
    
    patch_x = plot_1line("Y, ",entree_crop,draw,output,patch_x,patch_y,font,taille*2,ecart,offset,color=False)  
    patch_x=plot_1line(" ycbcr, :",true_crop,draw,output,patch_x,patch_y,font,taille*2,ecart,offset,color=color)
    patch_x=plot_1line("out",prediction_crop,draw,output,patch_x,patch_y,font,taille*2,ecart,offset,color=color)
    patch_x=2*ecart
    patch_y+=taille*2 + 2*offset
    for j in range(len(T_list)):
        bins_f_T_crop=bins_f_T[j][x:x+dx, y:y+dx]
        patch_x=plot_1line("out_T_"+str(T_list[j]),bins_f_T_crop,draw,output,patch_x,patch_y,font,taille*2,ecart,offset,color=color)
        
    patch_y+=3*taille + offset
                        
    output.save(os.path.join(save_rep,str(nom.replace(".png","_Rapport_image"))+".pdf"),"pdf")      
    
    
    