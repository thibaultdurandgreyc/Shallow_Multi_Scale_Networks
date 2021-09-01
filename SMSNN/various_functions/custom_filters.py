import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as fi
from various_functions.numpy_data_fonctions import *

# 0. Build 2D Gaussian blur on a given support size with chosen Standard deviation 
def gkern2(kernlen=6, nsig=2):
    """
    2D Gaussian kernel . Biblio  scipy.ndimage.filters as fi
    nsig : Standard deviation 
    """
    inp = np.zeros((kernlen, kernlen))
    inp[kernlen//2, kernlen//2] = 1
    t=fi.gaussian_filter(inp, nsig)
    
    return fi.gaussian_filter(inp, nsig)

# 1. DoG For St(y) (used for making highpass filter)
def two_Fi_filter(root_rep:str,variance1:float,variance2:float,save:bool):
    '''
    Prepare and Save A SINGLE 1D Dog Kernel (passband filter) (crop on a short support)
    variance 1 & 2 : Standard deviations
    '''
    t=11 
    g_kernel = prepare_2_gaussian_filters(variance1=variance1,variance2=variance2,taille=t) 
    
    if  save:
        save_gaussian_filters(g_kernel[2],nombre_class=1,root_save=root_rep,taille=t,name="gaussienne_style")
    return(g_kernel)   

def prepare_2_gaussian_filters(variance1,variance2,taille:int):
    '''
    Prepare and Save 1D Dog Kernel (passband filter)
    '''
    w=[] # list of kernels
    w.append((gkern2(np.int(taille),variance1)/sum(sum(gkern2(np.int(taille),variance1)))).reshape(np.int(taille),np.int(taille),1,1))
    w.append((gkern2(np.int(taille),variance2)/sum(sum(gkern2(np.int(taille),variance2)))).reshape(np.int(taille),np.int(taille),1,1))
   
    # 1D Gaussian & Normalization
    w_h,w_v=[],[]
    for i in range(2):
        taille=w[i].shape[0]
        noyau_h = w[i][int((taille)/2),:,:,:].reshape(np.int(taille),1,1,1)        
        noyau_v = w[i][:,int((taille)/2),:,:].reshape(1,np.int(taille),1,1)
        noyau_h_norm,noyau_v_norm=( noyau_h ) / sum(sum( noyau_h)) ,( noyau_v ) / sum(sum( noyau_v))         
        w_h.append( noyau_h_norm  )
        w_v.append(  noyau_v_norm )
    return(w_h,w_v,w)
    
# 2. DoG for MAIN NN (used for making multiple passband filters)
def Fi_filter(nombre_class:int,root_rep:str,FG_sigma_init:int,FG_sigma_puissance:int,save:bool):  
    '''
    Prepare and Save  MULTIPLE 1D Dog Kernels (passband filters)
    variance n and (n+1) defined with (FG_sigma_init*FG_sigma_puissance**n) and (FG_sigma_init*FG_sigma_puissance**n+1)
    '''
    t=30 # Not pair 
    
    liste = [0.01]+[FG_sigma_init*(FG_sigma_puissance**i) for i in range(nombre_class)]  # 0.01 kernel is an identity kernel
    w = np.flip(np.array(liste))
    g_kernel = prepare_gaussian_filters(variance=w,taille=t,nombre_class=nombre_class) 
    if nombre_class!=1 and save:
        save_gaussian_filters(g_kernel[2],nombre_class=nombre_class,root_save=root_rep,taille=t,name="gaussienne")
    return(g_kernel)
    
def prepare_gaussian_filters(variance,taille:int,nombre_class:int):
    '''
    Prepare and Save  MULTIPLE 1D Dog Kernels (passband filters) (crop on a short support)
    variance n and (n+1) defined with (FG_sigma_init*FG_sigma_puissance**n) and (FG_sigma_init*FG_sigma_puissance**n+1)
    '''
    w=[] # list of kernels
    w_not_cropped=[] # list of uncropped kernel 
    
    for i in range(nombre_class+1):
        n=gkern2(np.int(taille),(variance[i]))
        w_not_cropped.append(n/sum(sum(n)))
        
    for i in range(nombre_class):
        n=gkern2(np.int(taille),(variance[i]))
        noyau = (n/sum(sum(n)))
        
        # Minimal support
        bordure=0
        val_lim=0.00001
        val = noyau[(int(taille/2)+1),bordure] / noyau.max()
        while val<val_lim and bordure<taille-2:
            bordure+=2
            val = noyau[(int(taille/2)+1),bordure] / noyau.max()
        
        noyau=crop_center(noyau, noyau.shape[0]-bordure, noyau.shape[1]-bordure)           
        w.append(noyau.reshape(np.int(taille-bordure),np.int(taille-bordure),1,1))
                
        noyau_next = (gkern2(np.int(taille-bordure),(variance[i+1]))/sum(sum(gkern2(np.int(taille-bordure),(variance[i+1])))))
        w.append(noyau_next.reshape(np.int(taille-bordure),np.int(taille-bordure),1,1))
      
    # Creating 1D Gaussian kernels
    w_h,w_v=[],[]
    for i in range(len(w)):
        taille=w[i].shape[0]
        if taille!=1:
            noyau_h = w[i][int((taille+1)/2),:,:,:].reshape(np.int(taille),1,1,1) 
            noyau_v = w[i][:,int((taille+1)/2),:,:].reshape(1,np.int(taille),1,1)
            if sum(noyau_h)==0:
                noyau_h=np.zeros((np.int(taille),1,1,1))
                noyau_h[int((taille)/2),:,:,:]=1
                noyau_v=np.zeros((1,np.int(taille),1,1))
                noyau_v[:,int((taille)/2),:,:]=1                
        else:
            noyau_h = w[i][int((taille)/2),:,:,:].reshape(np.int(taille),1,1,1)        
            noyau_v = w[i][:,int((taille)/2),:,:].reshape(1,np.int(taille),1,1)
        noyau_h_norm=( noyau_h ) / sum(sum( noyau_h))
        noyau_v_norm= (   noyau_v  ) / sum(sum( noyau_v )) 
        w_h.append( noyau_h_norm   )
        w_v.append(  noyau_v_norm )
    return(w_h,w_v,w_not_cropped)

def save_gaussian_filters(w,nombre_class:int,root_save:str,taille:int,name:str):
    '''
    Save Gaussian kernels into .png file when ready to use.
    '''
    # Flous Gaussiens 
    fig, axs = plt.subplots(2,(nombre_class//2)+1, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    custom_ylim = (0,1)
    axs = axs.ravel()
    
    for i in range(nombre_class+1):
        gauss=w[i]
        gauss=gauss-gauss.min()
        gauss=gauss/gauss.max()
        im=axs[i].imshow(gauss.reshape(taille,taille),vmin=0,vmax=1, cmap='gray')
        
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)  
    plt.savefig(os.path.join(root_save,name+"noyaux_gaussiens.png"))
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close()
    # Gaussian Blurr FFT FFT
    fig, axs = plt.subplots(2,(nombre_class//2)+1, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    custom_ylim = (-0.5, 0.5)
    axs = axs.ravel()
    
    for i in range(nombre_class+1):
        fft_test=np.fft.fft2(w[i].reshape(taille,taille))
        fft_test_shift=np.fft.fftshift(fft_test)
        im=axs[i].imshow((np.abs(fft_test_shift)).reshape(taille,taille),vmin=0,vmax=1, cmap='gray')
        
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax) 
    plt.savefig(os.path.join(root_save,name+"_noyaux_gaussiens_FFT.png"))
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close()
     # DOG
    fig, axs = plt.subplots(2,(nombre_class//2)+1, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    custom_ylim = (-0.5, 0.5)
    axs = axs.ravel()

    for i in range(nombre_class):
        dog=(w[i+1].reshape(taille,taille)-w[i].reshape(taille,taille))
        #print(dog.max(),"dog max mean")
        dog=dog-dog.min()
        dog=dog/dog.max()
        im=axs[i].imshow(dog,vmin=0,vmax=1, cmap='gray')
        
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)  
    
    plt.savefig(os.path.join(root_save,name+"DoG.png"))
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close()
    # DOG FFT
    fig, axs = plt.subplots(2,(nombre_class//2)+1, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    custom_ylim = (0,1)
    axs = axs.ravel()

