from __future__ import division

import PIL
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pdf2image import convert_from_path
import os
from various_functions.numpy_data_fonctions import *

'''
File containing functions for building pdf reports
'''
    
# 1. Plotting functions for making pdf reports out of tensors

def Prepare_Display_Tensor(tensor):
    try:
        tensor_displayed = YCBCbCr2RGB(tensor*255.)
        colored=True
    except IndexError:#Gray image
        tensor_displayed=tensor*255.
        colored=False
    return(tensor_displayed,colored)

def Prepare_Display_Residual(tensor):
    tensor_displayed =  rescale_255(tensor) 
    colored=False
    return(tensor_displayed,colored)
    

def Save_Hist_2d_tf(h,folder_name,bins):
    hist_2d = h.numpy().reshape(bins,bins)
    plt.pcolor(hist_2d)
    plt.colorbar()
    plt.xlim([0,hist_2d.shape[1]])
    plt.ylim([0,hist_2d.shape[0]])
    plt.savefig(folder_name)
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close()

def Save_Hist_1d_tf(h,folder_name,bins):
    edges=np.array([i for i in range(bins*bins+1)])
    frq=h.cpu().numpy()
    fig, ax = plt.subplots()
    ax.bar(edges[:-1], frq, width=np.diff(edges), edgecolor="black", align="edge")
    plt.savefig(folder_name)
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close()
    
    
def plot_1line(caption,arr,draw,output,patch_x,patch_y,font,taille_output,ecart,offset,color=False):
    
    """
    Plot in Process_test_pdf an YCBCR image inside the report in RGB
    """
    caption = caption
    draw.text((patch_x, patch_y-offset),caption,(0,0,0),font=font)
    
    if color:
        new = Image.fromarray(arr)#,'RGB') 
    else:
        new = Image.fromarray(arr)  
    new=new.resize(((taille_output),(taille_output)),Image.ANTIALIAS)               
    size_x=new.size[0]
    size_y=new.size[1]
    output.paste(new, (patch_x, patch_y, patch_x + size_x, patch_y + size_y))
    patch_x += size_x + 2*ecart   
    return(patch_x)

def plot_1line_lab(caption,arr,draw,output,patch_x,patch_y,font,taille_output,ecart,offset,color=False):
    
    """
    Plot in Process_test_pdf an YCBCR image inside the report in RGB
    """
    caption = caption
    draw.text((patch_x, patch_y-offset),caption,(0,0,0),font=font)
    
    if color:
        new = Image.fromarray(arr,'LAB') 
    else:
        new = Image.fromarray(arr)  
    new=new.resize(((taille_output),(taille_output)),Image.ANTIALIAS)               
    size_x=new.size[0]
    size_y=new.size[1]
    output.paste(new, (patch_x, patch_y, patch_x + size_x, patch_y + size_y))
    patch_x += size_x + 2*ecart   
    return(patch_x)
    
def plot_1line_ycbcr(caption,arr,draw,output,patch_x,patch_y,font,taille_output,ecart,offset,color=False):
    
    """
    Plot in Process_test_pdf an YCBCR image inside the report in RGB
    """
    caption = caption
    draw.text((patch_x, patch_y-offset),caption,(0,0,0),font=font)
    
    if color:
        new = Image.fromarray(arr,'YCbCr') 
    else:
        new = Image.fromarray(arr)  
    new=new.resize(((taille_output),(taille_output)),Image.ANTIALIAS)               
    size_x=new.size[0]
    size_y=new.size[1]
    output.paste(new, (patch_x, patch_y, patch_x + size_x, patch_y + size_y))
    patch_x += size_x + 2*ecart   
    return(patch_x)
    
def plot_1line_pdf(caption,arr,draw,output,patch_x,patch_y,font,taille_output,ecart,offset,color=False):
    
    """
    Plot in Process_test_pdf an YCBCR image inside the report in RGB
    """
    caption = caption
    draw.text((patch_x, patch_y-offset),caption,(0,0,0),font=font)
    
    if color:
        new =  convert_from_path(arr)
    else:
        new = Image.fromarray(arr)
    new=new[0].resize(((taille_output),(taille_output)),Image.ANTIALIAS)               
    size_x=new.size[0]
    size_y=new.size[1]
    output.paste(new, (patch_x, patch_y, patch_x + size_x, patch_y + size_y))
    patch_x += size_x + 2*ecart   
    return(patch_x)
    
def plot_1line_y(caption,arr,draw,output,patch_x,patch_y,font,taille_output,ecart,offset):
    
    """
    Plot in Process_test_pdf an image inside the report
    """
    caption = caption
    draw.text((patch_x, patch_y-offset),caption,(0,0,0),font=font)
    new = Image.fromarray((arr))  
    new=new.resize(((taille_output),(taille_output)),Image.ANTIALIAS)               
    size_x=new.size[0]
    size_y=new.size[1]
    output.paste(new, (patch_x, patch_y, patch_x + size_x, patch_y + size_y))
    patch_y += size_y + 2*ecart   
    return(patch_y)


def histogram_contribution_channel(array,channel,result_glob,patch_x,patch_y,xlim1,xlim2):
    """
    Plot in Process_test_pdf a plt histogram inside the report as a PIL image
    """
    plt.figure(figsize=(5, 5))
    n, bins, patches = plt.hist(array[:,channel], 2500, density=True, facecolor='b')
    plt.xlabel('Contribution au delta psnr')
    plt.ylabel('Frequence')
    plt.title('Contribution' + str(channel) + (" Moyenne : ")+str(array[:,channel].mean())[0:4])
    plt.xlim(np.quantile(array[:,channel],0.10),np.quantile(array[:,channel],0.90))
    #plt.ylim(0,0.20)
    buffer = StringIO()
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    size_x=pil_image.size[0]
    size_y=pil_image.size[1]
    result_glob.paste(pil_image, (patch_x, patch_y, patch_x + size_x, patch_y + size_y))           
    patch_x+=85+size_x 
    return(patch_x,result_glob)

def plot_history(history,save_rep,rep):
    
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close()
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'PSNR' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'PSNR' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    # colors
    tr=["royalblue","midnightblue","darkblue","blue","mediumpurple","indigo","mediumorchid","violet","magenta","hotpink","royalblue","midnightblue","darkblue","blue","mediumpurple","indigo","mediumorchid","violet","magenta","hotpink","royalblue","midnightblue","darkblue","blue","mediumpurple","indigo","mediumorchid","violet","magenta","hotpink","royalblue","midnightblue","darkblue","blue","mediumpurple","indigo","mediumorchid","violet","magenta","hotpink","royalblue","midnightblue","darkblue","blue","mediumpurple","indigo","mediumorchid","violet","magenta","hotpink","royalblue","midnightblue","darkblue","blue","mediumpurple","indigo","mediumorchid","violet","magenta","hotpink"]
    va=["forestgreen","darkgreen","lime","springgreen","lightgreen","lawngreen","greenyellow","olivedrab","mediumspringgreen","green","forestgreen","darkgreen","lime","springgreen","lightgreen","lawngreen","greenyellow","olivedrab","mediumspringgreen","green","forestgreen","darkgreen","lime","springgreen","lightgreen","lawngreen","greenyellow","olivedrab","mediumspringgreen","green","forestgreen","darkgreen","lime","springgreen","lightgreen","lawngreen","greenyellow","olivedrab","mediumspringgreen","green","forestgreen","darkgreen","lime","springgreen","lightgreen","lawngreen","greenyellow","olivedrab","mediumspringgreen","green","forestgreen","darkgreen","lime","springgreen","lightgreen","lawngreen","greenyellow","olivedrab","mediumspringgreen","green"]
    ## Loss
    plt.figure(1, figsize=(13,10))
    c=0
    for l in loss_list:
        plt.plot(epochs, history.history[l], tr[c], label='Training loss '+str(l)+' (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        c+=1
    c=0
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], va[c], label='Validation loss '+str(l)+' (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        c+=1
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(os.path.join(save_rep,rep),"history_loss_"+".png"))
    
    ## Accuracy
    plt.figure(2,  figsize=(13,10))
    c=0
    for l in acc_list:
        plt.plot(epochs, history.history[l], tr[c], label='Training accuracy '+str(l)+'(' + str(format(history.history[l][-1],'.5f'))+')')
        c+=1
    c=0
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], va[c], label='Validation accuracy '+str(l)+'(' + str(format(history.history[l][-1],'.5f'))+')')
        c+=1

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(os.path.join(save_rep,rep),"history_acc_"+".png"))
    plt.show()
 