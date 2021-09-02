import os
import json
import random
import tensorflow as tf
from datetime import datetime   
import time  
import imageio 

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.applications.vgg19 import VGG19, preprocess_input

import tensorflow_addons as tfa
import tensorflow_probability as tfp

from Networks.MAIN.architecture_MAIN import *
from Networks.STr_3.architecture_Str_3 import *
from Networks.STr_col.architecture_Str_col import *
from Networks.STr_y.architecture_Str_y import *
from Networks.Generator_Constructor import *
from Networks.Compilation_NN import *
from Networks.Loss_Constructor import *
from Networks.model_management import *

from Testing.Test import *
from Testing.PDF_reports import *
from various_functions.tensor_tf_data_fonctions import *
from various_functions.rapports_pdf_fonctions import *
from various_functions.numpy_data_fonctions import *
from various_functions.directory_management import *
from various_functions.custom_filters import *
from tensorboard import *

from codes_externes.LearningRate import *


# 1. CallBacks & Learning Processes (MAIN & BRANCH)    
class Callback_test_rapport_BRANCH(tf.keras.callbacks.Callback):
    '''
    Process Inference on test patch database at the end of each epoch ; for branch NN
    '''
    def __init__(self, main_network,root_rep,taille,root,benchmark_folder,save_rep_energie,BN_init, BN_fin, DOG_init, DOG_fin,nombre_class,type_branch,sigma_noise_blur,border,nombre_patch_test, style_patch , clusters,bins): #stcol
        super(Callback_test_rapport_BRANCH, self).__init__()
        self.root_rep = root_rep
        self.taille = taille
        self.root = root
        self.main_network=main_network
        self.save_rep_energie=save_rep_energie
        self.BN_init = BN_init
        self.BN_fin = BN_fin
        self.DOG_init = DOG_init
        self.DOG_fin = DOG_fin
        self.style_patch=style_patch
        self.nombre_class = nombre_class
        self.border = border
        self.nombre_patch_test = nombre_patch_test
        self.benchmark_folder=benchmark_folder
        self.clusters=clusters
        self.bins=bins
        self.type_branch=type_branch
        self.sigma_noise_blur=sigma_noise_blur

    def on_epoch_end(self, epoch, logs=None):
        print ( "Procédure de Test - Contrôle sur la sortie - epoch "+str(epoch) )
        print(os.path.join(self.root_rep,os.path.join("Model_saved_epochs","_saved-model-{epoch:d}.h5")))
        model_identite=tf.keras.models.load_model(os.path.join(self.root_rep,os.path.join("Model_saved_epochs","_saved-model-"+str(epoch+1)+".h5")),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False)                 
        process_test_controle_BRANCH(model_identite,main_network=self.main_network,root=self.root,save_rep=(os.path.join(self.root_rep,"rapports_controle_autre_epochs/visual_reports"))+str(epoch),taille=self.taille,nombre_class=self.nombre_class,border=self.border,save_rep_energie=self.save_rep_energie,BN_init=self.BN_init, BN_fin=self.BN_fin, DOG_fin=self.DOG_fin, DOG_init=self.DOG_init, sigma_noise_blur=self.sigma_noise_blur,nombre_patch_test=self.nombre_patch_test,type_branch=self.type_branch,style_patch=self.style_patch,clusters=self.clusters,bins=self.bins) 
        
        
class LossHistory(tf.keras.callbacks.Callback):
    '''
    Plots and storages loss & Accuracy data at the end of each epoch
    '''
    def __init__(self,save_rep):
        super(LossHistory, self).__init__()
        self.save_rep = save_rep
    def on_train_begin(self, logs={}):        
        self.base_metrics = [metric for metric in self.params['metrics'] if not metric.startswith('val_')]
        self.loss_list=[[] for metric in self.params['metrics'] if not metric.startswith('val_')]
        self.logs = []
    def on_batch_end(self, batch, logs={}):
        for metric_id, metric in enumerate(self.base_metrics):
            self.loss_list[metric_id].append(logs.get(metric))

    def on_epoch_end(self, epoch, logs={}):
        plt.figure(2,  figsize=(13,10))
       
        for metric_id, metric in enumerate(self.base_metrics):
            plt.cla()   # Clear axis
            plt.clf()   # Clear figure
            plt.close()
            plt.plot(self.loss_list[metric_id], 'midnightblue', label='loss')
            plt.title('Loss')
            plt.xlabel('Itérations')
            plt.ylabel(str(metric))
            plt.savefig(os.path.join(os.path.join(self.save_rep),str(metric)+".png"))

def learning_MAIN_network(main_network:str,taille:int, border:int,R:int, filtres_sr:int,nombre_kernel:int,nombre_class:int,ponderation_features:list, BN_init:bool, BN_fin:bool, DOG_init:bool,DOG_fin:bool,loss_pixel:bool, loss_perceptuelle:list, loss_style:list, ponderation_perc:float, ponderation_style:float,  ponderation_pixel:float,learning_rate:int, epochs:int, step_per_epoc:int,batch_size:int,nom_par_epoch:str,nbre_validation_data:int, save_rep:str,root:str, log_dir:str,sigma_noise_blur:float,w_h,w_v):
    """
    Training Procedure for SR NN with callbacks and loss evolution saving. Saves training & model parameters. Same architecture and learning schedule for DENOISING or SR.
    """  
    # Generators Training data et Validation data
    trainDatagen = Generator_patch(main_network=main_network,data_role="train", size_type_input=(taille+2*border,taille+2*border) , size_type_output=(taille,taille), taille=taille,loss_pixel=loss_pixel, loss_perceptuelle=loss_perceptuelle,loss_style=loss_style, root=root,batch_size=batch_size, border=border, nombre_class=nombre_class,sigma_noise_blur=sigma_noise_blur)    
    validDatagen=Generator_patch(main_network=main_network,data_role="validation", size_type_input=(taille+2*border,taille+2*border) , size_type_output=(taille,taille), taille=taille, loss_pixel=loss_pixel, loss_perceptuelle=loss_perceptuelle,loss_style=loss_style,nombre_class=nombre_class, batch_size=batch_size,root=root, border=border,sigma_noise_blur=sigma_noise_blur,)  
    # SR NN
    entrainement = MAIN_network(sizex=taille,sizey=taille,ponderation_features=ponderation_features,nombre_class=nombre_class,filtres=filtres_sr,kernel=nombre_kernel,w_h=w_h , w_v=w_v, loss_pixel=loss_pixel, loss_perceptuelle=loss_perceptuelle,loss_style=loss_style,ponderation_pixel=1, ponderation_perc=ponderation_perc, ponderation_style=ponderation_style,learning_rate=learning_rate, BN_init=BN_init, BN_fin=BN_fin, DOG_fin=DOG_fin,DOG_init=DOG_init)    
    model=entrainement[0]

    # Parameters & Training folder
    rep_info = os.path.join(save_rep,"_info_model")
    ensure_dir(rep_info)
    
    # Saving model Summary
    with open((os.path.join(rep_info,'_summary_model.json')), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    # Saving Optim Name & Loss Functions per branch
    with open(os.path.join(rep_info,"_Loss_Optim_Names.txt"), "w") as text_file:
            text_file.write("List loss utilisées par branche :  "+ str(entrainement[1])+"\n")
            text_file.write("Optimizer utilisé : "+ str(entrainement[2])+"\n")
            text_file.write("features loss percpetuelles : "+ str(loss_perceptuelle)+"\n")
            text_file.write("features loss style : "+ str(loss_style)+"\n")

    # Callbacks
    filepath = nom_par_epoch
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1)
    #change_lr = LearningRateScheduler(scheduler)
    learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_loss',patience=5,verbose=5,factor=0.5,cooldown=0,min_delta=0.000001) 
    
    # Training
    h=model.fit_generator(trainDatagen, steps_per_epoch=step_per_epoc, validation_data=validDatagen, validation_steps=nbre_validation_data//batch_size,epochs=epochs,callbacks=[EarlyStopping(patience=14, verbose=1,monitor='val_loss'), checkpoint, learning_rate_reduction])
    plot_history(h,save_rep,"learning_data")

    # Saving learning history (loss & Acc) as JSON
    with open((os.path.join(save_rep,'history_learning.json')), 'w') as f:
        json.dump(str(h.history), f)
    hist_df = pd.DataFrame(h.history) 
    hist_csv_file = 'History_Model.csv'
    with open(os.path.join(save_rep,hist_csv_file), mode='w') as f:
        hist_df.to_csv(f)
    
    # Saving model 
    model.save(os.path.join(save_rep,"model.h5"))
    model.save(os.path.join(os.path.join(save_rep,"final_saved"),"model_MAIN.h5"))
    return(model)      

def learning_BRANCH_network(model_MAIN, main_network:str,type_branch:str,  root_rep:str ,root_folder:str,  taille:int, filtres_branch:int, border:int,loss_pixel:bool, loss_perceptuelle:list, loss_style:list, ponderation_pixel:float, ponderation_perc:float, ponderation_style:float,   DOG_init:bool, DOG_fin: bool, filtres_sr:int,nombre_kernel:int, learning_rate:int,epochs:int, step_per_epoc:int,batch_size:int,save_rep:str,root:str, benchmark_folder:str,nombre_class:int,       log_dir:str,nom_par_epoch:str,nbre_validation_data:int,nombre_patch_test:int,w_h_s,w_v_s,clusters:int,hist_style,bins:int,sigma_noise_blur:float,BN_init,  BN_fin, style_gram, style_img="",patch_style=0,name_specific_folder="",cbcr_sty=False):          
    """
    Training Procedure for BRANCH LEARNING (on the top of the trained MAIN 'model').
    """   
    # Set MAIN network layers not trainable (i.e freeze parameters)
    for layer in model_MAIN.layers:
        layer.trainable = False

    # Generators Training data et Validation data
    trainDatagen = Generator_patch(main_network=main_network,data_role="train", size_type_input=(taille+2*border,taille+2*border) , size_type_output=(taille,taille), taille=taille,loss_pixel=loss_pixel, loss_perceptuelle=loss_perceptuelle,loss_style=loss_style, root=root,batch_size=batch_size, border=border, nombre_class=nombre_class,sigma_noise_blur=sigma_noise_blur)    
    validDatagen=Generator_patch(main_network=main_network,data_role="validation", size_type_input=(taille+2*border,taille+2*border) , size_type_output=(taille,taille), taille=taille, loss_pixel=loss_pixel, loss_perceptuelle=loss_perceptuelle,loss_style=loss_style,nombre_class=nombre_class, batch_size=batch_size,root=root, border=border,sigma_noise_blur=sigma_noise_blur)  
    
    # Callbacks
    filepath,filepath2 = nom_par_epoch, os.path.join(save_rep+"/Model_saved_epochs/model.h5")
    checkpoint , checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1), ModelCheckpoint(filepath, monitor='val_loss', verbose=1)
    learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_loss',patience=3,verbose=1,factor=0.2,min_lr= 0.00001,cooldown=1,min_delta=0.000001) 
    die="/data/chercheurs/durand192/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=die, histogram_freq=0, write_graph=True, write_images=True)
    loss_instance = LossHistory(root_rep)
    
    # Graph Definition & Training
    if type_branch=="Stycbcr" : # St-y & St-y,cb,cr
        if cbcr_sty:
            new_model,loss_list = STy_branch_col(model_MAIN,main_network,taille,taille, filtres_branch, border, DOG_init, DOG_fin, BN_init,False,nombre_kernel,w_h_s,w_v_s,style_gram, loss_pixel, loss_perceptuelle, loss_style, ponderation_pixel, ponderation_perc, ponderation_style,learning_rate)
        else:
            new_model,loss_list = STy_branch(model_MAIN,main_network,taille,taille,filtres_branch,border, DOG_init, DOG_fin, BN_init,False,nombre_kernel,w_h_s,w_v_s,style_gram, loss_pixel, loss_perceptuelle, loss_style, ponderation_pixel, ponderation_perc, ponderation_style, learning_rate)

        bins,clusters=0,0
            
    elif type_branch=="Stcol" : # St-col
        new_model,loss_list = STrcol_branch(model_MAIN,main_network,taille,taille, filtres_branch, border, False, False, BN_init,BN_fin,nombre_kernel,clusters,w_h_s,w_v_s,hist_style,style_img, loss_pixel, loss_perceptuelle, loss_style, ponderation_pixel, ponderation_perc, ponderation_style, learning_rate)
                    
    elif type_branch=="St3":
        bins,clusters=0,0
        new_model,loss_list = ST3_branch(model_MAIN,main_network,taille,taille, filtres_branch, border, False, False, BN_init,BN_fin,nombre_kernel,w_h_s,w_v_s,style_gram, loss_pixel, loss_perceptuelle, loss_style, ponderation_pixel, ponderation_perc, ponderation_style, learning_rate)
    
    h=new_model.fit_generator(trainDatagen, steps_per_epoch=step_per_epoc, validation_data=validDatagen, validation_steps=nbre_validation_data//batch_size,epochs=epochs,callbacks=[EarlyStopping(patience=5, verbose=1,monitor='val_loss'), checkpoint, checkpoint2, Callback_test_rapport_BRANCH(main_network=main_network,root_rep=root_rep,taille=taille,root=root_folder, save_rep_energie=os.path.join(root_rep,"energie/"),BN_init=BN_init, BN_fin=BN_fin, DOG_init=DOG_init, DOG_fin=DOG_fin, benchmark_folder=benchmark_folder,nombre_class=nombre_class,border=border,nombre_patch_test=nombre_patch_test,style_patch=patch_style,bins=bins,clusters=clusters,type_branch=type_branch,sigma_noise_blur=sigma_noise_blur),loss_instance])
    plot_history(h,save_rep,"learning_data")
    
    # Saving model Summary
    with open((os.path.join(save_rep,'summary_model.json')), 'w') as f:
        new_model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Saving Optim Name & Loss Functions per branch
    with open(os.path.join(save_rep,"_Loss_Optim_Names.txt"), "w") as text_file:
            text_file.write("List loss utilisées par branche :  "+ str(loss_list)+"\n")
            text_file.write("features loss percpetuelles : "+ str(loss_perceptuelle)+"\n")
            text_file.write("features loss style : "+ str(loss_style)+"\n")

    # Saving learning history (loss & Acc)
    with open((os.path.join(os.path.join(save_rep,"learning_data"),'history_learning.json')), 'w') as f:
        json.dump(str(h.history), f)
    hist_df = pd.DataFrame(h.history) 
    hist_csv_file = 'History_Model.csv'
    with open(os.path.join(os.path.join(save_rep,"learning_data"),hist_csv_file), mode='w') as f:
        hist_df.to_csv(f)
    
    # Saving whole model(MAIN + BRANCH)
    new_model.save(os.path.join(save_rep,"model.h5"))
    
    # Saving model in final save folder
    save_rep=save_rep.replace(name_specific_folder,"")
    if type_branch=="Stycbcr" : # St-y & St-y,cb,cr
        if cbcr_sty:
            new_model_residual_branch = STy_residual_branch_col_none(filtres_branch=filtres_branch, border=border, DOG_init=DOG_init, DOG_fin=DOG_fin, BN_init=BN_init,BN_fin=BN_fin,nombre_kernel=nombre_kernel,w_h_s=w_h_s,w_v_s=w_v_s)
            new_model_residual_branch =copy_parameters(model=new_model_residual_branch,model_patch=new_model)
            new_model_residual_branch.save(os.path.join(os.path.join(save_rep,"final_saved"),"model_St(y)_col_"+str(style_img)+".h5"))  
        else:
            new_model_residual_branch = STy_residual_branch_none(filtres_branch=filtres_branch, border=border, DOG_init=DOG_init, DOG_fin=DOG_fin, BN_init=BN_init,BN_fin=BN_fin,nombre_kernel=nombre_kernel,w_h_s=w_h_s,w_v_s=w_v_s)
            new_model_residual_branch =copy_parameters(model=new_model_residual_branch,model_patch=new_model)
            new_model_residual_branch.save(os.path.join(os.path.join(save_rep,"final_saved"),"model_St(y)_"+str(style_img)+".h5"))        
            
    elif type_branch=="Stcol" : # St-col
        new_model_residual_branch = STrcol_residual_branch_none(filtres_branch=filtres_branch, border=border, DOG_init=DOG_init, DOG_fin=DOG_fin, BN_init=BN_init,BN_fin=BN_fin,nombre_kernel=nombre_kernel,w_h_s=w_h_s,w_v_s=w_v_s)
        new_model_residual_branch =copy_parameters(model=new_model_residual_branch,model_patch=new_model)
        new_model_residual_branch.save(os.path.join(os.path.join(save_rep,"final_saved"),"model_ST(col)_"+str(style_img)+".h5"))  
                
    elif type_branch=="St3": # st3 is not residual, but here we compute the output from BICUBIC, and not from the output of MAIN NN
        new_model_residual_branch = ST3_residual_branch_none(filtres_branch=filtres_branch, border=border, DOG_init=DOG_init, DOG_fin=DOG_fin, BN_init=BN_init,BN_fin=BN_fin,nombre_kernel=nombre_kernel,w_h_s=w_h_s,w_v_s=w_v_s)
        #new_model_residual_branch=extract_layer(new_model_residual_branch,"ST_ycbcr") # we want the ycbcr output ; not rgb (NOT RESIDUAL ON YCBCR)
        new_model_residual_branch =copy_parameters(model=new_model_residual_branch,model_patch=new_model)
        new_model_residual_branch.save(os.path.join(os.path.join(save_rep,"final_saved"),"model_ST3_"+str(style_img)+".h5"))


# 2. Testing pocedures main functions (MAIN & BRANCH)    
def procedure_test_MAIN(main_network:str,root_rep:str, ponderation_features:list, border:int,root_folder:str,benchmark_folder:str,taille:int,filtres:int,ouverture:int,kernel_sr:int,kernel:int,nombre_class:int,nombre_patch_test:int,FG_sigma_init:float,FG_sigma_puissance:float,BN_init:bool,BN_fin:bool,DOG_init:bool,DOG_fin:bool,nom_rep:str,sigma_noise_blur:float,w_h,w_v,bool_inference_benchmark:bool,bool_calcul_energie:bool,bool_process_test:bool):
    '''
    Testing procedure on the trained model. 4 steps : Statistical computations, Inference on grayscale images, energy statistics computation & Qualitative tests on patches
    '''
    model_identite=tf.keras.models.load_model(os.path.join(root_rep,"model.h5"),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False)     

    statistique_branch_MAIN(model=model_identite,nombre_class=nombre_class,DOG_init=DOG_init,DOG_fin=DOG_fin,BN_init=BN_init,BN_fin=BN_fin,root_rep=root_rep)
    
    print ( "Main NN Test Procedure - Inférence on Benchmark - "+str(bool_inference_benchmark))
    if bool_inference_benchmark:    
        Image_benchmark_MAIN(model_identite=model_identite,main_network=main_network,ponderation_features=ponderation_features,border=border,FG_sigma_init=FG_sigma_init,FG_sigma_puissance=FG_sigma_puissance,BN_init=BN_init, BN_fin=BN_fin,  DOG_init=DOG_init, DOG_fin=DOG_fin,sigma_noise_blur=sigma_noise_blur, benchmark_folder=benchmark_folder,save_rep=root_rep,root_folder=root_folder, nombre_class=nombre_class, filtres=filtres,  nombre_kernel=kernel, w_h=w_h,w_v=w_v)     
     
    print ( "Main NN Test Procedure - Energy & FFt Computation - "+str(bool_calcul_energie))
    if bool_calcul_energie:   
        model_identite=tf.keras.models.load_model(os.path.join(root_rep,"model.h5"),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False)
        Energy(model_identite,main_network,sigma_noise_blur=sigma_noise_blur,taille=taille,root=root_folder,save_rep=os.path.join(root_rep,"energie/"),BN_init=BN_init, BN_fin=BN_fin, DOG_init=DOG_init, DOG_fin=DOG_fin, nombre_class=nombre_class,border=border,nombre_patch_test=nombre_patch_test)

    print ( "Main NN Test Procedure - Individual Reports  - "+str(bool_process_test))
    if bool_process_test:    
        w=process_test_controle_MAIN(model_identite,main_network,sigma_noise_blur=sigma_noise_blur,taille=taille,root=root_folder,save_rep=os.path.join(root_rep,"visual_reports/"),save_rep_energie=os.path.join(root_rep,"energie/"),BN_init=BN_init,  BN_fin=BN_fin, DOG_init=DOG_init, DOG_fin=DOG_fin, nombre_class=nombre_class, border=border,nombre_patch_test=nombre_patch_test)


def procedure_test_BRANCH(main_network:str,root_rep:str, ponderation_features:list, border:int,root_folder:str,benchmark_folder:str,taille:int,filtres:int,filtres_branch:int,ouverture:int,kernel:int,kernel_branch:int,nombre_class:int,nombre_patch_test:int,FG_sigma_init:float,FG_sigma_puissance:float,BN_init:bool,BN_fin:bool,DOG_init:bool,DOG_fin:bool,nom_rep:str,sigma_noise_blur:float,w_h,w_v,w_h_s,w_v_s,bool_inference_benchmark:bool,bool_process_test:bool,type_branch:str,cbcr_sty:bool,style_patch,clusters,bins):
    '''
    Testing procedure on the BRANCH learned on the top of the MAIN MODEL. 3 steps : Statistical computations, Inference on grayscale images & Qualitative tests on patches
    '''
    # Other NN
    model_identite=tf.keras.models.load_model(os.path.join(root_rep,"model.h5"),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False)
    if type_branch=="St ycbcr":
        statistique_branch_Sty(model=model_identite,nombre_class=nombre_class,DOG_init=DOG_init,DOG_fin=DOG_fin,BN_init=BN_init,BN_fin=BN_fin,root_rep=root_rep)
    if type_branch=="Stcol":
        statistique_branch_Stcol(model=model_identite,nombre_class=nombre_class,DOG_init=DOG_init,DOG_fin=DOG_fin,BN_init=BN_init,BN_fin=BN_fin,root_rep=root_rep)
    if type_branch=="St3":
        statistique_branch_St3(model=model_identite,nombre_class=nombre_class,DOG_init=DOG_init,DOG_fin=DOG_fin,BN_init=BN_init,BN_fin=BN_fin,root_rep=root_rep)
            
    print ( "Branch NN Test Procedure - Inférence on Benchmark - "+str(bool_inference_benchmark))
    if bool_inference_benchmark:   
        if main_network=="SR_EDSR":
            model_identite =  edsr_sofa(scale=4, num_res_blocks=16)
            model_identite.load_weights(os.path.join(benchmark_folder,"MODEL/weights_github_krasserm/weights-edsr/weights-edsr-16-x4/weights/edsr-16-x4/weights.h5"))
        else:
            model_identite=tf.keras.models.load_model(os.path.join(root_rep,"model.h5"),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False)
        Image_benchmark_BRANCH(model_identite=model_identite,main_network=main_network,type_branch=type_branch,ponderation_features=ponderation_features,border=border,FG_sigma_init=FG_sigma_init,FG_sigma_puissance=FG_sigma_puissance,BN_init=BN_init, BN_fin=BN_fin,  DOG_init=DOG_init, DOG_fin=DOG_fin,benchmark_folder=benchmark_folder,save_rep=root_rep,root_folder=root_folder, nombre_class=nombre_class, filtres=filtres,filtres_branch=filtres_branch,nombre_kernel=kernel, w_h=w_h,w_v=w_v, w_h_s=w_h_s,w_v_s=w_v_s,style=style_patch,clusters=clusters,bins=bins,sigma_noise_blur=sigma_noise_blur,cbcr_sty=cbcr_sty) 

    print ( "Branch NN Test Procedure - Individual Reports  - "+str(bool_process_test))
    if bool_process_test:
        model_identite=tf.keras.models.load_model(os.path.join(root_rep,"model.h5"),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False)
        w=process_test_controle_BRANCH(model_identite,main_network=main_network,taille=taille,root=root_folder,save_rep=os.path.join(root_rep,"visual_reports/"),save_rep_energie=os.path.join(root_rep,"energie/"),BN_init=BN_init,  BN_fin=BN_fin, DOG_init=DOG_init, DOG_fin=DOG_fin, nombre_class=nombre_class, border=border,nombre_patch_test=nombre_patch_test,type_branch=type_branch,style_patch=style_patch,clusters=clusters,bins=bins,sigma_noise_blur=sigma_noise_blur)


def experience(taille:int,border:int,R:int,nombre_patch_test:int,nbre_validation_data:int,filtres_sr:int, filtres_st:int, filtres_col:int, filtres_st3:int,  ouverture:int,kernel_sr:int,kernel_st:int,kernel_col:int,kernel_st3:int,nombre_class:int,BN_init:bool,BN_fin:bool,DOG_init:bool,DOG_fin:bool,FG_sigma_init:float,FG_sigma_puissance:float,loss_pixel_sr:bool, loss_perceptuelle_sr:list,loss_style_sr:list,loss_pixel_st:bool, loss_perceptuelle_st:list,loss_style_st:list, loss_pixel_st_col:bool,loss_pixel_st3:bool, loss_perceptuelle_st_col:list,loss_style_st_col:list,loss_style_st3:list,loss_perceptuelle_st3:list,ponderation_pixel:float,ponderation_perc_sr:float,ponderation_perc_st:float,ponderation_perc_st_col:float,ponderation_perc_st3:float,ponderation_style:float,ponderation_features:list,learning_rate_sr:float,learning_rate_st:float,learning_rate_st_col:float,learning_rate_st3:float,epochs_sr:int,epochs_st:int,epochs_st_col:int,epochs_st3:int,spe_sr:int,spe_st:int,spe_st_col:int,spe_st3:int,batch_size_sr:int,batch_size_st:int,batch_size_st_col:int,batch_size_st3:int,root_folder:str,out_dir:str,testing_main:bool,training_main:bool,main_network:str,testing_style:bool,training_style:bool,style_model:bool,cbcr_sty:bool,sigma_noise_blur:float,col_model:bool, testing_col:bool, training_col:bool,ST3_model:bool, testing_ST3:bool,  training_ST3 :bool, text:str,Flags):
    """
    Training Procedure followed by Testing procedure. Saves all results in a specific folder. Saves model, training and testing parameters (from flags and from variables).
    """
    
    # Folders preparation
    if main_network=="SR_EDSR":
        main_network_display = "SR"
    else:
        main_network_display = main_network
    benchmark_folder=os.path.join(root_folder,"External_Data") 
    
    nom_rep=str(text)+"_"+str(main_network_display)+"_filtres:"+str(filtres_sr)+"_class:"+str(nombre_class)
    root_rep=os.path.join(os.path.join(root_folder,"Results"),nom_rep)
    original_root=root_rep
    ensure_dir(root_rep)    
    for i in ["visual_reports","rapports_benchmark","energie","Model_saved_epochs","learning_data","final_saved","_info_model","rapports_benchmark_bdd"]:
        ensure_dir(os.path.join(root_rep,i))            
    
    # Gaussian Filters preparation
    w_h,w_v = Fi_filter(nombre_class,os.path.join(root_rep,"_info_model"),FG_sigma_init,FG_sigma_puissance, save = True) [0:2]  # MAIN NN Gaussian filters
    w_h_s,w_v_s,w_style_2d=two_Fi_filter(os.path.join(root_rep,"_info_model"),ouverture,0.001,True)                             # SIDE BRANCHES Gaussian filters
    
    #. ---------------------------------------- MAIN NETWORK --------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
    # Flags Saving as JSON
    with open(os.path.join(os.path.join(root_rep,"_info_model"),"flags.json"), "w") as flag_file:
        json.dump(vars(Flags), flag_file) 

    # Main Parameters Saving as TXT
    if training_main or testing_main:
        with open(os.path.join(os.path.join(root_rep,"_info_model"),"_PARAMETERS.json"), "w") as text_params:
            text_params.write(str("Données : Taille input :  "+ str(taille)+" x "+ str(taille)+", bordure :"+str(border)+" R : upsampling x"+str(R)+"\n"))
            text_params.write(str("Modèle : nbre de filtres SR :"+str(filtres_sr)+" taille des kernels :"+str(kernel_sr)+" nombre de branches :"+str(nombre_class)+" pondération (i.e distribution) entre le nbre de features ente les branches SR :"+str(ponderation_features)+" Booleens dans le réseau (BN_init,BN_fin,Dog_init,Dog_fin) :"+str(BN_init)+str(BN_init)+str(DOG_init)+str(DOG_fin)+"\n"))
            text_params.write(str(" variance initiale & facteur géometrique : "+str(FG_sigma_init)+ " , "+str(FG_sigma_puissance)+"\n"))
            text_params.write(" loss MSE : "+ str(loss_pixel_sr)+ " pondération :"+str(ponderation_pixel)+  " loss Perceptuelle :"+ str(loss_perceptuelle_sr)+" pondération :"+str(ponderation_perc_sr)+ ". Loss Gram (i.e gatys pour SR_network)+ "+str(loss_style_sr)+ " pondération :"+str(ponderation_style)+ "\n")
            text_params.write(("Training : learning rate : "+str(learning_rate_sr)+ " epochs : "+str(epochs_sr)+ " step par epoch :"+ str(spe_sr)+" batch size : "+str(batch_size_sr)+ "nombre patchs de validation par epoch: "+str(nbre_validation_data)+"\n"))
            text_params.write(("testing : nombre de patchs de tests: "+str(nombre_patch_test)+"\n"))
            
    # Training Process : MAIN NN (SR / DENOISINg / DEBLURRING )    
    if training_main and main_network!="SR_EDSR":
        information_learning=learning_MAIN_network(main_network=main_network,taille=taille,ponderation_features=ponderation_features,border=border,w_h=w_h, w_v=w_v, BN_init=BN_init, BN_fin=BN_fin,DOG_init=DOG_init, DOG_fin=DOG_fin, epochs=epochs_sr, step_per_epoc=spe_sr, batch_size=batch_size_sr, learning_rate=learning_rate_sr,save_rep=root_rep,nombre_class=nombre_class, filtres_sr=filtres_sr, nombre_kernel=kernel_sr,loss_pixel=loss_pixel_sr, loss_perceptuelle=loss_perceptuelle_sr,loss_style=loss_style_sr,ponderation_pixel=ponderation_pixel, ponderation_perc=ponderation_perc_sr, ponderation_style=ponderation_style,nom_par_epoch=os.path.join(os.path.join(root_rep,"Model_saved_epochs"),nom_rep+"_saved-model-{epoch:d}.h5"), log_dir=out_dir, root=root_folder ,R=R, nbre_validation_data=nbre_validation_data,sigma_noise_blur=sigma_noise_blur)
        
    # Testing Process  :  MAIN NN (SR / DENOISINg / DEBLURRING )  
    if testing_main and main_network!="SR_EDSR":
        procedure_test_MAIN(main_network,root_rep,ponderation_features, border,root_folder,benchmark_folder,taille,filtres_sr,ouverture,kernel_sr,kernel_st,nombre_class,nombre_patch_test,FG_sigma_init,FG_sigma_puissance,BN_init,BN_fin,DOG_init,DOG_fin,nom_rep,sigma_noise_blur,w_h,w_v,
                    bool_inference_benchmark=True,bool_calcul_energie=True,bool_process_test=True)
        
    #. ----------------------------------------BRANCH ST(y) NETWORK -------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
    if style_model: 
        # Folders preparation
        folder_selected_styles = os.path.join(benchmark_folder,"SELECTED_STYLES")
        list_style = sorted([x for x in os.listdir(folder_selected_styles) if os.path.isdir(folder_selected_styles)]) [5:6]
            
        for nom_style in list_style:
            tf.keras.backend.clear_session()
            print("TRANSFERT DE STYLE ST(y) - "+str(nom_style))
            
            # Folders preparation - Specific Style
            if cbcr_sty:
                name_specific_folder = "model_st(y)_col_"+nom_style+"_lambdaPerc:"+str(ponderation_perc_st)+"_ouverture:"+str(ouverture)+"_"+str(main_network)
            else:
                name_specific_folder = "model_st(y)_"+nom_style+"_lambdaPerc:"+str(ponderation_perc_st)+"_ouverture:"+str(ouverture)+"_"+str(main_network)
            root_rep=os.path.join(original_root,name_specific_folder)
            ensure_dir(root_rep)
            for i in ["visual_reports","rapports_benchmark","energie","Model_saved_epochs","learning_data","_info_model"]:
                ensure_dir(os.path.join(root_rep,i))  
            for j in range(epochs_st):
                ensure_dir(os.path.join(root_rep,os.path.join("rapports_controle_autre_epochs","visual_reports"+str(j))))
              
            # Flags Saving as JSON
            with open(os.path.join(os.path.join(root_rep,"_info_model"),"Flags.json"), "w") as flag_file:
                json.dump(vars(Flags), flag_file) 
            
            # Main Parameters Saving as TXT
            if training_style or testing_style:
                with open(os.path.join(os.path.join(root_rep,"_info_model"),"_PARAMETERS.json"), "w") as text_params:
                    text_params.write(str("Données : Taille input :  "+ str(taille)+" x "+ str(taille)+", bordure :"+str(border)+" R : upsampling x"+str(R)+"\n"))
                    text_params.write(str("Modèle : nbre de filtres SR :"+str(filtres_st)+" sigma de la branche de style : "+str(ouverture) +" taille des kernels :"+str(kernel_st)+" nombre de branches :"+str(nombre_class)+" pondération (i.e distribution) entre le nbre de features ente les branches SR :"+str(ponderation_features)+" Booleens dans le réseau (BN_init,BN_fin,Dog_init,Dog_fin) :"+str(BN_init)+str(BN_init)+str(DOG_init)+str(DOG_fin)+"\n"))
                    text_params.write(str(" variance initiale & facteur géometrique : "+str(FG_sigma_init)+ " , "+str(FG_sigma_puissance)+"\n"))
                    text_params.write(" loss MSE : "+ str(loss_pixel_st)+ " pondération :"+str(ponderation_pixel)+  " loss Perceptuelle :"+ str(loss_perceptuelle_st)+" pondération :"+str(ponderation_perc_st)+ ". Loss Gram (i.e gatys pour SR_network)+ "+str(loss_style_st)+ " pondération :"+str(ponderation_style)+ "\n")
                    text_params.write(("Training : learning rate : "+str(learning_rate_st)+ " epochs : "+str(epochs_st)+ " step par epoch :"+ str(spe_st)+" batch size : "+str(batch_size_st)+ "nombre patchs de validation par epoch: "+str(nbre_validation_data)+"\n"))
                    text_params.write(("testing : nombre de patchs de tests: "+str(nombre_patch_test)+"\n"))
                    
                    
            # Style image Preprocessing ---
            true_style = openImg_tf_format(os.path.join(folder_selected_styles,nom_style),nom_style,taille,True)
            imageio.imwrite(os.path.join(root_rep,str(nom_style)+"_croped.png"),true_style)  
            
            true_style=tf.convert_to_tensor(true_style,dtype=tf.float32)
            true_style=tf.reshape(true_style,(1,true_style.shape[0],true_style.shape[1],3)) 
            true_style_rgb=tf.identity(true_style[0,:,:,:])/255.
            
            true_s=normalize_with_moments(true_style/255.)*255. 
            true_s_img=tf.reshape(true_s,(true_s.shape[1],true_s.shape[2],3))
            imageio.imwrite(os.path.join(root_rep,str(nom_style)+"_croped_normalize.png"),true_s_img)  
            true_s=tf.expand_dims(true_s_img,axis=0)
            true_s=tf.convert_to_tensor(true_s,dtype=tf.float32)
            
            # Prepare Style tensor
            vgg16 = VGG16(include_top=False, weights='imagenet')
            vgg16.trainable = False
            style_gram=[] 
            
            for lay_style in range(18): 
                loss_model_style = tf.keras.models.Model(inputs=vgg16.inputs, outputs=vgg16.layers[lay_style].output)   
                S = loss_model_style(preprocess_input(true_s))
                S=S[0,:,:,:]
                S = gram_matrix_nobatch(S)
                style_gram.append(S)
            true_style=tf.reshape(true_style,(true_style.shape[1],true_style.shape[2],3)) # avec la new normalization

            # OPTIONAL - Display Dog(Style) on the top of 'image_test.png' (baby)
            dog_style_2d = w_style_2d[0].copy()-w_style_2d[1].copy()
                
            img_test = imageio.imread("image_test.png")/255.
            true_style=true_style/255.
            img_test=cv2.resize(img_test, dsize=(true_style.shape[0],true_style.shape[1]), interpolation=cv2.INTER_CUBIC)
            true_style_ycbcr =RGB2Ycbcr_numpy(true_style[:,:,0],true_style[:,:,1],true_style[:,:,2])
            u=np.concatenate((np.expand_dims(true_style_ycbcr[0],axis=-1),np.expand_dims(true_style_ycbcr[1],axis=-1),np.expand_dims(true_style_ycbcr[2],axis=-1)),axis=-1)
            imageio.imwrite(os.path.join(root_rep,str(nom_style)+"_ycbcr_cropped_normalize.png"),u*255.)  
                
            out_style_dog = scipy.signal.convolve2d(u[:,:,0],dog_style_2d[:,:,0,0],mode="same")
            imageio.imwrite(os.path.join(root_rep,str(nom_style)+"_output_dog_img_test.png"),out_style_dog*255.)
                
            img_test_ycbcr =RGB2Ycbcr_numpy(img_test[:,:,0],img_test[:,:,1],img_test[:,:,2])
            img_test_stylise_1=np.concatenate((np.expand_dims(img_test_ycbcr[0],axis=-1),np.expand_dims(img_test_ycbcr[1],axis=-1),np.expand_dims(img_test_ycbcr[2],axis=-1)),axis=-1)
            img_test_stylise_1[:,:,0]+=out_style_dog.copy()/1
            img_test_stylise_1=YCBCbCr2RGB(img_test_stylise_1*255.)
            imageio.imwrite(os.path.join(root_rep,str(nom_style)+"_output_test_stylisé_1.png"),img_test_stylise_1)
                
            img_test_stylise_10=np.concatenate((np.expand_dims(img_test_ycbcr[0],axis=-1),np.expand_dims(img_test_ycbcr[1],axis=-1),np.expand_dims(img_test_ycbcr[2],axis=-1)),axis=-1)
            img_test_stylise_10[:,:,0]+=out_style_dog.copy()/4
            img_test_stylise_10=YCBCbCr2RGB(img_test_stylise_10*255.)
            imageio.imwrite(os.path.join(root_rep,str(nom_style)+"_output_test_stylisé_10.png"),img_test_stylise_10)
                
            img_test_stylise_50=np.concatenate((np.expand_dims(img_test_ycbcr[0],axis=-1),np.expand_dims(img_test_ycbcr[1],axis=-1),np.expand_dims(img_test_ycbcr[2],axis=-1)),axis=-1)
            img_test_stylise_50[:,:,0]+=out_style_dog.copy()/10
            img_test_stylise_50=YCBCbCr2RGB(img_test_stylise_50*255.)
            imageio.imwrite(os.path.join(root_rep,str(nom_style)+"_output_test_stylisé_50.png"),img_test_stylise_50)
            
            style_patch_normalise = imageio.imread(os.path.join(root_rep,str(nom_style)+"_croped_normalize.png"))
            
            # Training procedure
            clusters,hist_style,bins=0,0,0 # Stcol
            if training_style:
                # Load MAIN Network and build BRANCH network on the top of the MAIN network--
                # Other NN
                if main_network=="SR_EDSR":
                    model_identite =  edsr_sofa(scale=4, num_res_blocks=16)
                    model_identite.load_weights(os.path.join(benchmark_folder,"MODEL/weights_github_krasserm/weights-edsr/weights-edsr-16-x4/weights/edsr-16-x4/weights.h5"))
                # OUR NN
                else:
                    model_identite=tf.keras.models.load_model(os.path.join(original_root,"model.h5"),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False)
                
                BN_fin=False
                learning_BRANCH_network(model_MAIN=model_identite, main_network=main_network,  type_branch="Stycbcr",root_rep=root_rep,root_folder=root_folder, style_img=nom_style, style_gram=style_gram,taille=taille, filtres_branch=filtres_st,border=border,w_h_s=w_h_s, w_v_s=w_v_s, BN_init=BN_init, BN_fin=BN_fin, DOG_init=DOG_init, DOG_fin=DOG_fin, loss_pixel=loss_pixel_st, loss_perceptuelle=loss_perceptuelle_st,loss_style=loss_style_st, ponderation_pixel=ponderation_pixel, ponderation_perc=ponderation_perc_st, ponderation_style=ponderation_style,epochs=epochs_st, step_per_epoc=spe_st, batch_size=batch_size_st, learning_rate=learning_rate_st, save_rep=root_rep,nombre_class=nombre_class, filtres_sr=filtres_sr,nombre_kernel=kernel_st,sigma_noise_blur=sigma_noise_blur,nom_par_epoch=os.path.join(os.path.join(root_rep,"Model_saved_epochs"),"_saved-model-{epoch:d}.h5"), log_dir=out_dir, root=root_folder , benchmark_folder=benchmark_folder,nbre_validation_data=nbre_validation_data,nombre_patch_test=nombre_patch_test,patch_style=true_style_rgb.numpy(),name_specific_folder=name_specific_folder,clusters=clusters,hist_style=hist_style,bins=bins,cbcr_sty=cbcr_sty)
            
            # Testing procedure
            if testing_style :
                BN_fin=False
                procedure_test_BRANCH(main_network=main_network, root_rep=root_rep,ponderation_features=ponderation_features, border=border,root_folder=root_folder,benchmark_folder=benchmark_folder,taille=taille,filtres=filtres_sr,filtres_branch=filtres_st, ouverture=ouverture,kernel=kernel_sr,kernel_branch=kernel_st,nombre_class=nombre_class,nombre_patch_test=nombre_patch_test,FG_sigma_init=FG_sigma_init,FG_sigma_puissance=FG_sigma_puissance,BN_init=BN_init,BN_fin=BN_fin,DOG_init=DOG_init,DOG_fin=DOG_fin,nom_rep=nom_rep,sigma_noise_blur=sigma_noise_blur,w_h=w_h,w_v=w_v,w_h_s=w_h_s,w_v_s=w_v_s,style_patch=true_style_rgb.numpy(), clusters=clusters,bins=bins,cbcr_sty=cbcr_sty,
                                         bool_inference_benchmark=True, bool_process_test=True,type_branch="Stycbcr")

    #. ----------------------------------------Modèle STcol NETWORK -------------------------------------------------- #
    
    if col_model:
        # Folders preparation
        folder_selected_styles = os.path.join(benchmark_folder,"SELECTED_COLORS")
        list_style = sorted([x for x in os.listdir(folder_selected_styles) if os.path.isdir(folder_selected_styles)])   [1:2]

        for nom_style in list_style:
            tf.keras.backend.clear_session()
            print("TRANSFERT DE COULEURS - "+str(nom_style))
            name_specific_folder = "model_couleur_"+nom_style+"_lambdaPerc:"+str(ponderation_perc_st_col)+"_"+str(main_network)
            taille_output_stcol = 352
            
            # Folders preparation - Specific Style
            root_rep=os.path.join(original_root,name_specific_folder)
            ensure_dir(root_rep)
            for i in ["visual_reports","rapports_benchmark","energie","Model_saved_epochs","learning_data","_info_model"]:
                ensure_dir(os.path.join(root_rep,i))  
            for j in range(epochs_st_col):
                ensure_dir(os.path.join(root_rep,os.path.join("rapports_controle_autre_epochs","visual_reports"+str(j))))

            # Style image Preprocessing ---
            true_style = openImg_tf_format(os.path.join(folder_selected_styles,nom_style),nom_style,0,False)
            imageio.imwrite(os.path.join(root_rep,str(nom_style)+"_.png"),true_style)  
            true_style=tf.convert_to_tensor(true_style,dtype=tf.float32)
            
            true_style=tf.reshape(true_style,(1,true_style.shape[0],true_style.shape[1],3)) 
            true_style_rgb=tf.identity(true_style[0,:,:,:])/255.
            
            true_s=tf_rgb2ycbcr(true_style/255.)
            true_s=tf.reshape(true_s,(true_s.shape[1],true_s.shape[2],3))
            imageio.imwrite(os.path.join(root_rep,str(nom_style)+"_ycbcr.png"),true_s)  
            true_s=tf.expand_dims(true_s,axis=0)
            true_s=tf.convert_to_tensor(true_s,dtype=tf.float32)
            true_style=tf.reshape(true_style,(true_style.shape[1],true_style.shape[2],3)) 
            
            
            # Style Histogram
            true_s = tf.squeeze(true_s[:,:,:,1:3]/256.,axis=0)
            true_s_hist = tf.reshape(true_s,(true_s.shape[0]*true_s.shape[1],2))
            bins=14
            clusters_cb=tf.linspace(0.,256., bins)
            clusters_cr=tf.linspace(0.,256., bins)
            clusters=tf.stack([tf.tile([clusters_cb],[bins,1]),tf.transpose(tf.tile([clusters_cr],[bins,1]))],axis=0)
            clusters=tf.reshape(clusters,(2,bins*bins))
            hist_style = histogram_2d(true_s_hist, clusters , true_s_hist.shape[0],true_s_hist.shape[1])       
            Save_Hist_1d_tf(hist_style,os.path.join(root_rep,str(nom_style)+"__HISTOGRAM_1D_STYLE_.png"),bins)
            Save_Hist_2d_tf(hist_style,os.path.join(root_rep,str(nom_style)+"__HISTOGRAM_2D_STYLE__.png"),bins)

            
            # Saving process parameters 
            with open(os.path.join(os.path.join(root_rep,"_info_model"),"Flags.json"), "w") as flag_file:
                json.dump(vars(Flags), flag_file) 
        
            if training_col or testing_col:
                with open(os.path.join(os.path.join(root_rep,"_info_model"),"_PARAMETERS.json"), "w") as text_params:
                    text_params.write(str("Données : Taille input :  "+ str(taille_output_stcol)+" x "+ str(taille_output_stcol)+", bordure :"+str(border)+" R : upsampling x"+str(R)+"\n"))
                    text_params.write(str("Modèle : nbre de filtres SR :"+str(filtres_col)+" taille des kernels :"+str(kernel_col)+" nombre de branches :"+str(nombre_class)+" pondération (i.e distribution) entre le nbre de features ente les branches SR :"+str(ponderation_features)+" Booleens dans le réseau (BN_init,BN_fin,Dog_init,Dog_fin) :"+str(BN_init)+ str(BN_init)+str(DOG_init)+str(DOG_fin)+"\n"))
                    text_params.write(str(" variance initiale & facteur géometrique : "+str(FG_sigma_init)+ " , "+str(FG_sigma_puissance)+"\n"))
                    text_params.write(" loss MSE : "+ str(loss_pixel_st_col)+ " pondération :"+str(ponderation_pixel)+  " loss Perceptuelle :"+ str(loss_perceptuelle_st_col)+" pondération :"+str(ponderation_perc_st_col)+ ". Loss Gram (i.e gatys pour SR_network)+ "+str(loss_style_st_col)+ " pondération :"+str(ponderation_style)+ "\n")
                    text_params.write(("Training : learning rate : "+str(learning_rate_st_col)+ " epochs : "+str(epochs_st_col)+ " step par epoch :"+ str(spe_st_col)+" batch size : "+str(batch_size_st_col)+ "nombre patchs de validation par epoch: "+str(nbre_validation_data)+"\n"))
                    text_params.write(("testing : nombre de patchs de tests: "+str(nombre_patch_test)+"\n"))
            
            style_gram=0
            # Training procedure ---
            tf.keras.backend.clear_session()
            if training_col:
                # Load MAIN network and build COL network on the top of MAIN network
                if main_network=="SR_EDSR":# Other NN
                    model_identite =  edsr_sofa(scale=4, num_res_blocks=16)
                    model_identite.load_weights(os.path.join(benchmark_folder,"MODEL/weights_github_krasserm/weights-edsr/weights-edsr-16-x4/weights/edsr-16-x4/weights.h5"))
                
                else:# OUR NN
                    model_patch=tf.keras.models.load_model(os.path.join(original_root,"model.h5"),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False) 
                    new_model = MAIN_network(sizex=taille_output_stcol,sizey=taille_output_stcol,ponderation_features=ponderation_features,nombre_class=nombre_class,filtres=filtres_sr,kernel=kernel_sr,w_h=w_h,w_v=w_v,loss_pixel=loss_pixel_sr,loss_perceptuelle=loss_perceptuelle_sr,   loss_style=loss_style_sr,   ponderation_pixel=ponderation_pixel, ponderation_perc=ponderation_perc_sr, ponderation_style=ponderation_style, learning_rate=learning_rate_sr, BN_init=BN_init, BN_fin=BN_fin, DOG_fin=DOG_fin,DOG_init=DOG_init)   [0]
                    model_identite =copy_parameters(model=new_model,model_patch=model_patch)
                    
                learning_BRANCH_network(model_MAIN=model_identite, main_network=main_network, type_branch="Stcol",root_rep=root_rep,root_folder=root_folder, style_img=nom_style,taille=taille_output_stcol, filtres_branch=filtres_col,border=border, w_h_s=w_h_s, w_v_s=w_v_s, BN_init=BN_init, BN_fin=BN_fin, DOG_init=DOG_init, DOG_fin=DOG_fin, loss_pixel=loss_pixel_st_col, loss_perceptuelle=loss_perceptuelle_st_col,   loss_style=loss_style_st_col, ponderation_pixel=ponderation_pixel, ponderation_perc=ponderation_perc_st_col, ponderation_style=ponderation_style,epochs=epochs_st_col, step_per_epoc=spe_st_col, batch_size=batch_size_st_col, learning_rate=learning_rate_st_col, save_rep=root_rep,nombre_class=nombre_class, filtres_sr=filtres_sr, nombre_kernel=kernel_col,nom_par_epoch=os.path.join(os.path.join(root_rep,"Model_saved_epochs"),"_saved-model-{epoch:d}.h5"), log_dir=out_dir, root=root_folder , benchmark_folder=benchmark_folder,nbre_validation_data=nbre_validation_data,nombre_patch_test=nombre_patch_test,patch_style=true_style_rgb.numpy(),name_specific_folder=name_specific_folder,sigma_noise_blur=sigma_noise_blur,style_gram=style_gram,cbcr_sty=cbcr_sty,clusters=clusters,hist_style=hist_style,bins=bins)
                
            # Testing procedure ---
            if testing_col :
                procedure_test_BRANCH(main_network=main_network, root_rep=root_rep , ponderation_features=ponderation_features, border=border,root_folder=root_folder,benchmark_folder=benchmark_folder,taille=taille_output_stcol,filtres=filtres_sr,filtres_branch=filtres_col, ouverture=ouverture, kernel=kernel_sr,kernel_branch=kernel_col,nombre_class=nombre_class,nombre_patch_test=nombre_patch_test,FG_sigma_init=FG_sigma_init,FG_sigma_puissance=FG_sigma_puissance,BN_init=BN_init,BN_fin=BN_fin,DOG_init=DOG_init,DOG_fin=DOG_fin,nom_rep=nom_rep,sigma_noise_blur=sigma_noise_blur,w_h=w_h,w_v=w_v,w_h_s=w_h_s,w_v_s=w_v_s,style_patch=true_style_rgb.numpy(),clusters=clusters,bins=bins,cbcr_sty=cbcr_sty,
                                             bool_inference_benchmark=True,bool_process_test=True,type_branch="Stcol")

    #. ----------------------------------------Modèle ST3 NETWORK -------------------------------------------------- #
    
    if ST3_model:
        folder_selected_styles = os.path.join(benchmark_folder,"SELECTED_STYLES_ST3")
        list_style = sorted([x for x in os.listdir(folder_selected_styles) if os.path.isdir(folder_selected_styles)])[1:2]

        for nom_style in list_style:
            tf.keras.backend.clear_session()
            print("TRANSFERT DE STYLE COMPLET - "+str(nom_style))
            name_specific_folder = "model_ST3_"+nom_style+"_lambdaPerc:"+str(ponderation_perc_st3)+"_"+str(main_network)
            
            taille_output_st3= 352 # jonshon size 512
            
            # Folders preparation - Specific Style
            root_rep=os.path.join(original_root,name_specific_folder)
            ensure_dir(root_rep)
            for i in ["visual_reports","rapports_benchmark","energie","Model_saved_epochs","learning_data","_info_model"]:
                ensure_dir(os.path.join(root_rep,i))  
                
            for j in range(epochs_st3):
                ensure_dir(os.path.join(root_rep,os.path.join("rapports_controle_autre_epochs","visual_reports"+str(j))))
                    
            # Style image Preprocessing
            true_style = openImg_tf_format(os.path.join(folder_selected_styles,nom_style),nom_style,0,False)
            true_style=tf.image.resize(true_style,((taille_output_st3,taille_output_st3)))
            imageio.imwrite(os.path.join(root_rep,str(nom_style)+"_croped.png"),true_style)  
            true_style=tf.convert_to_tensor(true_style,dtype=tf.float32)
            true_style=tf.reshape(true_style,(1,true_style.shape[0],true_style.shape[1],3)) 
            true_style_rgb=tf.identity(true_style[0,:,:,:])/255.

            true_s=(true_style/255.)*255.  #normalize_with_moments_ycbcr
            true_s=tf.reshape(true_s,(true_s.shape[1],true_s.shape[2],3))
            imageio.imwrite(os.path.join(root_rep,str(nom_style)+"_croped_normalize.png"),true_s)  
            true_s=tf.expand_dims(true_s,axis=0)
            true_s=tf.convert_to_tensor(true_s,dtype=tf.float32)
            
            # Prepare Style tensor
            vgg16 = VGG16(include_top=False, weights='imagenet')
            vgg16.trainable = False
            style_gram=[] 
            
            for lay_style in range(16): 
                loss_model_style = tf.keras.models.Model(inputs=vgg16.inputs, outputs=vgg16.layers[lay_style].output)   
                S = loss_model_style(tf.keras.applications.vgg16.preprocess_input(true_s))
                S=S[0,:,:,:]
                S = gram_matrix_nobatch(S)
                style_gram.append(S)
            true_style=tf.reshape(true_style,(true_style.shape[1],true_style.shape[2],3)) # avec la new normalization
            true_style=true_style.numpy()
            
            # Saving process parameters 
            with open(os.path.join(os.path.join(root_rep,"_info_model"),"Flags.json"), "w") as flag_file:
                json.dump(vars(Flags), flag_file) 
        
            if training_ST3 or testing_ST3:
                with open(os.path.join(os.path.join(root_rep,"_info_model"),"_PARAMETERS.json"), "w") as text_params:
                    text_params.write(str("Données : Taille input :  "+ str(taille_output_st3)+" x "+ str(taille_output_st3)+", bordure :"+str(border)+" R : upsampling x"+str(R)+"\n"))
                    text_params.write(str("Modèle : nbre de filtres SR :"+str(filtres_st3)+" taille des kernels :"+str(kernel_st3)+" nombre de branches :"+str(nombre_class)+" pondération (i.e distribution) entre le nbre de features ente les branches SR :"+str(ponderation_features)+" Booleens dans le réseau (BN_init,BN_fin,Dog_init,Dog_fin) :"+str(BN_init)+str(BN_init)+str(DOG_init)+str(DOG_fin)+"\n"))
                    text_params.write(str(" variance initiale & facteur géometrique : "+str(FG_sigma_init)+ " , "+str(FG_sigma_puissance)+"\n"))
                    text_params.write(" loss MSE : "+ str(loss_pixel_st3)+ " pondération :"+str(ponderation_pixel)+  " loss Perceptuelle :"+ str(loss_perceptuelle_st3)+" pondération :"+str(ponderation_perc_st3)+ ". Loss Gram (i.e gatys pour SR_network)+ "+str(loss_style_st3)+ " pondération :"+str(ponderation_style)+ "\n")
                    text_params.write(("Training : learning rate : "+str(learning_rate_st3)+ " epochs : "+str(epochs_st3)+ " step par epoch :"+ str(spe_st3)+" batch size : "+str(batch_size_st3)+ "nombre patchs de validation par epoch: "+str(nbre_validation_data)+"\n"))
                    text_params.write(("testing : nombre de patchs de tests: "+str(nombre_patch_test)+"\n"))
            
            # Training procedure
            clusters,hist_style,bins=0,0,0
            if training_ST3:
                # Load MAIN network and build COL network on the top of MAIN network
                if main_network=="SR_EDSR":# Other NN
                    model_identite =  edsr_sofa(scale=4, num_res_blocks=16)
                    model_identite.load_weights(os.path.join(benchmark_folder,"MODEL/weights_github_krasserm/weights-edsr/weights-edsr-16-x4/weights/edsr-16-x4/weights.h5"))
                
                else:# OUR NN
                    model_patch=tf.keras.models.load_model(os.path.join(original_root,"model.h5"),custom_objects={'tf': tf,'PSNR':PSNR,'vgg_loss_ycbcr':vgg_loss_ycbcr},compile=False) 
                    new_model = MAIN_network(sizex=taille_output_st3,sizey=taille_output_st3,ponderation_features=ponderation_features,nombre_class=nombre_class,filtres=filtres_sr,kernel=kernel_sr,w_h=w_h , w_v=w_v, loss_pixel=loss_pixel_sr,loss_perceptuelle=loss_perceptuelle_sr,   loss_style=loss_style_sr,   ponderation_pixel=ponderation_pixel, ponderation_perc=ponderation_perc_sr, ponderation_style=ponderation_style, learning_rate=learning_rate_sr, BN_init=BN_init, BN_fin=BN_fin, DOG_fin=DOG_fin,DOG_init=DOG_init)   [0]
                    model_identite =copy_parameters(model=new_model,model_patch=model_patch)
    
                learning_BRANCH_network(model_MAIN=model_identite, main_network=main_network, type_branch="St3",root_rep=root_rep,root_folder=root_folder, style_img=nom_style,taille=taille_output_st3, filtres_branch=filtres_st3,border=border,w_h_s=w_h_s, w_v_s=w_v_s, BN_init=BN_init, BN_fin=BN_fin, DOG_init=DOG_init, DOG_fin=DOG_fin, loss_pixel=loss_pixel_st3, loss_perceptuelle=loss_perceptuelle_st3,   loss_style=loss_style_st3, ponderation_pixel=ponderation_pixel, ponderation_perc=ponderation_perc_st3, ponderation_style=ponderation_style,epochs=epochs_st3, step_per_epoc=spe_st3, batch_size=batch_size_st3, learning_rate=learning_rate_st3, save_rep=root_rep,nombre_class=nombre_class, filtres_sr=filtres_sr, nombre_kernel=kernel_st3,nom_par_epoch=os.path.join(os.path.join(root_rep,"Model_saved_epochs"),"_saved-model-{epoch:d}.h5"), log_dir=out_dir, root=root_folder , benchmark_folder=benchmark_folder,nbre_validation_data=nbre_validation_data,nombre_patch_test=nombre_patch_test,patch_style=true_style_rgb.numpy(),name_specific_folder=name_specific_folder,sigma_noise_blur=sigma_noise_blur,style_gram=style_gram, clusters=clusters,hist_style=hist_style,bins=bins,cbcr_sty=cbcr_sty)
                
            # Testing procedure
            if testing_ST3 :
                procedure_test_BRANCH(main_network=main_network, root_rep=root_rep , ponderation_features=ponderation_features, border=border,root_folder=root_folder,benchmark_folder=benchmark_folder,taille=taille_output_st3,filtres=filtres_sr,filtres_branch=filtres_st3, ouverture=ouverture,kernel=kernel_sr,kernel_branch=kernel_st3,nombre_class=nombre_class,nombre_patch_test=nombre_patch_test,FG_sigma_init=FG_sigma_init,FG_sigma_puissance=FG_sigma_puissance,BN_init=BN_init,BN_fin=BN_fin,DOG_init=DOG_init,DOG_fin=DOG_fin,nom_rep=nom_rep,sigma_noise_blur=sigma_noise_blur,w_h=w_h,w_v=w_v,w_h_s=w_h_s,w_v_s=w_v_s,style_patch=true_style_rgb.numpy(),clusters=clusters,bins=bins,cbcr_sty=cbcr_sty,
                                             bool_inference_benchmark=True,bool_process_test=True,type_branch="St3")
                
    


