# Allocating enough memory to Gpu
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

from Process_Train_Test import *
import argparse 
import tensorflow as tf

# Flags
if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    
    # --- 0. ROOT PATH TO DATA -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('-r', dest = 'root', type = str, default=''  ,help = 'Root folder for data')        #'/data/chercheurs/durand192/SMSNN/Data_results' : serveurs   'home/personnels/durand192/SMSNN/Data_results'   : local
    
    # --- I. MAIN NN --- (Denoising / SR or SR_EDSR / Blurring)-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('-training_main', dest = 'train_main', type = bool, default=False, help = 'boolean for TRAINING Main Neural Network (SR or DENOISE)') 
    parser.add_argument('-testing_main', dest = 'test_main', type = bool, default=False, help = 'boolean for TESTING Main Neural Network (SR or DENOISE)') 
    parser.add_argument('-main_network', dest = 'main_network', type = str, default="BLURRING"  ,help =" String for the main network to solve SR / DENOISING / BLURRING /// SR_EDSR")    
    parser.add_argument('-sigma_noise_blur', dest = 'sigma_noise_blur', type = float,default=3.0, help= 'Sigma for normal distribution defining gaussian blur or noise')      # 0.1 : noise       3.0 : blurring     
    
    # Learning Management
    parser.add_argument('-e_sr', dest = 'epochs_sr', type = int,default=60, help= 'number of epochs for MAIN Network training')
    parser.add_argument('-b_sr', dest = 'batch_size_sr', type = int, default=8, help = 'batch size for MAIN Network training') 
    parser.add_argument('-spe_sr', dest = 'spe_sr', type = int,default=2300, help= 'number of steps per epoch for MAIN Network training')
    parser.add_argument('-lrate_sr', dest = 'learning_rate_sr', type = float, default=0.00025, help= "learning rate MAIN Network") 
    
    # NN Parameters
    parser.add_argument('-f_sr', dest = 'filtres_sr', type = int, default=32, help = 'Number of filters in convolutional layers for the MAIN Network') 
    parser.add_argument('-k_sr', dest = 'kernel_sr', type = int, default=3, help = 'kernel size of the MAIN Network') 
    parser.add_argument('-c', dest = 'composantes', type = int, default=6, help = 'Number of MAIN branches in the MAIN Network') 
    parser.add_argument('-f_dist', dest = 'filter_distribution', type = list, default=[0.5, 0.8, 1.0, 1.2, 1.4, 1.2], help = 'weights distribution of the filters into the different MAIN branches of the MAIN Network')    
    
    # Loss
    parser.add_argument('-l_pixel_sr', dest = 'loss_pixel_sr', type = bool, default=True, help = 'boolean for training with a pixel-wise loss (L2 or L1) (MAIN Network)') 
    parser.add_argument('-l_perc_sr', dest = 'loss_perceptuelle_sr', type = list, default=[5,7,9], help = 'List of the VGG 16 for computing perceptual losses (MAIN Network)') 
    parser.add_argument('-l_style_sr', dest = 'loss_style_sr', type = list, default=[], help = 'List of the VGG 16 for computing texture losses (MAIN Network)') 
    parser.add_argument('-pon_pixel', dest = 'pon_pixel', type = float,default=1, help= 'weight for pixel-wise loss (MAIN Network)') 
    parser.add_argument('-pon_style', dest = 'pon_style', type = float,default=1, help= 'weight for texture losses (MAIN Network)') 
    parser.add_argument('-pon_perc_sr', dest = 'pon_perc_sr', type = float,default=1, help= "weight for perceptual losses (MAIN Network) ") 

    # --- II. Style BRANCH - ST-step (STy AND Stycbcr)-(y/cb,cr)--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('-style_model', dest = 'style_model', type = bool, default=False, help='BOOLEAN FOR PROCESSING ST(y/cb,cr) BRANCH TRAINING OR TESTING') 
    parser.add_argument('-training_style', dest = 'train_style_bool', type = bool, default=True, help = 'boolean for TRAINING ST(y/cb,cr)') 
    parser.add_argument('-testing_style', dest = 'test_style_bool', type = bool, default=True, help = 'boolean for TESTING ST(y/cb,cr)') 
    parser.add_argument('-cbcr_sty', dest = 'cbcr_sty', type = bool, default=False, help = 'boolean ; predict color if True') 
    
    # Learning Management
    parser.add_argument('-e_st', dest = 'epochs_st', type = int,default=5, help= 'number of epochs for ST(y/cb,cr) Network training')   
    parser.add_argument('-b_st', dest = 'batch_size_st', type = int, default=6, help = 'batch size for ST(y/cb,cr) Network training') 
    parser.add_argument('-spe_st', dest = 'spe_st', type = int,default=4000, help= 'number of steps per epoch for ST(y/cb,cr) Network training')                
    parser.add_argument('-lrate_st', dest = 'learning_rate_st', type = float, default=0.0035, help= "learning rate ST(y/cb,cr) Network")  # (lr 0.0035 ; 6+ epochs )
    
    # NN Parameters
    parser.add_argument('-f_st', dest = 'filtres_st', type = int, default=24, help = 'Number of filters in convolutional layers for the  ST(y/cb,cr) Network') 
    parser.add_argument('-k_st', dest = 'kernel_st', type = int, default=3, help = 'kernel size for the  ST(y/cb,cr) Network') 
    parser.add_argument('-ouv', dest = 'ouverture', type = int, default=2.0, help = 'Difference between variance of the gaussian kernel for building the high frequency passband filter (DoG corresponding to : DoG(sigma=ouverture)-DoG(0.01)')

    # Loss
    parser.add_argument('-l_pixel_st', dest = 'loss_pixel_st', type = bool, default=False, help = 'boolean for training with a pixel-wise loss (L2 or L1) (ST(y/cb,cr))') 
    parser.add_argument('-l_perc_st', dest = 'loss_perceptuelle_st', type = list, default=[7], help = 'List of the VGG 16 for computing perceptual losses (ST(y/cb,cr))') 
    parser.add_argument('-l_style_st', dest = 'loss_style_st', type = list, default=[2,5,9,13], help = 'List of the VGG 16 for computing Style losses (ST(y/cb,cr))') 
    parser.add_argument('-pon_perc_st', dest = 'pon_perc_st', type = float,default=160000, help= "weight for perceptual losses (ST(y/cb,cr)) ") # weight for LTexture and Lpixel are 1.

    # --- III. Color Style BRANCH - COL-step (STcol)-(col)--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('-col_model', dest = 'col_model', type = bool, default=False, help = 'BOOLEAN FOR PROCESSING ST(col) BRANCH TRAINING OR TESTING') 
    parser.add_argument('-training_col', dest = 'train_col_bool', type = bool, default=True, help = 'boolean for TRAINING ST(col)') 
    parser.add_argument('-testing_col', dest = 'test_col_bool', type = bool, default=True, help = 'boolean for TESTING ST(col)') 

    # Learning Management
    parser.add_argument('-e_st_col', dest = 'epochs_st_col', type = int,default=10, help= 'number of epochs for STcol Network training')  
    parser.add_argument('-b_st_col', dest = 'batch_size_st_col', type = int, default=4, help = 'batch size for STcol Network training') 
    parser.add_argument('-spe_st_col', dest = 'spe_st_col', type = int,default=1800, help= 'number of steps per epoch for STcol Network training')                
    parser.add_argument('-lrate_st_col', dest = 'learning_rate_st_col', type = float, default=0.15, help= "learning rate STcol Network")   
    
    # NN Parameters
    parser.add_argument('-f_col', dest = 'filtres_col', type = int, default=32, help = 'Number of filters in convolutional layers for the  STcol Network') 
    parser.add_argument('-k_col', dest = 'kernel_col', type = int, default=3, help = 'kernel size for the  STcol Network') 

    # Loss
    parser.add_argument('-l_pixel_st_col', dest = 'loss_pixel_st_col', type = bool, default=True, help = 'boolean for training with a pixel-wise loss (L2 or L1) (STcol)') 
    parser.add_argument('-l_perc_st_col', dest = 'loss_perceptuelle_st_col', type = list, default=[9], help = 'List of the VGG 16 for computing perceptual losses (STcol)') 
    parser.add_argument('-l_style_st_col', dest = 'loss_style_st_col', type = list, default=[], help = 'List of the VGG 16 for computing Style losses (STcol)') 
    parser.add_argument('-pon_perc_st_col', dest = 'pon_perc_st_col', type = float,default=0.3, help= "weight for perceptual losses (STcol)") 

    # ---  IV. Classic Style Transfert BRANCH - Style-Transfert-step (ST3)-(y,cb,cr)------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('-ST3_model', dest = 'ST3_model', type = bool, default=False, help = 'BOOLEAN FOR PROCESSING ST3 BRANCH TRAINING OR TESTING') 
    parser.add_argument('-training_ST3', dest = 'train_ST3_bool', type = bool, default=True, help = 'boolean for TRAINING ST3') 
    parser.add_argument('-testing_ST3', dest = 'test_ST3_bool', type = bool, default=True, help = 'boolean for TESTING ST3') 

    # Learning Management
    parser.add_argument('-e_st3', dest = 'epochs_st3', type = int,default=25, help= 'number of epochs for ST3 Network training') 
    parser.add_argument('-b_st3', dest = 'batch_size_st3', type = int, default=4, help = 'batch size for ST3 Network training') 
    parser.add_argument('-spe_st3', dest = 'spe_st3', type = int,default=1800, help= 'number of steps per epoch for ST(Y,CB,CR) Network training')
    parser.add_argument('-lrate_st3', dest = 'learning_rate_st3', type = float, default=0.0001, help= "learning rate ST3_NETWORK")

    # NN Parameters
    parser.add_argument('-f_st3', dest = 'filtres_st3', type = int, default=84, help = 'Number of filters in convolutional layers for the  STcol Network') 
    parser.add_argument('-k_st3', dest = 'kernel_st3', type = int, default=3, help = 'kernel size for the  STcol Network') 

    # Loss
    parser.add_argument('-l_pixel_st3', dest = 'loss_pixel_st3', type = bool, default=False, help = 'boolean for training with a pixel-wise loss (L2 or L1) (ST3)') 
    parser.add_argument('-l_perc_st3', dest = 'loss_perceptuelle_st3', type = list, default=[4], help = 'List of the VGG 16 for computing perceptual losses (ST3)') 
    parser.add_argument('-l_style_st3', dest = 'loss_style_st3', type = list, default=[2,5,9,13], help = 'List of the VGG 16 for computing Style losses (ST3)') 
    parser.add_argument('-pon_perc_st3', dest = 'pon_perc_st3', type = float,default=3000000, help= "weight for perceptual losses (ST3)") 

    # ---  V. Common Parameters --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('-DOG_init', dest = 'DOG_init', type = bool, default=False, help = 'Boolean for using Dog as branch inputs')  
    parser.add_argument('-DOG_fin', dest = 'DOG_fin', type = bool, default=True, help = 'Boolean for using Dog as branch outputs') 
    parser.add_argument('-BatchNorm_init', dest = 'BN_init', type = bool, default=True, help = 'BatchNorm or not on input data (after Dog)') 
    parser.add_argument('-BatchNorm_fin', dest = 'BN_fin', type = bool, default=True, help = 'BatchNorm or not on output data (after Dog)') 

    
    parser.add_argument('-t', dest = 'out_dir',default='/export/home/durand192/Bureau/Python/PatchGenerator/Graph', type = str, help = 'Path for the tensorboard log')  
    parser.add_argument('-l_unique', dest = 'loss_unique', type = bool, default=True, help = '(single output) booléen indiquant si la loss concerne la sortie globale ou l ensemble des sorties intermediaires') 

    #  Gaussian blurs for making Dog
    parser.add_argument('-sigma_0', dest = 'sigma_0', type = float, default=1., help = 'Scale 0 for gaussian decomposition') 
    parser.add_argument('-k_i', dest = 'sigma_k_i', type = float, default=1.6, help = 'Geometrical evolution of gaussian scales')     # SIFT sqrt(2) et 1.5=sigma0    
    
    # Data Manegement (must match 'main_datapreprocessing.py')
    parser.add_argument('-u', dest = 'upfactor', type = int,default=4, help= 'up factor from HR to LR')
    parser.add_argument('-tl', dest = 'taille', type = int,default=224, help= 'Original Patch size without border (High resolution size)') 
    parser.add_argument('-bo', dest = 'border', type = int,default=15, help= 'border size')
    
    # Size of Validation & Test Samples
    parser.add_argument('-npt', dest = 'nbre_patch_test', type = int,default=200, help= 'Number of test patch to extract for statistical testing procedure')
    parser.add_argument('-npv', dest = 'nbre_patch_validation', type = int,default=4000, help= 'Number of validation patch to extract for cross-validation during training')

    args = parser.parse_args()
    
    # -----------------------------------------------------------ASSERT------------------------------------------------------------------------------------------------------------
    
    assert args.main_network=="SR" or args.main_network=="DENOISING" or args.main_network=="BLURRING" or args.main_network=="SR_EDSR" , " Please choose 'SR' or 'DENOISING' or 'BLURRING' as main task "
    
    # ----------------------------------------------------------- EXPERIMENTS -----------------------------------------------------------------------------------------------------    
    experience( filtres_sr=args.filtres_sr, filtres_st=args.filtres_st, filtres_col=args.filtres_col, filtres_st3=args.filtres_st3, kernel_col = args.kernel_col, kernel_sr=args.kernel_sr,kernel_st=args.kernel_st, kernel_st3=args.kernel_st3,ouverture = args.ouverture,   nombre_class=args.composantes, border = args.border,  R=args.upfactor, taille=args.taille, root_folder=args.root, out_dir=args.out_dir, epochs_sr=args.epochs_sr,epochs_st=args.epochs_st, epochs_st_col=args.epochs_st_col, epochs_st3=args.epochs_st3,spe_sr=args.spe_sr,spe_st=args.spe_st, spe_st_col=args.spe_st_col, spe_st3=args.spe_st3,batch_size_sr=args.batch_size_sr,batch_size_st=args.batch_size_st, batch_size_st_col=args.batch_size_st_col, batch_size_st3=args.batch_size_st3,learning_rate_sr = args.learning_rate_sr,learning_rate_st=args.learning_rate_st,learning_rate_st_col=args.learning_rate_st_col,learning_rate_st3=args.learning_rate_st3,testing_main=args.test_main, training_main = args.train_main,  sigma_noise_blur=args.sigma_noise_blur,style_model=args.style_model, testing_style=args.test_style_bool, training_style = args.train_style_bool , cbcr_sty=args.cbcr_sty,  col_model=args.col_model, testing_col=args.test_col_bool, training_col = args.train_col_bool , ST3_model=args.ST3_model, testing_ST3=args.test_ST3_bool, training_ST3 = args.train_ST3_bool , nombre_patch_test = args.nbre_patch_test, nbre_validation_data=args.nbre_patch_validation,ponderation_features=args.filter_distribution,ponderation_pixel=args.pon_pixel, ponderation_perc_sr = args.pon_perc_sr, ponderation_style= args.pon_style, FG_sigma_init=args.sigma_0, FG_sigma_puissance=args.sigma_k_i, BN_init=args.BN_init, BN_fin=args.BN_fin, DOG_init=args.DOG_init,DOG_fin=args.DOG_fin,main_network=args.main_network,  
               loss_pixel_sr=args.loss_pixel_sr, loss_perceptuelle_sr=args.loss_perceptuelle_sr,  loss_style_sr=args.loss_style_sr, loss_pixel_st=args.loss_pixel_st, loss_perceptuelle_st=args.loss_perceptuelle_st,loss_style_st=args.loss_style_st, loss_pixel_st_col=args.loss_pixel_st_col, loss_perceptuelle_st_col=args.loss_perceptuelle_st_col,loss_style_st_col=args.loss_style_st_col, loss_pixel_st3=args.loss_pixel_st3, loss_perceptuelle_st3=args.loss_perceptuelle_st3,loss_style_st3=args.loss_style_st3,ponderation_perc_st =args.pon_perc_st, ponderation_perc_st_col=args.pon_perc_st_col, ponderation_perc_st3=args.pon_perc_st3,   
               text="Main_NeuralNetwork",Flags=args)  
    
    