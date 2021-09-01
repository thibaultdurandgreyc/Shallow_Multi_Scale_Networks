from Data_preprocessing.Data_preprocessing import *
from various_functions.tensor_tf_data_fonctions import *
from various_functions.numpy_data_fonctions import *
from Networks.model_management import *
from various_functions.directory_management import *
from various_functions.custom_filters import *
import argparse

'''
Create Patchs data in specific folders
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Launch training - Testing -Etude réseau')    
    
    parser.add_argument('-r', dest = 'root_folder', type = str, default="", help = 'Folder to save patches')    #  "/home/personnels/durand192/SMSNN/Data_results" /// "/data/chercheurs/durand192/SMSNN/Data_results"
    parser.add_argument('-d', dest = 'delta', type = int,default=0, help = 'overlap entre les patchs d entrainement')
    parser.add_argument('-u', dest = 'upfactor', type = int,default=4, help= 'up factor from HR to LR')
    parser.add_argument('-tl', dest = 'taille', type = int,default=224, help= 'patch size')#224   352  512
    parser.add_argument('-s', dest = 'seuil', type = int,default=40, help= 'minimal variance for saving patches (8 bits)')
    parser.add_argument('-b', dest = 'border', type = int,default=15, help= 'borders to save with patches')#15

    # MAIN NN Parameters (SR)
    parser.add_argument('-tu', dest = 'type_upsampling', type = str, default="bicubic", help = 'SR upsampling for making LR out of HR') 
    parser.add_argument('-td', dest = 'type_downsampling', type = str, default="automatic", help = ' Downsampling used for making LR out of HR (automatic : take data from div2K folders / bicubic ; bilinear : downsampling method)') 
    parser.add_argument('-bd', dest = 'bluring_downsampling', type = float, default=2, help = ' Variance of the kernel used before downsampling ') 
    
    args = parser.parse_args()
        
    assert  args.type_upsampling=="bicubic" or args.type_upsampling=="bilinear", " Veuillez choisir une méthode de upsampling adaptée : 'bicubic' ou'bilinear'  pour l'argument 'type_upsampling' "

    assert  args.type_downsampling=="bicubic" or args.type_downsampling=="bilinear" or args.type_downsampling=="automatic", " Veuillez choisir une méthode de downsampling adaptée : 'bicubic', 'bilinear' ou 'Automatic' (si les données ont déjà des versions LR)  pour l'argument 'type_downsampling' "
    
    information_data=data_preparation_sr(type_upsampling=args.type_upsampling, type_downsampling=args.type_downsampling,bluring_downsampling=args.bluring_downsampling, root_folder = args.root_folder, R = args.upfactor, size = args.taille, delta = args.delta, seuil=args.seuil,border=args.border)
