from __future__ import division
import os

def ensure_dir(path:str)->None:
    """
    make the directory specified in the path
    :param path: path to the directory
    """
    os.makedirs(path, exist_ok=True)
    
def delete_all_npy(images_dir:str):
    """
    Deletes all the images in a folder
    :param images_dir : directory where all the png images are located
    """
    
    for file in os.listdir(images_dir):
        if file.endswith(".npy"):
            file_path = os.path.join(images_dir,file)
            os.remove(file_path)


def delete_all_png(images_dir:str):
    """
    Deletes all the images in a folder
    :param images_dir : directory where all the png images are located
    """
    
    for file in os.listdir(images_dir):
        if file.endswith(".png"):
            file_path = os.path.join(images_dir,file)
            os.remove(file_path)

def delete_all_png_string(images_dir:str,string:str):
    """
    Deletes all the images in a folder
    :param images_dir : directory where all the png images are located
    """
    
    for file in os.listdir(images_dir):
        if file.startswith(string):
            if file.endswith(".png"):
                file_path = os.path.join(images_dir,file)
                os.remove(file_path)
                

    
    
    
    
    
    
    