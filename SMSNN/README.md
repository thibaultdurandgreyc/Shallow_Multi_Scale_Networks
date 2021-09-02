
# Shallow_Multi_Scale_Networks
![teaser_poster_eng](https://user-images.githubusercontent.com/87061912/131678960-06801817-89f8-4dd1-8841-b534effb5ede.png)

## **PRESENTATION**

### **INTRODUCTION**

The goal of this project is to upscale and improve the quality of low resolution images with control over the output image.

This project includes Tensorflow (Keras A.P.I) implementation of Shallow Multi Scale Network for Stylized Super-Resolution (ICIP 2021 / ORASIS 2021) which consists in performing Super-Resolution (_SR_) with parallel branches specialized in stylizing high frequency details. 

### **DIFFERENT NETWORKS**
Also, other Branches and Options are available. We denote :
* **'MAIN NN'** : The main network for performing 'SR' as discussed in the papers, but also 'DENOISING' or 'BLURRING'. The network, as shown in following scheme, is composed by different branches linearly combined to reconstruct the output.
 
_Input_ : Y,cb,cr channels _Output_ : Y channel  

* **'St(y,cbcr)'BRANCH NN** : High frequency style transfer residual branches which transfer details from a Style image to the Main neural network output image through gram matrices. The branch is plugged on the top of the MAIN model ('SR' model Typically as discussed in the papers). The branch is trained on the top of the Main model for which parameters are frozen. Residual output is litteraly added on the top of the Main model Output Y channel. Note that it is possible for the user to generate also Cb & Cr from style transfert (not discussed in the paper).
 
_Input_ : Y,cb,cr channels   _Output_ : Y channel   OR  Y,cb,cr channels
 
* **'St(Col)' BRANCH NN** : Color transfer branch not mentionned in the papers. Transfers color from a Style image to the colors of the Main model output with through an histogram matching loss.

_Input_ : Y,cb,cr channels   _Output_ : cb , cr channels

* **'St3' BRANCH NN** : Style transfer branches not mentionned in the papers. Transfers the Style of an image to the Main model output through gram matrices. The branch is not residual.

_Input_ : Y,cb,cr channels   _Output_ :  Y,cb,cr channels

![1 0_multiple_branches](https://user-images.githubusercontent.com/87061912/131682395-2083a2a8-7f2f-4013-ae2b-ba05640d25bc.png)

### **ARCHITECTURES**

* **'MAIN NN'** : multi-scale network with multiple independant branches. Each branch is specialized into synthetising a band of frequency because of a passband filter at the end of the branch (Difference of gaussians). With the default parameters, contains more or less _200k parameters_

* **'St(y,cbcr)'BRANCH NN and 'St(Col)' BRANCH NN** : The same branch is used, only the number of filters differs ( _St(y,cbcr) < 40k parameters_ ; _St(col) < 70k parameters_ ), consisting in 5 convolutional layers with high frequency passband filter at the end of the branch.

* **'St3' BRANCH NN** : (Jonshon auto-encodeur)

### **EXAMPLES**

Multiple usage of the branch :

* 0. Main model : _patch_example_TODO

* 1. Main model with St(y) (as discussed in papers) : _patch_example_TODO

* 2. Main model with St(y,cbcr) : _patch_example_TODO

* 3. Main model with St(y) **and** St(col) : _patch_example_TODO

* 4. Main model with St(col) : _patch_example_TODO

* 5. Main model with St3 : _patch_example_TODO (style transfer)

## **USE CODE**

### **ENVIRONMENT SETUP**

python : 2.7.17
tensorflow : 2.0.0
keras : 2.2.4

### **DATA**

Experiments are performed on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset. Training & cross validation are performed on patches (224 or 352). Some tests are performed on benchmark data but also test patches.

### **FOLDERS**

_/SMSNN/Data_results_ : folder to store data 

	* /Patchs : Folder for Patches build out of /SMSNN/Data_results/External_Data/ORIGINAL_DATASET/DIV2K
        
	* /Results : Folder (1 per experiment) containing models and test outputs procedures
        
	* /External_Data :  
        
			** /BENCHMARK_VISUEL : contains /Masks & /Rapports used as benchmark for testing procedures
                        
			** /MODEL : model trained externally 
                        
			** /ORIGINAL_DATASET : contains [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset which need to be downloaded (Bicubic_x4)
                        
			** /SELECTED_COLORS ; /SELECTED_STYLE ; /SELECTED_STYLES_ST3 : Styles used for training branches (respectively St(col) ; St(y,cbcr) ; St3)

_/SMSNN/Networks_ : folder containing all the tools to build and compile architecture out of Main architecture, Branches,Loss & Data generators

_/SMSNN/Data_preprocessing_ : folder containing python functions to create patches out of DIV2K images

_/SMSNN/Testing_ : folder containing python test functions

_/SMSNN/Main.py_ & _/SMSNN/Main_datapreprocessing.py_ : Main files (respectivly Training & Testing, and Data preprocessing


### **GETTING STARTED WITH DATA**

Prepare Data out of downloaded div2K dataset;

```console
python3 Main_Preprocessing.py -r .../SMSNN/Data_results
```

### **TRAIN & TEST MAIN MODEL**
You need to make sure you process Training/Testing on Gpu; main_network should be 'SR', 'DENOISING' or 'BLURRING'
```console
(CUDA_VISIBLE_DEVICE=i) python3 Main.py -r .../SMSNN/Data_results -training_main 'True' -testing_main 'True' -main_network 'SR' -style_model 'False' -col_model 'False' -ST3_model 'False'
```

### **TRAIN & TEST BRANCH MODEL**

Training & Testing St(y) on the top of the pretrained 'SR' model (Branch are trained one by one sorting image from /SELECTED_STYLE )

```console
(CUDA_VISIBLE_DEVICE=i) python3 Main.py -r .../SMSNN/Data_results -training_main 'False' -testing_main 'False' -main_network 'SR' -style_model 'True' -col_model 'False' -ST3_model 'False'
```

Training & Testing St(ycbcr) on the top of the pretrained 'SR' model (Branch are trained one by one sorting image from /SELECTED_COLORS ) 

```console
(CUDA_VISIBLE_DEVICE=i) python3 Main.py -r .../SMSNN/Data_results -training_main 'False' -testing_main 'False' -main_network 'SR' -style_model 'False' -col_model 'True' -ST3_model 'False' -cbcr_sty 'True'
```

Training & Testing St(col) on the top of the pretrained 'SR' model (Branch are trained one by one sorting image from /SELECTED_STYLE 
```console
(CUDA_VISIBLE_DEVICE=i) python3 Main.py -r .../SMSNN/Data_results -training_main 'False' -testing_main 'False' -main_network 'SR' -style_model 'False' -col_model 'False' -ST3_model 'True'
```

Training & Testing ST3 on the top of the pretrained 'SR' model (Branch are trained one by one sorting image from /SELECTED_STYLE_ST3 )
```console
(CUDA_VISIBLE_DEVICE=i) python3 Main.py -r .../SMSNN/Data_results -training_main 'False' -testing_main 'False' -main_network 'SR' -style_model 'False' -col_model 'False' -ST3_model 'True'
```

### **MODEL FOLDER**

_TODO_

### **COMBINING BRANCHES**

_TODO_
