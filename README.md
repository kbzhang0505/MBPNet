# MBPNet

We provide code for the Paper:  Multi-Branch and Progressive Network for Low-Light Image Enhancement    
Journal: IEEE Transactions on Image Processing  
Publication Date: 2023  
Volume: 32  
On Page(s): 2295-2308   
Print ISSN: 1057-7149   
Online ISSN: 1941-0042    
Digital Object Identifier: 10.1109/TIP.2023.3266171 

  Low-light images incur several complicated degradation factors such as poor brightness, low contrast, color degradation, and noise. Most previous deep learning-based approaches, however, only learn the mapping relationship of single channel between the input low-light images and the expected normal-light images, which is insufficient enough to deal with low-light images captured under uncertain imaging environment. Moreover, too deeper network architecture is not conducive to recover low-light images due to extremely low values in pixels. To surmount aforementioned issues, in this paper we propose a novel multi-branch and progressive network (MBPNet) for low-light image enhancement. To be more specific, the proposed MBPNet is comprised of four different branches which build the mapping relationship at different scales. The followed fusion is performed on the outputs obtained from four different branches for the final enhanced image. Furthermore, to better handle the difficulty of delivering structural information of low-light images with low values in pixels, a progressive enhancement strategy is applied in the proposed method, where four convolutional long short-term memory networks (LSTM) are embedded in four branches and an recurrent network architecture is developed to iteratively perform the enhancement process. In addition, a joint loss function consisting of the pixel loss, the multi-scale perceptual loss, the adversarial loss, the gradient loss, and the color loss is framed to optimize the model parameters. To evaluate the effectiveness of proposed MBPNet, three popularly used benchmark databases are used for both quantitative and qualitative assessments. The experimental results confirm that the proposed MBPNet obviously outperforms other state-of-the-art approaches in terms of quantitative and qualitative results.

![20230427214203](https://user-images.githubusercontent.com/97494153/234880605-20718611-fa59-49eb-9408-6e447fa9e7b3.png)  
Architecture of proposed MBPNet.
![2](https://user-images.githubusercontent.com/97494153/234884818-9153afbe-6bf1-4703-968f-c7d118fcc6f0.png) 
Illustration of progressive network in the proposed MBPNet.

### Install ###
- python>=3.6
- torch>=1.10
- torchvision>=0.11
- opencv-python>=4.5
- scikit-image==0.17.2
- lpips==0.1.4
- pillow>=8.4.0

### Dataset ###
You can refer to the following links to download the datasets
- [LOL](https://daooshee.github.io/BMVC2018website/)
- [SYS] comes from [LOL](https://daooshee.github.io/BMVC2018website/)
- [MIT](https://data.csail.mit.edu/graphics/fivek/)


### dataset Structure ###
//For training and testing data, we recommend this structure.    

//testhigh and testlow store low-light images and labels for testing, respectively, trainhigh and trainlow store low-light images and labels for training, respectively.    

//We recommend using the top 97% of the dataset for training, and the last 3% for testing   

*├─datasets    
**└─training_data   
***├─testhigh    
***├─testlow   
***├─trainhigh   
***└─trainlow    
    

### Train ###
All log files during training will be saved to `./checkpoints`.

*First, prepare training data and test data as suggested by '### dataset Structure ###'

**Second, you can check '''./options/base_options.py''' to set appropriate training configuration

***Lastly, You can run '''train.py'''


### Test ###
All test results will be saved to `./results`.

*First, Put the low-light image and label image into './datasets/training_data/testlow' and './datasets/training_data/testhigh' ,
#To accommodate unpaired test data, our code can allow for label images not present#

** Then, You can run '''test.py'''


### Citation ###
If you find this code and data useful, please consider citing citing our paper:

```
@ARTICLE{10102793,
  author={Zhang, Kaibing and Yuan, Cheng and Li, Jie and Gao, Xinbo and Li, Minqi},
  journal={IEEE Transactions on Image Processing}, 
  title={Multi-Branch and Progressive Network for Low-Light Image Enhancement}, 
  year={2023},
  volume={32},
  number={Apr.},
  pages={2295-2308},
  doi={10.1109/TIP.2023.3266171}}
```



