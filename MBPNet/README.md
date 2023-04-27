### MBPNet ###

We provide code for the Paper:  
Journal: IEEE Transactions on Image Processing
Publication Date: 2023
Volume: 32
On Page(s): 2295-2308
Print ISSN: 1057-7149
Online ISSN: 1941-0042
Digital Object Identifier: 10.1109/TIP.2023.3266171


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
//For training and testing data, we recommend this structure

//testhigh and testlow store low-light images and labels for testing, respectively, trainhigh and trainlow store low-light images and labels for training, respectively.
//We recommend using the top 97% of the dataset for training, and the last 3% for testing

├─datasets
    └─training_data
        ├─testhigh
        ├─testlow
        ├─trainhigh
        └─trainlow


### Train ###
All log files during training will be saved to `./checkpoints`.

* First, prepare training data and test data as suggested by '### dataset Structure ###'

** Second, you can check '''./options/base_options.py''' to set appropriate training configuration

*** Lastly, You can run '''train.py'''


### Test ###
All test results will be saved to `./results`.

* First, Put the low-light image and label image into './datasets/training_data/testlow' and './datasets/training_data/testhigh' ,
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



