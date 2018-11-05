# DeepCatnDogRecognition

It's all about being a:

`0`: Cat <br>
`1`: Dog <br>

# Introduction

This project proposes three differents Deep Neural Network (DNN) to face the well known *Kaggle Dog and Cat Classification* challenge. You can get more details about it at https://www.kaggle.com/c/dogs-vs-cats.

The DNN here used for the purpose of beating the challenge are the following:

- `LeNet`
- `DeepConvNet`
- `AlexNet` (pre-trained model)

You can find more information about these DNN here: https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5

Each of these DNN is really sensitive to several hyperparameters, which in turn provide a better or worse fine tuning.

# Requirements

  - python 2.7 <br>
  - tensorflow 1.9.0 or greater <br>
  - opencv <br>
  
# Usage

First of all, download the repository by running:

`git clone https://github.com/DavideNardone/DeepCatnDogRecognition.git` <br>

`unzip DeepCatnDogRecognition-master.py`

Before training or testing any of the DNN available in this project, you must create some directories under the root project
by running the following commands:

`mkdir plots` <br>
`mkdir results` <br>
`mkdir ckpt` <br>
`mkdir npys` <br>

Once you've done this, you might need to set up some paths, located in the following python files:

`src/train_net`
`src/configs`
`src/create_test_submission`
`tools/loader`

To train the network run:

`python train_net.py <MODEL>`

where the `MODEL` parameter specify the kind of DNN you want to train (e.g., LE-NET, DEEP-CONVNET, ALEX-NET)

The latter network is a pre-trained network and it works by loading its proper weights which can be downloaded at the following link: <br>
http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ that need to be located at the root of the project folder.

To create the submission test useful for classifying the test set, run the following command:

`python create_test_submission.py <MODEL>`

# Dataset

The dataset used for train and test the DNN can be downloaded at the following link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

Once you've downloaded it, you must unzip and locate it under the root project in order to make everything run smoothly.

# Results

The models accuracy/loss achieved both on train and validation are shown below:

**TRAINING_SET**
 
 **EPOCHES: ~98** <br> 
`LE-NET` <br>
+/- 0.78 ACC  <br>
+/- 0.5 LOSS <br>

 **EPOCHES: ~20** <br> 
`DEEP-CONVNET` <br>
+/- 0.86 ACC <br>
+/- 0.4 LOSS <br>

 **EPOCHES: ~15** <br> 
`ALEX-NET` <br>
+/- 0.93 ACC <br>
+/- 0.9 LOSS <br>

**VALIDATION SET**

`LE-NET` <br>
+/- 0.70 ACC <br>
+/- 0.5 LOSS <br>

`DEEP-CONVNET` <br>
+/- 0.93 ACC <br>
+/- 0.2 LOSS <br>

`ALEX-NET` <br>
+/- 0.9 ACC <br>
+/- 0.9 LOSS <br>

NB: As you might know, DNN are ruled by several hyperparameters that in turn affect the fine tuning process. The results presented here might not be the best one since a better fine tuning process and a more efficient image pre-processing as well might led to better results.

# Troubleshootings

In case something goes wrong, make sure:

1. You've created all the required directories
2. Any configuration path is rightly set up
3. You've installed the right library versions

# Authors

Davide Nardone, University of Naples Parthenope, Science and Techonlogies Departement,<br> Msc Applied Computer Science <br/>
https://www.linkedin.com/in/davide-nardone-127428102

# Contacts

For any kind of problem, questions, ideas or suggestions, please don't esitate to contact me at: 
- **davide.nardone@live.it**
