# DeepCatnDogRecognition

It's all about being a:

`1`: Dog <br>
`0`: Cat <br>

# Introduction

This project proposes three differents Deep Neural Network (DNN) to face the well known *Kaggle Dog and Cat Classification* challenge. You can get more detail about at https://www.kaggle.com/c/dogs-vs-cats.

The DNN here used for the purpose of beating the challenge are the followings:

- `LeNet`
- `DeepConvNet`
- `AlexNet` (pre-trained model)

You can find more information abou these DNN here: https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5

Each of these DNN is really sensible to several hyperparameters, which in turn provide a better or a worse finetuning.

# Requirements

  - python 2.7 <br>
  - tensorflow 1.9.0 <br>
  - opencv <br>
  
# Usage

First of all, download the repository by running:

`git clone https://github.com/DavideNardone/DeepCatnDogRecognition.git` <br>

`unzip DeepCatnDogRecognition-master.py`

Before training or testing any of the DNN available in this project, you must create some directories by running the following commands:

`mkdir plots` <br>
`mkdir results` <br>
`mkdir ckpt` <br>
`mkdir npys` <br>

To train the network run:

`python train_net.py <MODEL>`

where the `MODEL` parameter specify the kind of DNN you want to train (e.g., LE-NET, DEEP-CONVNET, ALEX-NET)

The latter network is a pre-trained network and it works by loading its proper weights which can be downloaded at the following link: <br>
http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ that need to be located at the root of the project folder.

To reproduce the submission test useful for classify the test set, run the following command:

`python create_test_submission.py <MODEL>`

# Dataset

The dataset used for train and test the DNN can be downloaded at the following link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

Once downloaded, you must unzip and locate it under the root project in order to make everything going smooth.

# Troubleshootings

In case something goes wrong, make sure:

1. You've created all the required directories
2. Any configuration path is rightly set-up
3. You've the right library versions

# Authors

Davide Nardone, University of Naples Parthenope, Science and Techonlogies Departement,<br> Msc Applied Computer Science <br/>
https://www.linkedin.com/in/davide-nardone-127428102

# Contacts

For any kind of problem, questions, ideas or suggestions, please don't esitate to contact me at: 
- **davide.nardone@live.it**
