# DeepCatnDogRecognition

`1`: Dog
'0': Cat

# Introduction

This project propose 3 differents Deep Neural Network (DNN) to face the Kaggle Dog and Cat Classification challenge, where you can give a look here https://www.kaggle.com/c/dogs-vs-cats.

The DNN here used for the purpose of beating the challes are the followings:

- `LeNet`
- `DeepConvNet`
- `AlexNet` (pre-trained model)

Each of these DNN is sensible to several hyperparameters, which in turn provide a better or worse finetuning.

# Requirements

  - python 2.7 or greater <br>
  - tensorflow <br>
  - opencv <br>
  
# Usage

`git clone https://github.com/DavideNardone/PySMRS.git` <br>

`unzip PySMRS-master.py`

Before training or testing any of DNN available in this project, you must create some directories, running the following commands:

`mkdir plots`
`mkdir results`
`mkdir ckpt`

1. To train the network run:

`python train_net.py <MODEL>`

The `MODEL` parameter specify the kind of DNN you want to train, that is:



The latter network is a pre-trained network. It works by loading its proper weights which can be downloaded at the following link: <br>
http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ that need to be located at the root of the project folder.

2. To reproduce the submission for 



# Dataset

The dataset used for train and test the NN can be retrieved at the following link: <br>
https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

Once downloaded it, you must unzip it running the following command:

`unzip`...

NB: The uncompressed folder must be located under the root project in order to have the project working or differently you may need to change some configuration paths.

# Authors

Davide Nardone, University of Naples Parthenope, Science and Techonlogies Departement,<br> Msc Applied Computer Science <br/>
https://www.linkedin.com/in/davide-nardone-127428102

# Contacts

For any kind of problem, questions, ideas or suggestions, please don't esitate to contact me at: 
- **davide.nardone@live.it**
