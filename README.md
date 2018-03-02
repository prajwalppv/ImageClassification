# ImageClassification
Image classification on the CIFAR 10 dataset using CNN - 80% Accuracy

Python pre-requisites:-
1) Python 3.5+
2) Tensorflow 1.5+

Porting project to Python 2.7
The code can also be ported to python 2.7. Changes that need to be made are:
  1) In all files, import _pickle must be changed to import cPickle
  2) Also take care about loading the dataset with the correct encoding.
  
How to run the code:-

1) Download the cifar-10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html
2) In the scipts folder, set data path argument to the path of the downloaded dataset
3) All the python files are present in the "src" folder. Make sure the path the main.py file is specified correctly in the bash scipt.
4) Run the required bash script- 1) cifar-10.sh runs a basic softmax classifier
                                 2) feed_forward.sh runs a simple MLP which gives ~ 50% accuracy on cifar-10 
                                 3) conv_net.sh runs the CNN algorithm which produced 80% accuracy on cifar-10

How to customize your model:-

1) Open the models.py file in "src" folder.
2) Find the function of the model you wish to customize.
3) Make changes as required and run as before.

Outputs:-
1) The code logs validation accuracy and train loss every 100 steps of training and finally evaluates accuracy on the test dataset.
2) The code automatically saves the model after every 10000 steps of training. This can be configured as required.


 
