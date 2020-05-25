# Triplet SemiHardLoss
PyTorch semi hard [triplet loss](https://arxiv.org/pdf/1503.03832.pdf). Based on tensorflow addons version that can be found [here](https://www.tensorflow.org/addons/tutorials/losses_triplet). 

The triplet loss is a great choice for classification problems with *N_CLASSES >> N_SAMPLES_PER_CLASS*. For example, face recognition problems. 
<br/><br/>
<img src="https://user-images.githubusercontent.com/18154355/61485418-1cbb1f00-a96f-11e9-8de8-3c46eef5a7dc.png" width="400" height="178" />

The CNN architecture we use with triplet loss needs to be cut off before the classification layer. In addition, a L2 normalization layer has to be added. 
<br/><br/>
<img src="https://user-images.githubusercontent.com/18154355/61485417-1cbb1f00-a96f-11e9-8d6a-94964ce8c4db.png" width="400" height="178" />

## TBD
Soon I will include an example with MNIST in the "main_train.py"
