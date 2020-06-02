# Triplet SemiHardLoss
PyTorch semi hard [triplet loss](https://arxiv.org/pdf/1503.03832.pdf). Based on tensorflow addons version that can be found [here](https://www.tensorflow.org/addons/tutorials/losses_triplet). There is no need to create a siamese architecture with this implementation, it is as simple as following *main_train.py* cnn creation process!

The triplet loss is a great choice for classification problems with *N_CLASSES >> N_SAMPLES_PER_CLASS*. For example, face recognition problems. 
<br/><br/>
<img src="https://user-images.githubusercontent.com/18154355/61485418-1cbb1f00-a96f-11e9-8de8-3c46eef5a7dc.png" width="400" height="178" />

The CNN architecture we use with triplet loss needs to be cut off before the classification layer. In addition, a L2 normalization layer has to be added. 
<br/><br/>
<img src="https://user-images.githubusercontent.com/18154355/61485417-1cbb1f00-a96f-11e9-8d6a-94964ce8c4db.png" width="406" height="158" />

## Results on MNIST
I tested the triplet loss on the MNIST dataset. We can't compare directly to TF addons as I didn't run the experiment but this could be interesting from the point of view of performance. Here are the training logs if you want to compare results. Accuracy is not relevant and shouldn't be there as we are not training a classification model.

### Phase 1

<img src="https://github.com/alfonmedela/TripletSemiHardLoss-PyTorch/blob/master/figures/freezed.PNG" width="400" height="200" />
![freezed](https://github.com/alfonmedela/TripletSemiHardLoss-PyTorch/blob/master/figures/freezed.PNG)

### Phase 2
![unfreezed](https://github.com/alfonmedela/TripletSemiHardLoss-PyTorch/blob/master/figures/unfreezed.PNG)
