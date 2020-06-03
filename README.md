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
First we train last layer and batch normalization layers, getting close to 0.079 validation loss.
<br/><br/>
<img src="https://github.com/alfonmedela/TripletSemiHardLoss-PyTorch/blob/master/figures/freezed.PNG" width="400" height="200" />

### Phase 2
Finally, unfreezing all the layers it is possible to get close to 0.05 with enough training and hyperparmeter tuning.
<br/><br/>
<img src="https://github.com/alfonmedela/TripletSemiHardLoss-PyTorch/blob/master/figures/unfreezed.PNG" width="400" height="200" />

### Test
In order to test, there are two interesting options, training a classification model on top of the embeddings and plotting the train and test embeddings to see if same categories cluster together. The following figure contains the original 10,000 validation samples.

![TSNE](https://github.com/alfonmedela/TripletSemiHardLoss-PyTorch/blob/master/figures/tsne_val.png)

We get an accuracy around **99.3%** on validation by training a Linear SVM or a simple kNN. This repository is not focused on maximizing this accuracy by tweaking data augmentation, arquitecture and hyperparameters but to provide an effective implementation of triplet loss in torch. For more info on the state-of-the-art results on MNIST check out this amazing [kaggle discussion](https://www.kaggle.com/c/digit-recognizer/discussion/61480)
