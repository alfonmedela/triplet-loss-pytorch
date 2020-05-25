from fastai.vision import *
import torch
from loss_functions.triplet_loss import TripletLoss

class L2_norm(nn.Module):
    def __init__(self):
        super(L2_norm, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

if __name__ == '__main__':

    # choose cuda or cpu device
    device = 0
    torch.cuda.set_device(device)

    # dataloader
    path = ''
    data = (ImageList.from_folder(path)
          .split_by_rand_pct(0.2)
          .label_from_folder()
          .transform(tfms, size=sz, padding_mode='reflection')
          .databunch(num_workers=4, bs=bs)
          .normalize(imagenet_stats)
          )

    # define learning, with the data, model and metrics
    learn = cnn_learner(data, models.resnet34)

    # add L2 norm layer on top of final embedding
    layers = learn.model[1]
    learn.model[1] = nn.Sequential(layers[0], layers[1], layers[2], layers[3], layers[4], L2_norm()).to(device)
    print(learn.model)

    # use triplet loss_functions
    learn.loss_func = TripletLoss(device)



