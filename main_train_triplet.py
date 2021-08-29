from fastai.vision import *
import torch
from loss_functions.triplet_loss import TripletLoss


class L2_norm(nn.Module):
    def __init__(self):
        super(L2_norm, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)

if __name__ == '__main__':

    # choose cuda or cpu device
    device = 1
    torch.cuda.set_device(device)

    # dataloader
    bs = 128
    sz = 28
    tfms = get_transforms()
    path = '/mnist_dataset/'

    valid_names = np.load('mnist_data/val_names.npy')
    data = (ImageList.from_folder(path)
          .split_by_files(valid_names)
          .label_from_folder()
          .transform(tfms, size=sz, padding_mode='reflection')
          .databunch(num_workers=4, bs=bs)
          .normalize(imagenet_stats)
          )

    print(data)
    
    # this is important, otherwise the triplet loss blows up
    data.valid_dl = data.valid_dl.new(shuffle=True)

    learn = cnn_learner(data, models.resnet34, metrics=accuracy)
    layers = learn.model[1]
    learn.model[1] = nn.Sequential(layers[0], layers[1], layers[2], layers[3], nn.Linear(in_features=1024, out_features=128, bias=False), L2_norm()).to(device)
    learn.model_dir = '/models/'

    #triplet loss
    learn.loss_func = TripletLoss(device)

    # learn.lr_find()
    # fig = learn.recorder.plot(return_fig=True)
    # fig.savefig('lr_figure.png')

    # lr = 5e-2
    # learn.fit_one_cycle(10, slice(lr))
    # learn.save('stage1_weights')

    # learn.load('stage1_weights')
    # learn.unfreeze()

    # learn.lr_find()
    # fig = learn.recorder.plot(return_fig=True)
    # fig.savefig('lr_figure_unfreezed.png')

    # lr = 1e-4
    # learn.fit_one_cycle(10, slice(lr))
    # learn.save('stage2_weights')

    learn.load('stage2_weights')
    learn.unfreeze()
    lr = 1e-4
    learn.fit_one_cycle(10, slice(lr))
    learn.save('stage3_weights')






