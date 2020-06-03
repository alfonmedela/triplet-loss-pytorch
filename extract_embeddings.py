from fastai.vision import *
import torch
from loss_functions.triplet_loss import TripletLoss
import glob
from sklearn.manifold.t_sne import TSNE

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
    path = '/mnt/RAID5/users/alfonsomedela/projects/triplet-loss-torch/mnist_data/dataset/'

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
    learn.model[1] = nn.Sequential(layers[0], layers[1], layers[2], layers[3],
                                   nn.Linear(in_features=1024, out_features=128, bias=False), L2_norm()).to(device)
    learn.model_dir = '/home/alfonsomedela/projects/triplet-loss-torch/results/models/'

    #triplet loss
    learn.loss_func = TripletLoss(device)
    learn.load('stage3_weights')


    # load data
    path = '/mnt/RAID5/users/alfonsomedela/projects/triplet-loss-torch/mnist_data/dataset/'
    folders = glob.glob(path + '*')

    valid_names = np.load('mnist_data/val_names.npy')

    x_val = []
    y_val = []
    x_train = []
    y_train = []
    n_class = 0
    for folder in folders:
        images = glob.glob(folder + '/*')
        print(n_class)
        for image in images:
            name = image.split('/')[-1]

            img = open_image(image)
            pred = learn.predict(img)
            embedding = pred[-1].detach().numpy()

            if name in valid_names:
                x_val.append(embedding)
                y_val.append(n_class)
            else:
                x_train.append(embedding)
                y_train.append(n_class)
        n_class += 1

    x_val, y_val = np.asarray(x_val), np.asarray(y_val)
    x_train, y_train = np.asarray(x_train), np.asarray(y_train)

    np.save('x_train', x_train)
    np.save('y_train', y_train)
    np.save('x_val', x_val)
    np.save('y_val', y_val)




