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

    x = []
    y = []
    n_class = 0
    for folder in folders:
        images = glob.glob(folder + '/*')
        for image in images:
            name = image.split('/')[-1]
            if name in valid_names:
                img = open_image(image)
                pred = learn.predict(img)
                embedding = pred[-1].detach().numpy()

                x.append(embedding)
                y.append(n_class)

        n_class += 1

    x, y = np.asarray(x), np.asarray(y)
    print(x.shape, y.shape)
    print(np.unique(y))

    x_tsne = TSNE(n_components=2).fit_transform(x)
    print(x_tsne.shape)

    color = ['darkcyan', 'g', 'y', 'magenta', 'lightgreen', 'mediumslateblue', 'yellow', 'brown', 'k', 'b']
    labels = [str(i) for i in range(10)]
    times = [0 for i in range(10)]
    for i in range(len(x_tsne)):
        if times[y[i]] == 0:
            plt.scatter(x_tsne[i, 0], x_tsne[i, 1], color=color[y[i]], label=labels[y[i]])
            times[y[i]] += 1
        else:
            plt.scatter(x_tsne[i, 0], x_tsne[i, 1], color=color[y[i]])
    plt.legend()
    plt.title('TSNE validation')
    plt.savefig('tsne_val.png')




