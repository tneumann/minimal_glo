import plac
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as fnn
from torch.autograd import Variable
from torch.optim import SGD
from torchvision.datasets import LSUN
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid



def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w), 
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = fnn.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return fnn.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = fnn.avg_pool2d(filtered, 2)

    pyr.append(current)
    return pyr


class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None
        
    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(
                size=self.k_size, sigma=self.sigma, 
                n_channels=input.shape[1], cuda=input.is_cuda
            )
        pyr_input  = laplacian_pyramid( input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(fnn.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))


class IndexedDataset(Dataset):
    """ 
    Wraps another dataset to sample from. Returns the sampled indices during iteration.
    In other words, instead of producing (X, y) it produces (X, y, idx)
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return (img, label, idx)


class Generator(nn.Module):
    def __init__(self, code_dim, n_filter=64, out_channels=3):
        super(Generator, self).__init__()
        self.code_dim = code_dim
        nf = n_filter
        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(code_dim, nf * 8, 4, 1, 0, bias=False), # 2x2
            nn.BatchNorm2d(nf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False), # 4x4
            nn.BatchNorm2d(nf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False), # 8x8
            nn.BatchNorm2d(nf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf    , 4, 2, 1, bias=False), # 16x16
            nn.BatchNorm2d(nf), nn.ReLU(True),
            nn.ConvTranspose2d(nf, out_channels, 4, 2, 1, bias=False), # 32x32
            nn.Tanh(),
        )

    def forward(self, code):
        return self.dcnn(code.view(code.size(0), self.code_dim, 1, 1))


def project_l2_ball(z):
    """ project the vectors in z onto the l2 unit norm ball"""
    return z / np.maximum(np.sqrt(np.sum(z**2, axis=1))[:, np.newaxis], 1)


def imsave(filename, array):
    im = Image.fromarray((array * 255).astype(np.uint8))
    im.save(filename)


def main(
        lsun_data_dir: ('Base directory for the LSUN data'),
        image_output_prefix: ('Prefix for image output', 
                              'option', 'o')='glo',
        code_dim: ('Dimensionality of latent representation space', 
                   'option', 'd', int)=128, 
        epochs: ('Number of epochs to train', 
                 'option', 'e', int)=25,
        use_cuda: ('Use GPU?', 
                   'flag', 'gpu')=False,
        batch_size: ('Batch size', 
                     'option', 'b', int)=128,
        lr_g: ('Learning rate for generator', 
               'option', None, float)=1.,
        lr_z: ('Learning rate for representation_space', 
               'option', None, float)=10.,
        max_num_samples: ('Cap on the number of samples from the LSUN dataset', 
                          'option', 'n', int)=-1,
        init: ('Initialization strategy for latent represetation vectors', 
               'option', 'i', str, ['pca', 'random'])='pca',
        n_pca: ('Number of samples to take for PCA',
                'option', None, int)=(64 * 64 * 3 * 2),
        loss: ('Loss type (Laplacian loss as in the paper, or L2 loss)',
               'option', 'l', str, ['lap_l1', 'l2'])='lap_l1',
):

    def maybe_cuda(tensor):
        return tensor.cuda() if use_cuda else tensor

    train_set = IndexedDataset(
        LSUN(lsun_data_dir, classes=['bedroom_train'], 
             transform=transforms.Compose([
                 transforms.Resize(64),
                 transforms.CenterCrop(64),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             ]))
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, 
        shuffle=True, drop_last=True,
        num_workers=8, pin_memory=use_cuda,
    )
    # we don't really have a validation set here, but for visualization let us 
    # just take the first couple images from the dataset
    val_loader = torch.utils.data.DataLoader(train_set, shuffle=False, batch_size=8*8)

    if max_num_samples > 0:
        train_set.base.length = max_num_samples
        train_set.base.indices = [max_num_samples]

    # initialize representation space:
    if init == 'pca':
        from sklearn.decomposition import PCA

        # first, take a subset of train set to fit the PCA
        X_pca = np.vstack([
            X.cpu().numpy().reshape(len(X), -1)
            for i, (X, _, _)
             in zip(tqdm(range(n_pca // train_loader.batch_size), 'collect data for PCA'), 
                    train_loader)
        ])
        print("perform PCA...")
        pca = PCA(n_components=code_dim)
        pca.fit(X_pca)
        # then, initialize latent vectors to the pca projections of the complete dataset
        Z = np.empty((len(train_loader.dataset), code_dim))
        for X, _, idx in tqdm(train_loader, 'pca projection'):
            Z[idx] = pca.transform(X.cpu().numpy().reshape(len(X), -1))

    elif init == 'random':
        Z = np.random.randn(len(train_set), code_dim)

    Z = project_l2_ball(Z)

    g = maybe_cuda(Generator(code_dim))
    loss_fn = LapLoss(max_levels=3) if loss == 'lap_l1' else nn.MSELoss()
    zi = maybe_cuda(torch.zeros((batch_size, code_dim)))
    zi = Variable(zi, requires_grad=True)
    optimizer = SGD([
        {'params': g.parameters(), 'lr': lr_g}, 
        {'params': zi, 'lr': lr_z}
    ])

    Xi_val, _, idx_val = next(iter(val_loader))
    imsave('target.png',
           make_grid(Xi_val.cpu() / 2. + 0.5, nrow=8).numpy().transpose(1, 2, 0))

    for epoch in range(epochs):
        losses = []
        progress = tqdm(total=len(train_loader), desc='epoch % 3d' % epoch)

        for i, (Xi, yi, idx) in enumerate(train_loader):
            Xi = Variable(maybe_cuda(Xi))
            zi.data = maybe_cuda(torch.FloatTensor(Z[idx.numpy()]))

            optimizer.zero_grad()
            rec = g(zi)
            loss = loss_fn(rec, Xi)
            loss.backward()
            optimizer.step()

            Z[idx.numpy()] = project_l2_ball(zi.data.cpu().numpy())

            losses.append(loss.data[0])
            progress.set_postfix({'loss': np.mean(losses[-100:])})
            progress.update()

        progress.close()

        # visualize reconstructions
        rec = g(Variable(maybe_cuda(torch.FloatTensor(Z[idx_val.numpy()]))))
        imsave('%s_rec_epoch_%03d.png' % (image_output_prefix, epoch), 
               make_grid(rec.data.cpu() / 2. + 0.5, nrow=8).numpy().transpose(1, 2, 0))

if __name__ == "__main__":
    plac.call(main)

