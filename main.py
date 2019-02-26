import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from tensorboardX import SummaryWriter
from torch.autograd import Variable
writer = SummaryWriter()

from util import fiddle_in_latent_vector, plot_latent_space_sampling


# Hyper Parameters
EPOCH = 400
BATCH_SIZE = 128
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False

if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
    download=DOWNLOAD_MNIST,
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.latent_space_dims = 2

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(32, 10, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
        )

        self.encode_lin = nn.Linear(10*2*2, self.latent_space_dims)

        self.decode_lin_reshape = nn.Linear(self.latent_space_dims,10*2*2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 32, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def __str__(self):
        return "Autoencoder"

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        x = self.encode_lin(x.view(x.size(0),-1))
        return x

    def decode(self, x):
        x = self.decode_lin_reshape(x)
        x = self.decoder(x.view(x.size(0), 10, 2, 2))
        return x

    def train(self, num_epoch):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR,  weight_decay=1e-5)
        mse_loss = nn.MSELoss()

        for epoch in range(num_epoch):
            print("[" + str(self) + "] Training in epoch " + str(epoch))
            offset = epoch * len(train_loader)

            for step, data in enumerate(train_loader):
                img, _ = data
                img = Variable(img)

                # forward
                reconstructed = self(img)

                loss = mse_loss(img, reconstructed) + mse_loss(img+1, reconstructed+123)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step % 10) == 0:
                    writer.add_images('reconstruction/input',
                                      img[0:128, :, :, :].view(128, 1, 28, 28).repeat(1, 3, 1, 1),
                                      step + offset)

                    writer.add_images('reconstruction/output',
                                      reconstructed[0:128,:,:,:].clamp(0,1).view(128, 1, 28, 28).repeat(1, 3, 1, 1),
                                      step + offset)

                    writer.add_scalar('loss/total',
                                      loss.item(),
                                      step + offset)

            if (epoch % 10) == 0:
                torch.save(self.state_dict(), './autoencoder_' + str(epoch) + '_model')


class variational_autoencoder(nn.Module):
    def __init__(self):
        super(variational_autoencoder, self).__init__()
        self.latent_space_dims = 2

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(32, 16, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
        )

        self.encode_lin = nn.Linear(16*2*2, self.latent_space_dims)

        self.decode_lin_reshape = nn.Linear(self.latent_space_dims,16*2*2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 40, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(40, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def __str__(self):
        return "Variational Autoencoder"

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        x = self.encode_lin(x.view(x.size(0), -1))
        return x

    def decode(self, x):
        x = self.decode_lin_reshape(x)
        x = self.decoder(x.view(x.size(0), 10, 2, 2))
        return x

    def train(self, num_epoch):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR,  weight_decay=1e-5)
        mse_loss = nn.MSELoss()

        for epoch in range(num_epoch):
            print("[" + str(self) + "] Epoch " + str(epoch))
            offset = epoch * len(train_loader)

            for step, data in enumerate(train_loader):
                img, _ = data
                img = Variable(img)

                # forward
                mu = 0.7
                img_latent = self.encode(img)
                reconstructed = self.decode(img_latent)
                reconstructed_latent = self.encode(reconstructed)

                pixel_space_loss = mu * mse_loss(img, reconstructed)
                feature_space_loss = (1-mu) * mse_loss(img_latent, reconstructed_latent)
                loss = pixel_space_loss + feature_space_loss

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step % 10) == 0:
                    writer.add_images('reconstruction/input',
                                      img[0:128, :, :, :].view(128, 1, 28, 28).repeat(1, 3, 1, 1),
                                      step + offset)

                    writer.add_images('reconstruction/output',
                                      reconstructed[0:128,:,:,:].clamp(0,1).view(128, 1, 28, 28).repeat(1, 3, 1, 1),
                                      step + offset)

                    writer.add_scalar('loss/total',
                                      loss.item(),
                                      step + offset)

                    writer.add_scalar('loss/pixel_space',
                                      pixel_space_loss.item(),
                                      step + offset)

                    writer.add_scalar('loss/feature_space',
                                      feature_space_loss.item(),
                                      step + offset)


            if (epoch % 10) == 0:
                torch.save(self.state_dict(), './' + str(self) + '_' + str(epoch))


if __name__ == "__main__":
    # vae = variational_autoencoder()
    vae = variational_autoencoder()
    vae.train(500)
