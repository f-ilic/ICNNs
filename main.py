import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch.nn.functional as F
import random

writer = SummaryWriter()


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

# test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)/255.

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

#todo torch scatter
def num_to_one_hot(label, num_classes):
    current_batch_size = len(label)
    labels_onehot = torch.zeros(current_batch_size, num_classes)
    labels_onehot.scatter_(1, label.unsqueeze(1), 1.)
    return labels_onehot

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()

        # APPROX ARCH FROM https://codetolight.files.wordpress.com/2017/11/network.png?w=1108
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 20, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(  # 225 parameters
            nn.Conv2d(20, 32, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(  # 225 parameters
            nn.Conv2d(32, 16, 5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 3, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.lin = nn.Sequential(  # 1500 parameters
            nn.Linear(2*2*3, 10),
            nn.Softmax()
        )

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.softmax(self.lin(x.view(len(x), -1)))
        return x

    def forward_without_softmax(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.lin(x.view(len(x), -1))
        return x


    # def get_activations(self, x):
    #     a1 = self.conv1(x)
    #     a2 = self.conv2(a1)
    #     a3 = self.conv3(a2)
    #     return (a1,a2,a3)

    def __str__(self):
        return "Classifier"

    def train(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR,  weight_decay=1e-5)

        for epoch in range(EPOCH):
            print("[" + str(self) + "] Training in epoch " + str(epoch))


            # Test error
            test_correct = 0
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_correct = (100 * test_correct) /float(len(test_loader.dataset))
            writer.add_scalar('error/test', test_correct, epoch)
            print("test error is at " + str(test_correct))


            # Train error
            train_correct = 0
            for data, target in train_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_correct = (100 * train_correct) /float(len(train_loader.dataset))
            writer.add_scalar('error/train', train_correct, epoch)
            print("train error is at " + str(train_correct))

            offset = epoch * len(train_loader)

            for step, data in enumerate(train_loader):
                img, label = data
                img = Variable(img)
                label = Variable(label)

                # forward
                output = self(img)
                loss = F.nll_loss(output, label)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step % 10) == 0:
                    # # Samples
                    inputs = img[0, :, :, :].view(1, 1, 28, 28).repeat(1, 3, 1, 1).clamp(0,1)
                    writer.add_images('sample/input', inputs, step + offset)

                    classified = output[0,:].view(1,1,1,10).repeat(1, 3, 1, 1).clamp(0,1)
                    writer.add_images('sample/classified', classified, step + offset)

                    # Loss
                    writer.add_scalar('loss/MSELoss', loss.item(), step + offset)


            # Activations
            # a1, a2, a3= self.get_activations(img)
            # writer.add_images('activations/a1', a1[0,:,:,:].view(-1,1,12,12).repeat(1, 3, 1, 1), step + offset)
            # writer.add_images('activations/a2', a2[0,:,:,:].view(-1,1,8,8).repeat(1, 3, 1, 1), step + offset)
            # writer.add_images('activations/a3', a3[0,:,:,:].view(-1,1,3,3).repeat(1, 3, 1, 1), step + offset)

            if (epoch % 10) == 0:
                torch.save(self.state_dict(), './classfier_' + str(epoch) + '_model')

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 2, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Sigmoid()
        )

    def __str__(self):
        return "Autoencoder"

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
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

                loss = mse_loss(img, reconstructed)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step % 10) == 0:
                    writer.add_images('reconstruction/input', to_img(img[0:128,:,:,:]).view(128, 1, 28, 28).repeat(1, 3, 1, 1), step + offset)
                    writer.add_images('reconstruction/output', to_img(reconstructed[0:128,:,:,:]).view(128, 1, 28, 28).repeat(1, 3, 1, 1), step+ offset)
                    writer.add_scalar('loss/total', loss.item(), step + offset)

            if (epoch % 10 ) == 0:
                torch.save(self.state_dict(), './autoencoder' + str(epoch) + '_model')

class classfify_decoder(nn.Module):
    def __init__(self):
        super(classfify_decoder, self).__init__()

        self.decoder1 = nn.Sequential(
            nn.Linear(10, 25)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(1, 16, 3, stride=2, padding=3),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )


        self.classifier = classifier()
        self.classifier.load_state_dict(torch.load('./classfier_150_model'))

        for child in self.classifier.children():
            for param in child.parameters():
                param.requires_grad = False

    def __str__(self):
        return "Classifier+Decoder"

    def forward(self, x):
        x = self.classifier.forward_without_softmax(x)
        x = self.decoder1(x)
        x = self.decoder2(x.view(x.size(0), 1, 5,5))
        return x

    def encode(self,x):
        return self.classifier.forward_without_softmax(x)

    def decode(self, x):
        x = self.decoder1(x)
        x = self.decoder2(x.view(x.size(0), 1, 5,5))
        return x

    def train(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR,  weight_decay=1e-5)
        mse = nn.MSELoss()
        # l1_func = nn.L1Loss()

        for epoch in range(EPOCH):
            print("[" + str(self) + "] Training in epoch " + str(epoch))
            offset = epoch * len(train_loader)

            for step, data in enumerate(train_loader):
                img, _ = data
                img = Variable(img)

                # forward
                reconstructed = self(img)

                # Compute layerwise loss of classifier
                loss = mse(img, reconstructed)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step % 10) == 0:
                    writer.add_images('reconstruction/input', to_img(img[0:128,:,:,:]).view(128, 1, 28, 28).repeat(1, 3, 1, 1), step + offset)
                    writer.add_images('reconstruction/output', to_img(reconstructed[0:128,:,:,:]).view(128, 1, 28, 28).repeat(1, 3, 1, 1), step+ offset)
                    writer.add_scalar('loss/total', loss.item(), step + offset)
                    # writer.add_scalar('loss/layer1_loss', loss1.item(), step + offset)
                    # writer.add_scalar('loss/layer2_loss', loss2.item(), step + offset)

    def test(self):
        mse = nn.MSELoss()

        for step, data in enumerate(test_loader):
            img, _ = data
            img = Variable(img)

            # forward
            reconstructed = self(img)

            loss = mse(img, reconstructed)

            size_batch = img.size(0)
            writer.add_images('reconstruction/input', to_img(img[0:size_batch,:,:,:]).view(size_batch, 1, 28, 28).repeat(1, 3, 1, 1), step)
            writer.add_images('reconstruction/output', to_img(reconstructed[0:size_batch,:,:,:]).view(size_batch, 1, 28, 28).repeat(1, 3, 1, 1), step)
            writer.add_scalar('loss/total', loss.item(), step)


if __name__ == "__main__":
    ae = autoencoder()
    ae.load_state_dict(torch.load('./autoencoder_10_model'))

    from tkinter import *

    def encode():
        output = ae.encoder(image.view(1, 1, 28, 28))
        print(output.view(-1))
        outnp = output.detach().flatten()

        for num, btn in enumerate(button_identities):
            btn.set(outnp[num].item())

        plt.imshow(image.view(28,28))
        plt.show()


    def decode_slider(self):
        latent = torch.Tensor([[[[button_identities[0].get(),button_identities[1].get()],[button_identities[2].get(), button_identities[3].get()]],
                                [[button_identities[4].get(), button_identities[5].get()], [button_identities[6].get(), button_identities[7].get()]]]])
        print(latent)
        decoded = ae.decoder(latent).detach()

        plt.imshow(decoded.view(28,28))
        plt.show()

    def decode():
        latent = torch.Tensor([[[[button_identities[0].get(), button_identities[1].get()],
                                 [button_identities[2].get(), button_identities[3].get()]],
                                [[button_identities[4].get(), button_identities[5].get()],
                                 [button_identities[6].get(), button_identities[7].get()]]]])
        print(latent)
        decoded = ae.decoder(latent).detach()

        plt.imshow(decoded.view(28, 28))
        plt.show()


    def load_rand_img():
        global image
        image, label = [x[0] for x in iter(test_loader).next()]
        plt.imshow(image.view(28, 28), cmap="gray")
        plt.show()


    master = Tk()
    button_identities = []

    for i in range(8):
        button = Scale(master, from_=-40., to=40., orient=HORIZONTAL, tickinterval =1000, length=600, digits = 3, resolution=0.01)

        button.pack()
        button.bind("<ButtonRelease-1>", decode_slider)
        button_identities.append(button)

    Button(master, text="Load New Random Image", command=load_rand_img).pack()
    Button(master, text="Encode", command=encode).pack()
    Button(master, text="Decode", command=decode).pack()

    mainloop()
