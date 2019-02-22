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
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 2, 3, padding=1),
            nn.ReLU(True)
        )

        self.lin = nn.Sequential(  # 1500 parameters
            nn.Linear(5*5*2, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.lin(x.view(len(x),-1))
        return x

    def get_activations(self, x):
        a1 = self.conv1(x)
        a2 = self.conv2(a1)
        a3 = self.conv3(a2)
        return (a1,a2,a3)

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



            # if (epoch % 10) == 0:
                # torch.save(self.state_dict(), './classfier.model')

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 10, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Sigmoid()
        )

        self.classifier = classifier()
        self.classifier.load_state_dict(torch.load('./classfier_model'))
        # self.classifier.eval()

    def __str__(self):
        return "Autoencoder"

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR,  weight_decay=1e-5)
        l2_func = nn.MSELoss()
        l1_func = nn.L1Loss()

        for epoch in range(EPOCH):
            print("[" + str(self) + "] Training in epoch " + str(epoch))
            offset = epoch * len(train_loader)

            for step, data in enumerate(train_loader):
                img, _ = data
                img = Variable(img)

                # forward
                reconstructed = self(img)

                # Compute layerwise loss of classifier
                loss1 = l2_func(self.classifier.layer1_activations(img), self.classifier.layer1_activations(reconstructed))
                loss2 = l2_func(self.classifier.layer2_activations(img), self.classifier.layer2_activations(reconstructed))
                loss = loss1 + loss2

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step % 10) == 0:
                    writer.add_images('reconstruction/input', to_img(img[0:128,:,:,:]).view(128, 1, 28, 28).repeat(1, 3, 1, 1), step + offset)
                    writer.add_images('reconstruction/output', to_img(reconstructed[0:128,:,:,:]).view(128, 1, 28, 28).repeat(1, 3, 1, 1), step+ offset)
                    writer.add_scalar('loss/total', loss.item(), step + offset)
                    writer.add_scalar('loss/layer1_loss', loss1.item(), step + offset)
                    writer.add_scalar('loss/layer2_loss', loss2.item(), step + offset)


if __name__ == "__main__":
    model = classifier()
    model.train()
