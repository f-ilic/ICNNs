import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import torch.nn.functional as F
import tkinter as tk

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def num_to_one_hot(label, num_classes):
    current_batch_size = len(label)
    labels_one_hot = torch.zeros(current_batch_size, num_classes)
    labels_one_hot.scatter_(1, label.unsqueeze(1), 1.)
    return labels_one_hot


def img_scatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()

    im = OffsetImage(image, zoom=zoom, cmap="gray")
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def plot_latent_space_sampling(model):
    num_grid = 32
    x,y = np.meshgrid(np.linspace(-8,12,num_grid), np.linspace(-4,16,num_grid), indexing='ij')
    fig, ax = plt.subplots()

    for i in range(num_grid):
        for j in range(num_grid):
            latent = [x[i,j], y[i,j]]
            latent = torch.Tensor(latent)
            decoded = model.decode(latent.view(1,-1)).detach()
            image = decoded.view(28, 28)
            image = np.array(image.data.detach().numpy())
            img_scatter(x[i,j], y[i,j], image,zoom=0.5, ax=ax)
            ax.plot(x[i,j], y[i,j])
    plt.show()
    return fig, ax


def fiddle_in_latent_vector(model, minimap=None):
    LATENT_DIMS = model.latent_space_dims

    def encode():
        output = model.encode(image.view(1, 1, 28, 28))
        print(output.view(-1))
        outnp = output.detach().flatten()

        for num, btn in enumerate(button_identities):
            btn.set(outnp[num].item())


    def decode():
        latent = [float(button_identities[x].get()) for x in range(LATENT_DIMS)]

        if minimap != None:
            minimap.scatter(latent[0], latent[1])

        latent = torch.Tensor(latent)

        print(latent)

        decoded = model.decode(latent.view(1,-1)).detach()
        plt.figure(3)
        plt.imshow(decoded.view(28, 28), cmap="gray")
        plt.draw()

    def slide_release(self):
        decode()

    def load_rand_img():
        global image
        image, label = [x[0] for x in iter(test_loader).next()]
        plt.figure(2)
        plt.imshow(image.view(28, 28), cmap="gray")
        plt.draw()

    master = tk.Tk()
    button_identities = []

    for i in range(LATENT_DIMS):
        button = tk.Scale(master, from_=-10., to=10., orient=tk.HORIZONTAL, tickinterval=1000, length=500, digits=3,
                       resolution=0.01)

        button.pack()
        button.bind("<ButtonRelease-1>", slide_release)
        button_identities.append(button)

    tk.Button(master, text="Load New Random Image", command=load_rand_img).pack()
    tk.Button(master, text="Encode", command=encode).pack()
    tk.Button(master, text="Decode", command=decode).pack()

    plt.figure(1)
    plt.figure(2)
    plt.show()
    tk.mainloop()