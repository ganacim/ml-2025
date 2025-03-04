import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_2d_data_ax(ax, X, y, d=0, auto=False, zoom=1):
    X = X.cpu()
    y = y.cpu()
    ax.scatter(X.numpy()[:, 0], X.numpy()[:, 1], c=y.numpy(), s=20, cmap=plt.cm.Spectral)
    ax.axis("square")
    ax.axis(np.array((-1.1, 1.1, -1.1, 1.1)) * zoom)
    if auto is True:
        ax.set_aspect("equal")
    ax.axis("off")

    _m, _c = 0, ".15"
    ax.axvline(0, ymin=_m, color=_c, lw=1, zorder=0)
    ax.axhline(0, xmin=_m, color=_c, lw=1, zorder=0)


def plot_2d_data(X, y, d=0, auto=False, zoom=1):
    fig, ax = plt.subplots()
    plot_2d_data_ax(ax, X, y, d, auto, zoom)
    # fig.show()


def plot_2d_model_ax(ax, X, y, model, detail=0.01):
    print(model.parameters())
    mesh = np.arange(-1.1, 1.1, detail)
    xx, yy = np.meshgrid(mesh, mesh)
    with torch.no_grad():
        # get models device
        model_device = next(model.parameters()).device
        # create data tensor and send to device
        data = torch.from_numpy(np.vstack((xx.reshape(-1), yy.reshape(-1))).T).float().to(model_device)
        Z = model(data).detach().cpu().numpy()
    Z = np.argmax(Z, axis=1).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.2)
    plot_2d_data_ax(ax, X, y)


def plot_2d_model(X, y, model, detail=0.01):
    fig, ax = plt.subplots()
    plot_2d_model_ax(ax, X, y, model, detail)
    # fig.show()
