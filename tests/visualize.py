import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.widgets import Slider

import numpy as np

import imageio
import time

def compare_slice_window(left, right, title):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title, fontsize = 20)
    plt.subplots_adjust(bottom=0.3)

    lmax, lmin, rmax, rmin = left.max(), left.min(), right.max(), right.min()
    print("left  > max =", lmax, ", min =", lmin)
    print("right > max =", rmax, ", min =", rmin)

    xslice = 0
    yslice = 0

    ax1.set_title("FFT")
    ax2.set_title("Render en frecuencias")
    
    ax1.plot(np.arange(left.shape[2]), left[xslice, yslice, :], '-b')
    ax2.plot(np.arange(right.shape[2]), right[xslice, yslice, :], '-r')

    def update_x_plot(val):
        xslice = int(sliderx.val)
        ax1.cla()
        ax2.cla()
        ax1.plot(np.arange(left.shape[2]), left[xslice, yslice, :], '-b')
        ax2.plot(np.arange(right.shape[2]), right[xslice, yslice, :], '-r')
        fig.canvas.draw_idle()

    def update_y_plot(val):
        yslice = int(slidery.val)
        ax1.cla()
        ax2.cla()
        ax1.plot(np.arange(left.shape[2]), left[xslice, yslice, :], '-b')
        ax2.plot(np.arange(right.shape[2]), right[xslice, yslice, :], '-r')
        fig.canvas.draw_idle()

    # Sliders
    ax_x = plt.axes([0.25, 0.15, 0.5, 0.03])
    ax_y = plt.axes([0.25, 0.05, 0.5, 0.03])

    sliderx = Slider(ax_x, 'X slice', 0, right.shape[0] - 1, valinit=0, valfmt='%d')
    sliderx.on_changed(update_x_plot)

    slidery = Slider(ax_y, 'Y slice', 0, right.shape[1] - 1, valinit=0, valfmt='%d')
    slidery.on_changed(update_y_plot)

    plt.show()

def compare_window(left, right, labels, slider_label, title, colormap = cm.hot, bounds=None):

    fig, axes = plt.subplots(1,2)
    fig.suptitle(title, fontsize = 20)
    plt.subplots_adjust(bottom=0.3)

    lmax, lmin, rmax, rmin = left.max(), left.min(), right.max(), right.min()

    print("left  > max =", lmax, ", min =", lmin)
    print("right > max =", rmax, ", min =", rmin)

    lv, hv = None, None
    if bounds is not None:
        lv, hv = bounds
        lmin = lv
        rmin = lv
        lmax = hv
        rmax = hv
    else:
        left = colormap(left)
        right = colormap(right)

    axes[0].set_title("FFT")
    axes[1].set_title("Render en frecuencias")

    print("current label:", labels[0])

    # create dummy invisible image, colorbar trick
    img1 = plt.imshow(np.array([[lmin,lmax]]), cmap=colormap)
    img1.set_visible(False)
    plt.colorbar(img1, ax=axes[0])

    img2 = plt.imshow(np.array([[rmin,rmax]]), cmap=colormap)
    img2.set_visible(False)
    plt.colorbar(img2, ax=axes[1])

    pos1 = axes[0].imshow(left[:, :, 0], vmin=lv, vmax=hv, cmap=colormap if bounds is not None else None)
    pos2 = axes[1].imshow(right[:, :, 0], vmin=lv, vmax=hv, cmap=colormap if bounds is not None else None)

    def update_plot(val):
        idx = int(sliderwave.val)
        # dirty trick for removing last line
        print ("\033[A                                           \033[A")
        print("current label:", labels[idx])
        axes[0].cla()
        axes[1].cla()
        axes[0].set_title("FFT")
        axes[1].set_title("Render en frecuencias")
        pos1 = axes[0].imshow(left[:, :, idx], vmin=lv, vmax=hv, cmap=colormap if bounds is not None else None)
        pos2 = axes[1].imshow(right[:, :, idx], vmin=lv, vmax=hv, cmap=colormap if bounds is not None else None)
        fig.canvas.draw_idle()

    # Sliders
    axwave = plt.axes([0.25, 0.05, 0.5, 0.03])

    sliderwave = Slider(axwave, slider_label, 0, right.shape[2] - 1, valinit=0, valfmt='%d')
    sliderwave.on_changed(update_plot)

    plt.show()