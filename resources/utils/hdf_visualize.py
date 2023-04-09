import numpy as np
import h5py
import argparse
import tonemapper

import matplotlib.pyplot as plt
from matplotlib import cm

def main(args):
    f = h5py.File(args.infile, 'r')

    print(f['hdf5'].shape)
    data = np.nan_to_num(f['hdf5'][:, :, :3]) # drop alpha

    print("max :", np.amax(data))
    print("min :", np.amin(data))

    data = np.where(data < 0, 0, data)

    if ( args.color ):
        plt.imsave("converted-color.png", tonemapper.tonemap_gamma(data))
    elif ( args.heatmap ):
        plt.imsave("heatmap-red.png", cm.hot(data[:, :, 0]))
        plt.imsave("heatmap-green.png", cm.hot(data[:, :, 1]))
        plt.imsave("heatmap-blue.png", cm.hot(data[:, :, 2]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize HDF data as RGB image")

    parser.add_argument("-i", "--infile", type=str, help="The path to the HDF5 directory.")
    parser.add_argument("-d", "--dataset", type=str, help="The dataset to load. Defaults to 'hdf5'.", default="hdf5")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--color", action="store_true", help="show colour")
    group.add_argument("--heatmap", action="store_true", help="Show intensity of each channel as a colormap.")
    
    args = parser.parse_args()
    main(args)
