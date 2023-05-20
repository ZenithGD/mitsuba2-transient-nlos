import argparse
import glob
import os
from pathlib import Path

from matplotlib import cm
from matplotlib import colors
from matplotlib.widgets import Slider

import imageio
import time

import cv2 as cv
import mitsuba

from tonemapper import *
from tqdm import tqdm

import h5py

def read_streakimg(dir: str, extension: str = "exr") -> np.array:
    """
    Reads all the images x-t that compose the streak image.

    :param dir: path where the images x-t are stored
    :param extension: of the images x-t
    :return: a streak image of shape [height, width, time, nchannels]
    """
    number_of_xtframes = len(glob.glob(f"{dir}/frame_*.{extension}"))
    fileList = []
    for i_xtframe in range(number_of_xtframes):
        img = imageio.imread(f"{dir}/frame_{i_xtframe}.{extension}")
        fileList.append(np.expand_dims(img, axis=0))

    streak_img = np.concatenate(fileList)
    return np.nan_to_num(streak_img, nan=0.)

def read_streakimg_hdf5(dir: str) -> np.array:
    """
    Reads all the images x-t that compose the streak image. It assumes that the format is HDF5.

    :param dir: path where the images x-t are stored
    :param extension: of the images x-t
    :return: a streak image of shape [height, width, time, nchannels]
    """
    number_of_xtframes = len(glob.glob(f"{dir}/frame_*.hdf5"))
    fileList = []
    with tqdm(total=number_of_xtframes, ascii=True) as pbar:
        for i_xtframe in range(number_of_xtframes):
            
            f = h5py.File(f"{dir}/frame_{i_xtframe}.hdf5", 'r')

            data = np.nan_to_num(f['hdf5'][:, :, :3]) # drop alpha
            fileList.append(np.expand_dims(data, axis=0))
            pbar.update(1)

    streak_img = np.concatenate(fileList)
    return np.nan_to_num(streak_img, nan=0.)


def read_streakimg_mitsuba(dir_path: str, extension: str = "exr") -> np.array:
    """
    Reads all the images x-t that compose the streak image.

    :param dir: path where the images x-t are stored
    :param extension: of the images x-t
    :return: a streak image of shape [height, width, time, nchannels]
    """
    from mitsuba.core import Bitmap, Struct, float_dtype
    number_of_xtframes = len(glob.glob(os.path.join(glob.escape(dir_path), f'frame_*.{extension}')))
    first_img = np.array(Bitmap(f"{dir_path}/frame_0.{extension}"), copy=False)
    streak_img = np.empty((number_of_xtframes, *first_img.shape), dtype=first_img.dtype)
    with tqdm(total=number_of_xtframes, ascii=True) as pbar:
        for i_xtframe in range(number_of_xtframes):
            other = Bitmap(f"{dir_path}/frame_{i_xtframe}.{extension}")
            #     .convert(Bitmap.PixelFormat.RGBA, Struct.Type.Float32, srgb_gamma=False)
            streak_img[i_xtframe] = np.nan_to_num(np.array(other, copy=False), nan=0.)
            pbar.update(1)

    return streak_img

def diff_images(img1: np.array, img2: np.array):
    """
    Shows the difference in values between both images.
    :param img1:
    :param img2:
    :return:
    """
    diff = img1 - img2
    show_image(diff)


def maxmin(img: np.array):
    if len(img.shape) == 3:
        axis = (0,1)
    else:
        axis = (0,1,2)

    maximum = np.amax(img, axis=axis)
    minimum = np.amin(img, axis=axis)
    print(f"Max: {str(maximum)} - Min: {str(minimum)}")


def write_video_custom(streakimg_ldr: np.array, filename: str):
    """
    Creates a video from a HDR streak image (dimensions [height, width, time, channels]) in RGB format. The tonemap is
    needed to transform the HDR streak image to a LDR streak image.

    :param streakimg_hdr:
    :param tonemap:
    """
    number_of_frames = streakimg_ldr.shape[2]

    st = cm.seismic(streakimg_ldr)
    
    # 1. Get the streak image (already done) and define the output
    writer = imageio.get_writer(filename + ".mp4", fps=10)
    # 2. Iterate over the streak img frames
    with tqdm(total=number_of_frames, ascii=True) as pbar:
        for i in range(number_of_frames):
            writer.append_data((st[:, :, i] * 255.0).astype(np.uint8))
            pbar.update(1)
    # 3. Write the video
    writer.close()


def write_frames(streakimg_ldr: np.array, folder: str):
    """
    Writes the frames separately of a HDR streak image.

    :param streakimg:
    :param filename:
    :return:
    """
    Path(folder).mkdir(parents=True, exist_ok=True)
    number_of_frames = streakimg.shape[1]
    # 2. Iterate over the streak img frames
    for i in range(number_of_frames):
        frame = (cm.hot(streakimg_ldr[:, i, :]) * 255).astype(np.uint8)
        imageio.imwrite(folder + f"/frame_{str(i)}.png", frame)
        print(f"{i}/{number_of_frames}", end="\r")

def load_streakimg(path, ext):
    streakimg = None
    if ext != "hdf5":
        streakimg = read_streakimg_mitsuba(path, extension=ext)
    else:
        streakimg = read_streakimg_hdf5(path)

    return streakimg