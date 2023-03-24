import argparse
import glob
import os
from pathlib import Path

from matplotlib import cm

import imageio
import time

import cv2 as cv
import mitsuba

from tonemapper import *
from tqdm import tqdm

mitsuba.set_variant("scalar_rgb")

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
    # 1. Get the streak image (already done) and define the output
    writer = imageio.get_writer(filename + ".mp4", fps=10)
    # 2. Iterate over the streak img frames
    with tqdm(total=number_of_frames, ascii=True) as pbar:
        for i in range(number_of_frames):
            writer.append_data((cm.hot(streakimg_ldr[:, :, i]) * 255).astype(np.uint8))
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

def validate(streakimg):
    print("Applying fft to each streak image")
    ini = time.time()
    
    transformed = np.empty(streakimg.shape, dtype=np.complex128)

    # loop through all streak images
    for i in range(streakimg.shape[1]):
        streak = streakimg[:, i, :]

        transformed[:, i, :] = np.fft.fftshift(np.fft.fft(streak, axis=1), axes=1)

    elapsed = time.time() - ini
    print(f"took {elapsed} secs.")

    
    plt.imshow(np.real(transformed[:, 60, :]), cmap='hot')
    plt.savefig("real.png")

    plt.imshow(np.imag(transformed[:, 60, :]), cmap='hot')
    plt.savefig("imag.png")

    # 8. Write video of streak image
    if "v" in args.result:
        name_video_file = args.dir + "/streak_video"
        print(f"Writing streak image video to {name_video_file}")

        write_video_custom(np.real(transformed), filename=name_video_file + "_real")
        write_video_custom(np.imag(transformed), filename=name_video_file + "_imag")
        write_video_custom(np.angle(transformed), filename=name_video_file + "_phase")

    if "f" in args.result:
        name_folder = args.dir + "/freq_streak"
        print("Writing frames separately")
        write_frames(np.real(transformed), folder=name_folder + "_real")
        write_frames(np.imag(transformed), folder=name_folder + "_imag")

def visualize(streakimg):
    if "v" in args.result:
        name_video_file = args.dir + "/streak_video"
        print(f"Writing streak image video to {name_video_file}")

        write_video_custom(streakimg, filename=name_video_file)

    if "f" in args.result:
        name_folder = args.dir + "/freq_streak"
        print("Writing frames separately")
        write_frames(streakimg, folder=name_folder)

def main(args):

    # Load streak image
    print("Loading streak image")
    streakimg = read_streakimg_mitsuba(args.dir, extension=args.extension)

    streakimg = streakimg[:, :, :, 0].astype(np.float32)  # only red

    if args.validate:
        validate(streakimg)
    elif args.visualize:
        visualize(streakimg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool for Frequency Streak Image validation")
    # Options for reading steady and streak image
    parser.add_argument('-d', '--dir', type=str, help="Directory where the streak images are stored",
                        default="cbox")

    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--validate", action="store_true", help="Show validation video with full frequency spectrum")
    task_group.add_argument("--visualize", action="store_true", help="Visualize results on ")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-r', "--red", action="store_true", help="Show red channel only")
    group.add_argument('-g', '--green', type=str, help="Show green channel only")
    group.add_argument('-b', '--blue', type=str, help="Show blue channel only")

    parser.add_argument('-ext', '--extension', type=str, help="Name of the extension of the images", default="exr")
    # Option to show intermediate result
    parser.add_argument('-s', '--show', action="store_true", help="Show images or results visually")
    parser.add_argument('--result', type=str, nargs='+', help="Result: video (v), frames (f)", default=['v'])
    parser.add_argument('-e', '--exposure-time', type=float, help="exposure time for each frame, measured in optical length", default=8)
    args = parser.parse_args()

    main(args)