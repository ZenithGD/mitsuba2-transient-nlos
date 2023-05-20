import argparse
import glob
import os
from pathlib import Path

import imageio

import cv2 as cv
import mitsuba

from tonemapper import *
from tqdm import tqdm

mitsuba.set_variant("scalar_rgb")

from read_write_imgs import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool for Streak Images")
    # Options for reading steady and streak image
    parser.add_argument('-d', '--dir', type=str, help="Directory where the steady and streak images are stored",
                        default="cbox")
    parser.add_argument('-ext', '--extension', type=str, help="Name of the extension of the images", default="exr")
    # Option to show intermediate result
    parser.add_argument('-s', '--show', action="store_true", help="Show images or results visually")
    # Tonemapping options
    parser.add_argument('-n', '--normalize', action="store_true", help="Normalize values before applying tonemapper")
    parser.add_argument('-e', '--exposure', type=float, help="Exposure: 2^exposure_value", default=0)
    parser.add_argument('-o', '--offset', type=float, help="Offset: value + offset_value", default=0)
    parser.add_argument('-t', '--tonemapper', type=str, help="Tonemapper applied: SRGB, GAMMA, PN", default="SRGB")
    parser.add_argument('-g', '--gamma', type=float, help="Float value of the gamma", default=2.2)
    # Options for result
    parser.add_argument('-r', '--result', type=str, nargs='+', help="Result: video (v), frames (f)", default=["v"])
    args = parser.parse_args()

    # 1. Load streak image
    print("Loading streak image")
    path_streak_img = args.dir
    streakimg = load_streakimg(path_streak_img, ext=args.extension)

    # NOTE(diego): lower memory usage, similar results
    streakimg = streakimg.astype(np.float16)

    streakimg = streakimg[:, :, :, :3]  # drop alpha

    # 2. Apply tonemap to streak image
    print("Applying tonemap to streak image")
    streakimg_ldr = tonemap(streakimg,
                            normalize=args.normalize,
                            exposure=args.exposure,
                            offset=args.offset,
                            tonemapper=args.tonemapper,
                            gamma=args.gamma)
    maxmin(streakimg_ldr)

    # 3. Write video of streak image
    if "v" in args.result:
        name_video_file = args.dir + "/streak_video"
        print("Writing streak image video")
        write_video_custom(streakimg_ldr, filename=name_video_file)

    if "f" in args.result:
        name_folder = args.dir + "/frames"
        print("Writing frames separately")
        write_frames(streakimg_ldr, folder=name_folder)
