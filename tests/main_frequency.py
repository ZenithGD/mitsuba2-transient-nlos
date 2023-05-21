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

from read_write_imgs import *
from tonemapper import *
from tqdm import tqdm

import h5py

mitsuba.set_variant("scalar_rgb")

def apply_fft(streakimg, et):
    ini = time.time()
    
    transformed = np.empty(streakimg.shape, dtype=np.complex64)

    # loop through all streak images
    for i in range(streakimg.shape[1]):
        streak = streakimg[:, i, :]
        transformed[:, i, :] = np.fft.fftshift(np.fft.fft(streak, axis=1), axes=1)

    freqs = np.fft.fftshift(np.fft.fftfreq(n=streakimg.shape[2], d=et))

    return transformed, freqs

def validate(streakimg, exposure_time, out):
    print("Applying fft to each streak image")
    ini = time.time()
    
    transformed = np.empty(streakimg.shape, dtype=np.complex64)

    # loop through all streak images
    for i in range(streakimg.shape[1]):
        streak = streakimg[:, i, :]

        transformed[:, i, :] = np.fft.fftshift(np.fft.fft(streak, axis=1), axes=1)
    
    print("min : ", np.amin(transformed))
    print("max : ", np.amax(transformed))

    freqs = np.fft.fftshift(np.fft.fftfreq(n=streakimg.shape[2], d=exposure_time))
    print("min freq :", freqs.min())
    print("max freq :", freqs.max())
    
    elapsed = time.time() - ini
    print(f"took {elapsed} seWcs.")

    # 8. Write video of streak image
    if "v" in args.result:
        name_video_file = out
        print(f"Writing streak image video to {name_video_file}")

        write_video_custom(np.real(transformed), filename=name_video_file + "_real")

    if "f" in args.result:
        name_folder = out + "/freq_streak"
        print("Writing frames separately")
        write_frames(np.real(transformed), folder=name_folder + "_real")
        write_frames(np.imag(transformed), folder=name_folder + "_imag")

def show_streakimg(streakimg):
    tst = cm.hot(streakimg)

    fig, ax1 = plt.subplots(1,1)
    plt.subplots_adjust(bottom=0.15)
    
    ax1.imshow(tst[:, :, 0])

    print("showing plot")
    def update_plot(val):
        idx = int(sliderwave.val)
        ax1.cla()
        ax1.imshow(tst[:, :, idx])
        fig.canvas.draw_idle()

    # Sliders
    axwave = plt.axes([0.25, 0.05, 0.5, 0.03])

    sliderwave = Slider(axwave, 'Y slice', 0, streakimg.shape[2] - 1, valinit=0, valfmt='%d')
    sliderwave.on_changed(update_plot)
    plt.show()

def visualize(streakimg, args):

    print("min : ", np.amin(streakimg))
    print("max : ", np.amax(streakimg))
    print("mean : ", np.mean(streakimg))
    print("stddev : ", np.std(streakimg))

    print(streakimg.shape)

    # real part
    show_streakimg(streakimg[:,:,:,0])

    # imag part
    show_streakimg(streakimg[:,:,:,1])

    cplx = streakimg[:,:,:,0] + 1j * streakimg[:,:,:,1]

    # phase part
    phase = np.angle(cplx)
    phase = np.where(phase < 0, phase + 2 * np.pi, phase)
    del cplx
    show_streakimg(phase)

    # depth reconstruction
    wl = 5000
    depth = phase[:,:,0] * np.abs(wl)/(4*np.pi);

    plt.imshow(depth, cmap="hot")
    print(depth.min(), depth.max())

    fig, ax = plt.subplots()
    cont = ax.contour(np.arange(depth.shape[1]), np.arange(depth.shape[0] - 1, -1, -1), depth)
    ax.clabel(cont, inline=True, fontsize=10)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    gridY, gridX = np.mgrid[1:depth.shape[0]:depth.shape[0] * 1j,
                           1:depth.shape[1]:depth.shape[1] * 1j]
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    pSurf = ax.plot_surface(gridX, gridY, depth, cmap='viridis')
    fig.colorbar(pSurf)
    plt.show()

    plt.show()

    if "v" in args.result:
        name_video_file = args.out
        print(f"Writing streak image video to {name_video_file}")

        write_video_custom(phase, filename=name_video_file)

    if "f" in args.result:
        name_folder = args.out + "/freq_streak"
        print("Writing frames separately")
        write_frames(phase, folder=name_folder)

def compare(t_streakimg, f_streakimg, et):
    
    print("Applying fft to each streak image")
    t_transformed, freqs = apply_fft(t_streakimg, et)
    t_transformed = np.real(t_transformed) 
    print("done.")
    print("min freq :", freqs.min())
    print("max freq :", freqs.max())

    tst = cm.hot(t_transformed)
    fst = cm.hot(f_streakimg)

    fig, (ax1, ax2) = plt.subplots(1,2)
    plt.subplots_adjust(bottom=0.15)
    
    ax1.imshow(tst[:, :, 0])
    ax2.imshow(fst[:, :, 0])

    print("showing plot")
    def update_plot(val):
        idx = int(sliderwave.val)
        ax1.cla()
        ax2.cla()
        ax1.imshow(tst[:, :, idx])
        ax2.imshow(fst[:, :, idx])
        fig.canvas.draw_idle()

    # Sliders

    axwave = plt.axes([0.25, 0.05, 0.5, 0.03])

    sliderwave = Slider(axwave, 'Y slice', 0, f_streakimg.shape[2] - 1, valinit=0, valfmt='%d')
    sliderwave.on_changed(update_plot)

    plt.show()

def main(args):

    if args.validate:

        streakimg = load_streakimg(args.validate[0], args.extension)
        validate(streakimg[:, :, :, 0], args.exposure_time, args.out)
    elif args.visualize:

        streakimg = load_streakimg(args.visualize[0], args.extension)
        visualize(streakimg[10:-10,10:-10], args)
    elif args.compare:
        
        tstreakimg = load_streakimg(args.compare[0], args.extension)[:, :, :, 0]
        fstreakimg = load_streakimg(args.compare[1], args.extension)[:, :, :, 0]
        compare(tstreakimg, fstreakimg, args.exposure_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool for Frequency Streak Image visualization and validation")
    # Options for reading steady and streak image

    parser.add_argument("--out", help="optional output file name")

    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--validate", nargs=1, help="Show validation video with full frequency spectrum")
    task_group.add_argument("--visualize", nargs=1, help="Visualize results on video")
    task_group.add_argument("--compare", nargs=2, help="Compare validation and result")

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