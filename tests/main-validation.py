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
from visualize import *

import h5py

mitsuba.set_variant("scalar_rgb")

def apply_fft(streakimg, et):
    transformed = np.fft.fftshift(np.fft.fft(streakimg, axis=2), axes=2)

    freqs = np.fft.fftshift(np.fft.fftfreq(n=streakimg.shape[2], d=et))

    return transformed, freqs

def apply_ifft(streakimg):
    return np.fft.ifft(np.fft.ifftshift(streakimg, axes=2), axis=2)

def compare(t_streakimg, f_streakimg, et):
    
    # 1. Apply FFT to time-resolved image
    print("Applying fft to each streak image")
    t_transformed, freqs = apply_fft(t_streakimg, et)
    print("done.")
    print("min freq :", freqs.min())
    print("max freq :", freqs.max())

    # 2. Compare FFT of time-resolved streakimg and freq transformed streakimg side by side
    compare_window(np.real(t_transformed), np.real(f_streakimg), freqs, "Bin frecuencia", "Imagen, parte real")
    compare_slice_window(np.real(t_transformed), np.real(f_streakimg), "Perfiles, parte real")
    compare_window(np.imag(t_transformed), np.imag(f_streakimg), freqs, "Bin frecuencia", "Imagen, parte imaginaria")
    compare_slice_window(np.imag(t_transformed), np.imag(f_streakimg), "Perfiles, parte imaginaria")

    # 3. Apply IFFT to frequency domain streakimg and obtain real part
    print("Applying ifft to each streak image")
    f_inv_transformed = np.real(apply_ifft(f_streakimg))
    print("done.")

    # 4. Compare time-resolved streakimg and IFFT of freq transformed streakimg side by side
    compare_window(t_streakimg, f_inv_transformed, np.arange(t_streakimg.shape[2]), "Bin temporal", "Transformada inversa")
    del f_inv_transformed
    
    # 5. Compute phase of both FFT transform and manual transform
    print("Computing phase")
    t_phase = np.angle(t_transformed)
    f_phase = np.angle(f_streakimg)

    print(f_phase.max())
    print(f_phase.min())
    print("done.")

    # 6. Compare phase of both FFT transform and manual transform
    compare_window(t_phase, f_phase, "Bin frecuencia", "Fase", cm.twilight_shifted, (-np.pi, np.pi))

def main(args):
    # 1. Load time-resolved streakimg and frequency streakimg
    t_streakimg = load_streakimg(args.time_dir, args.extension)[10:-10,10:-10,:,0]
    f_streakimg = load_streakimg(args.freq_dir, args.extension)[10:-10,10:-10]
    f_streakimg = f_streakimg[:,:,:,0] + 1j * f_streakimg[:,:,:,1]

    # 2. Compare FFT of time-resolved streakimg and freq transformed streakimg side by side
    compare(t_streakimg, f_streakimg, args.exposure_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool for Frequency Streak Image visualization and validation")
    # Options for reading steady and streak image

    parser.add_argument("time_dir", metavar="time-dir", help="Directory where the time-resolved streak images are stored")
    parser.add_argument("freq_dir", metavar="freq-dir", help="Directory where the frequency domain streak images are stored")

    parser.add_argument("--out", help="output directory")

    parser.add_argument('-ext', '--extension', type=str, help="Name of the extension of the images", default="exr")

    parser.add_argument('-e', '--exposure-time', type=float, help="exposure time for each frame, measured in optical length", default=8)
    args = parser.parse_args()

    main(args)