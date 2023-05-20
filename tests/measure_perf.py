import os, time, argparse
import mitsuba
mitsuba.set_variant('scalar_rgb')
from mitsuba.core import Thread
from mitsuba.core.xml import load_file

import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

def folder_size(folder_path):
    size = 0
    for path, _, files in os.walk(folder_path):
        for f in files:
            fp = os.path.join(path, f)
            size += os.path.getsize(fp)
    return size

def main(args):

    # Absolute or relative path to the XML file
    filename = args.scene

    fnum = np.floor(np.linspace(1, 380, num = args.number, endpoint=True)).astype(np.int32)
    snum = [ 2 ** i for i in range(5, 9) ]

    # setup plots
    
    plt.style.use('fivethirtyeight')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.set_title("Coste temporal")
    ax1.set_xlabel("Número de frecuencias")
    ax1.set_ylabel("Tiempo (s)")

    ax2.set_title("Coste en memoria")
    ax2.set_xlabel("Número de frecuencias")
    ax2.set_ylabel("Tiempo (s)")

    for s in snum:

        time_values = []
        memory_values = []

        for f in fnum:
            
            ini = time.time()
            # render scene
            os.system(f'mitsuba {filename} -D fres={f} -D spp={s}')
            
            time_values.append(time.time() - ini)

            # strip extension and measure folder size
            memory_values.append(folder_size(os.path.splitext(filename)[0]))

        ax1.plot(fnum, time_values, label=f'{s}')

        ax2.plot(fnum, memory_values, label=f'{s}')

    ax1.legend(fancybox=True, title="Muestras por píxel")
    ax2.legend(fancybox=True, title="Muestras por píxel")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate time and memory performance")

    parser.add_argument("scene", help="Scene file")
    parser.add_argument("-n", "--number", type=int, help="Number of frequency values to test", default=10)
    parser.add_argument("--out", help="out file graphs")

    main(parser.parse_args())