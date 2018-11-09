#!/usr/bin/python

import numpy as np
import pylab as pl
from scipy.ndimage import imread

from gefolki.algorithm import GEFolki, EFolki
from gefolki.tools import wrapData


def run_demo():
    print("Starting Lidar/Radar co-registration...\n")
    radar = imread("../datasets/radar_bandep.png")
    Ilidari = imread("../datasets/lidar_georef.png")

    pl.figure()
    pl.imshow(radar)
    pl.title("Radar in pauli color")

    pl.figure()
    pl.imshow(Ilidari)
    pl.title("Lidar in colormap viridis")

    Iradar = radar[:, :, 0]
    Iradar = Iradar.astype(np.float32) / 255
    Ilidar = Ilidari.astype(np.float32) / 255

    u, v = EFolki(Iradar, Ilidar, iteration=2, radius=[32, 24, 16, 8], rank=4, levels=5)
    N = np.sqrt(u ** 2 + v ** 2)
    pl.figure()
    pl.imshow(N)
    pl.title("Norm of LIDAR to RADAR registration")
    pl.colorbar()

    Ilidar_resampled = wrapData(Ilidar, u, v)

    C = np.dstack((Ilidar, Iradar, Ilidar))
    pl.figure()
    pl.imshow(C)
    pl.title("Imfuse of RADAR and LIDAR")

    D = np.dstack((Ilidar_resampled, Iradar, Ilidar_resampled))
    pl.figure()
    pl.imshow(D)
    pl.title("Imfuse of RADAR and LIDAR after coregistration")

    print("End of Lidar/Radar co-registration\n\n")

    print("Starting optical/Radar co-registration\n")
    radar = imread("../datasets/radar_bandep.png")
    Ioptique = imread("../datasets/optiquehr_georef.png")

    pl.figure()
    pl.imshow(radar)
    pl.title("Radar in pauli color")

    pl.figure()
    pl.imshow(Ioptique)
    pl.title("Optical")

    Iradar = radar[:, :, 0]
    Iradar = Iradar.astype(np.float32)
    Ioptique = Ioptique[:, :, 1]
    Ioptique = Ioptique.astype(np.float32)

    u, v = GEFolki(
        Iradar, Ioptique, iteration=2, radius=range(32, 4, -4), rank=4, levels=6
    )

    N = np.sqrt(u ** 2 + v ** 2)
    pl.figure()
    pl.imshow(N)
    pl.title("Norm of OPTICAL to RADAR registration")
    pl.colorbar()

    Ioptique_resampled = wrapData(Ioptique, u, v)

    C = np.dstack((Ioptique / 255, Iradar / 255, Ioptique / 255))
    pl.figure()
    pl.imshow(C)
    pl.title("Imfuse of RADAR and OPTICAL")

    D = np.dstack((Ioptique_resampled / 255, Iradar / 255, Ioptique_resampled / 255))
    pl.figure()
    pl.imshow(D)
    pl.title("Imfuse of RADAR and OPTIC after coregistration")
    print("End of optical/Radar co-registration\n\n")


if __name__ == "__main__":
    run_demo()
    pl.show()
else:
    pl.interactive(True)
    run_demo()
