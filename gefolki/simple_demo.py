import numpy as np
import pylab as pl
from PIL import Image

from gefolki.algorithm import EFolki
from gefolki.settings import DATA_DIR
from gefolki.tools import wrapData


def run_simple_demo():
    img1 = np.zeros((512, 512), dtype=np.float32)
    img1[50:300, 50:300] = 1.0

    img2 = np.zeros((512, 512), dtype=np.float32)
    img2[60:310, 60:310] = 1.0

    img1_pil = Image.fromarray((img1 * 255).astype(np.uint8))
    img1_pil.save(DATA_DIR / "img1.png")

    img2_pil = Image.fromarray((img2 * 255).astype(np.uint8))
    img2_pil.save(DATA_DIR / "img2.png")

    u, v = EFolki(img1, img2, iteration=2, radius=[32, 24, 16, 8], rank=4, levels=5)

    optical_flow_norm = np.sqrt(u ** 2 + v ** 2)

    pl.figure()
    pl.imshow(optical_flow_norm)
    pl.title("Norm of optical flow")
    pl.colorbar()
    pl.savefig(DATA_DIR / "optical_flow_norm.png")

    img2_resampled = wrapData(img2, u, v)

    img2_resampled_pil = Image.fromarray((img2_resampled * 255).astype(np.uint8))
    img2_resampled_pil.save(DATA_DIR / "img2_resampled.png")



if __name__ == "__main__":
    run_simple_demo()
