import multiprocessing
import os

from scipy.ndimage import find_objects, gaussian_filter, generate_binary_structure, label, maximum_filter1d, binary_fill_holes
import numpy as np
import fill_voids
import sys
from functools import partial
from multiprocessing.pool import Pool
from multiprocessing.dummy import Pool as ThreadPool
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import active_children
from scipy.sparse import coo_array
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def app(masks):
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array' % masks.ndim)

    slices = find_objects(masks)
    idx = np.arange(len(slices))
    logger.info("Found %d slices" % len(slices))
    si = list(zip(slices, idx))
    out = task(masks, si)
    return out


def task(masks, chnk):
    with ThreadPool() as pool:
        out = pool.map(partial(fill_holes, masks), chnk)
    return out


def fill_holes(masks, argin):
    i = argin[1]
    slc = argin[0]

    out = []
    if i % 1000 == 0:
        logger.info("Doing %d" % i)
    if slc is not None:
        msk = masks[slc] == (i + 1)
        npix = msk.sum()
        if npix > 0:
            temp = []
            if msk.ndim == 3:
                for k in range(msk.shape[0]):
                    temp.append(coo_array(fill_voids.fill(msk[k], in_place=False)))
            else:
                temp.append(coo_array(fill_voids.fill(msk, in_place=False)))
            out.append((slc, temp))
    return out


if __name__ == "__main__":
    global masks

    logger.info('Hello')
    masks = np.load("/home/dimitris/data/Mathieu/stitching_masks/stitched_masks.npy")
    logger.info('start')
    out = app(masks)

    j = 0
    for el in out:
        logger.info('doing j:%d' % j)
        slc = el[0][0]
        msk = np.stack([d.toarray() for d in el[0][1]])
        masks[slc][msk] = (j + 1)
        j += 1
    logger.info('end')



