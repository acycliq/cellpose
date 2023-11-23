import multiprocessing
import os

from scipy.ndimage import find_objects, gaussian_filter, generate_binary_structure, label, maximum_filter1d, binary_fill_holes
import numpy as np
import fill_voids
import sys
import gc
from functools import partial
from multiprocessing.pool import Pool
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.dummy import Lock
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


def fill_holes(data):

    lock = Lock()
    si_tuples = data[0]
    masks = data[1]
    items = [(si, lock) for si in si_tuples]
    with ThreadPool() as pool:
        # async isnot necessary here, maybe change it
        res = pool.map(partial(_fill_holes, masks), items)
        # pool.starmap_async(task, items)
        res.wait()


def _fill_holes(argin):
    bbox = argin[0]
    i = argin[1]

    out = []
    if i % 1 == 0:
        logger.info("Doing %d" % i)
    msk = bbox == (i + 1)  ## bbox = masks[slc]
    npix = msk.sum()
    if npix > 0:
        temp = []
        if msk.ndim == 3:
            for k in range(msk.shape[0]):
                temp.append(coo_array(fill_voids.fill(msk[k], in_place=False)))
        else:
            temp.append(coo_array(fill_voids.fill(msk, in_place=False)))
        out.append((i, temp))
    return out


# def task(masks, chnk):
#     with ThreadPool() as pool:
#         out = pool.map(partial(_fill_holes, masks), chnk)
#     return out

def app(msks):
    out = []
    n_workers = os.cpu_count()
    # n_workers = 1

    # slices = find_objects(masks)
    # slices = slices[:300]
    idx = np.arange(len(msks))
    logger.info("Found %d slices" % len(msks))
    si = list(zip(msks, idx))
    # determine chunksize
    # n_workers = 2
    chunksize = max(1, round(len(si) / n_workers))
    # create the process pool
    with ThreadPoolExecutor() as executor:
        futures = []
        # split the load operations into chunks
        futures = [executor.submit(_fill_holes, d) for d in si]
        for future in as_completed(futures):
            print('Completed future')
            # print(len(future.result()))
            out.append(future.result())

    # flatten the list of lists
    out = sum(out, [])
    print('Done')
    return out


def helper(masks, slc):
    out = masks[slc]
    # out = [coo_array(d) for d in out]
    # del masks
    return out


if __name__ == "__main__":
    global masks

    logger.info('Hello')
    masks = np.load("/home/dimitris/data/Mathieu/stitching_masks/stitched_masks.npy")
    logger.info('start')
    slices = find_objects(masks)

    msks = []
    with ThreadPool() as pool:
        res = pool.map(partial(helper, masks), slices)

    # remove_labels = remove_small_masks_par(masks, min_size=15)
    # out = fill_holes_par(masks)
    out = app(res[:100])

    j = 0
    for el in out:
        logger.info('doing j:%d' % j)
        slc = el[0][0]
        msk = np.stack([d.toarray() for d in el[0][1]])
        masks[slc][msk] = (j + 1)
        j += 1
    logger.info('end')
