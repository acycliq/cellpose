import multiprocessing
import os

from scipy.ndimage import find_objects, gaussian_filter, generate_binary_structure, label, maximum_filter1d, binary_fill_holes
import numpy as np
import fill_voids
import sys
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

try:
    from skimage.morphology import remove_small_holes
    SKIMAGE_ENABLED = True
except:
    SKIMAGE_ENABLED = False


def fill_holes_and_remove_small_masks(masks, min_size=15):
    """ fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)

    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes

    (might have issues at borders between cells, todo: check and fix)

    Parameters
    ----------------

    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    min_size: int (optional, default 15)
        minimum number of pixels per mask, can turn off with -1

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with holes filled and masks smaller than min_size removed,
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    """

    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array' % masks.ndim)

    slices = find_objects(masks)
    slices = slices[14500:15001]
    logger.info("Found %d slices" % len(slices))
    j = 0
    for i, slc in enumerate(slices): # was 25000
        if i % 500 == 0:
            logger.info("Doing %d" % i)
        if slc is not None:
            msk = masks[slc] == (i + 1)
            npix = msk.sum()
            if npix > 0:
                if min_size > 0 and npix < min_size:
                    masks[slc][msk] = 0

                if msk.ndim == 3:
                    for k in range(msk.shape[0]):
                        if msk[k].sum() > 0:
                            # msk[k] = binary_fill_holes(msk[k])
                            msk[k] = fill_voids.fill(msk[k], in_place=False)
                            # assert np.all(msk[k] == fv)
                else:
                    # msk = binary_fill_holes(msk)
                    msk = fill_voids.fill(msk, in_place=False)
                masks[slc][msk] = (j + 1)
                j += 1
    return masks



def fill_holes_and_remove_small_masks_par(masks, min_size=15):

    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array' % masks.ndim)

    slices = find_objects(masks)
    logger.info("Found %d slices" % len(slices))

    slc = slices[45000:]
    i = np.arange(len(slc))
    minsize = min_size * np.ones(len(slc))
    pool = Pool(1)
    res = pool.map(helper, zip(i, slc, minsize))
    pool.close()
    # wait a moment
    pool.join()
    # report a message
    print('Main all done.')
    # report the number of child processes that are still active
    children = active_children()
    print(f'Active children: {len(children)}')


    # j = 0
    # for i, slc in enumerate(slices[25000:]):
    #     i = 1925
    #     slc = slices[25000:][1925]
    #     if i%500 == 0:
    #         logger.info("Doing %d" % i)
    #     if slc is not None:
    #         msk = masks[slc] == (i + 1)
    #         npix = msk.sum()
    #         if npix > 0:
    #             if min_size > 0 and npix < min_size:
    #                 logger.info("zero-ing out mask %d" % (i + 1))
    #                 masks[slc][msk] = 0
    #
    #             if msk.ndim == 3:
    #                 for k in range(msk.shape[0]):
    #                     logger.info("filling voids k:%d" % k)
    #                     # msk[k] = binary_fill_holes(msk[k])
    #                     msk[k] = fill_voids.fill(msk[k], in_place=False)
    #                     # assert np.all(msk[k] == fv)
    #             else:
    #                 # msk = binary_fill_holes(msk)
    #                 msk = fill_voids.fill(msk, in_place=False)
    #             masks[slc][msk] = (j + 1)
    #             j += 1
    # return masks

def helper(d):
    i = d[0]
    slc = d[1]
    min_size = d[2]
    logger.info("Doing %d" % i)
    if slc is not None:
        if i in [1925]:
            logger.info('stop1')
        msk = masks[slc] == (i + 1)
        npix = msk.sum()
        if npix > 0:
            """
            moving this inside the npix > 0 condition because it hits an error:
            'ValueError: zero-size array to reduction operation minimum which has no identity'
            and fails silently when msk.sum() = 0
            """
            if min_size > 0 and npix < min_size:
                logger.info("zero-ing out mask %d" % (i+1))
                if i in [2567, 1925]:
                    logger.info('stop2')
                masks[slc][msk] = 0

            if msk.ndim == 3:
                for k in range(msk.shape[0]):
                    # msk[k] = binary_fill_holes(msk[k])
                    logger.info("filling voids k:%d" % k)
                    if i in [641, 3851]:
                        logger.info('msk.sum = %d' % msk[k].sum())
                        logger.info('stop3')
                    if msk[k].sum() > 0:
                        msk[k] = fill_voids.fill(msk[k], in_place=False)
                        logger.info('voids filled!')

                    # assert np.all(msk[k] == fv)
            else:
                # msk = binary_fill_holes(msk)
                msk = fill_voids.fill(msk, in_place=False)
#

def remove_small_masks(masks, min_size=15):
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array' % masks.ndim)

    slices = find_objects(masks)
    logger.info("Found %d slices" % len(slices))
    for i, slc in enumerate(slices): # was 25000
        if i % 5 == 0:
            logger.info("Doing %d" % i)
        if slc is not None:
            msk = masks[slc] == (i + 1)
            npix = msk.sum()
            if npix > 0:
                if min_size > 0 and npix < min_size:
                    masks[slc][msk] = 0
    return masks



def remove_small_masks_par(masks, min_size=15):
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array' % masks.ndim)
    slices = find_objects(masks)
    idx = np.arange(len(slices))
    minsize = min_size * np.ones(len(slices))
    sim = zip(idx, slices, minsize)
    logger.info("Found %d slices" % len(slices))
    with multiprocessing.Pool() as pool:
        out = pool.map(_remove_small_masks, sim)
    return out



def _remove_small_masks(argin):
    i = argin[0]
    slc = argin[1]
    min_size = argin[2]
    out = []
    if i % 1 == 0:
        logger.info("Doing %d" % i)
    if slc is not None:
        zmin = slc[0].start
        zmax = slc[0].stop

        ymin = slc[1].start
        ymax = slc[1].stop

        xmin = slc[2].start
        xmax = slc[2].stop

        logger.info((i+1, zmin, zmax, ymin, ymax, xmin, xmax))
        msk = masks[zmin:zmax, ymin:ymax, xmin:xmax] == (i+1)
        npix = msk.sum()
        if npix > 0:
            if min_size > 0 and npix < min_size:
                logger.info('append %d' % (i+1))
                out.append(i+1)
    return out


def fill_holes_par(masks):
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array' % masks.ndim)

    slices = find_objects(masks)
    idx = np.arange(len(slices))
    logger.info("Found %d slices" % len(slices))
    si = zip(slices, idx)
    logger.info("Found %d slices" % len(slices))
    with multiprocessing.Pool() as pool:
        out = pool.map(_fill_holes, si)
    return out


def fill_holes_exe(masks):
    out = []
    # n_workers = os.cpu_count()
    n_workers = 1
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array' % masks.ndim)

    slices = find_objects(masks)
    # slices = slices[:300]
    idx = np.arange(len(slices))
    logger.info("Found %d slices" % len(slices))
    si = list(zip(slices, idx))
    chunksize = max(1, round(len(si) / n_workers))

    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(si), chunksize):
            # select a chunk of filenames
            chunk = si[i:(i + chunksize)]
            # make a dict to carry around other data we need and submit the task
            future = executor.submit(task, chunk)
            futures.append(future)
        for future in as_completed(futures):
            print('Completed future')
            print(len(future.result()))
            out.append(future.result())

    # flatten the list of lists
    out = sum(out, [])
    print('Done')
    return out


def task(chnk):
    with ThreadPool() as pool:
        out = pool.map(_fill_holes, chnk)
    return out


# def task(chnk):
#     with ThreadPoolExecutor(len(chnk)) as exe:
#         futures = [exe.submit(_fill_holes, d) for d in chnk]
#         data_list = [future.result() for future in futures]
#         # return data and file paths
#         return (data_list, chnk)


def _fill_holes(argin):
    i = argin[1]
    slc = argin[0]

    out = []
    if i % 500 == 0:
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
    # remove_labels = remove_small_masks_par(masks, min_size=15)
    # out = fill_holes_par(masks)
    out = fill_holes_exe(masks)

    j = 0
    for el in out:
        logger.info('doing j:%d' % j)
        slc = el[0][0]
        msk = np.stack([d.toarray() for d in el[0][1]])
        masks[slc][msk] = (j + 1)
        j += 1
    logger.info('end')



