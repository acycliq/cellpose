from scipy.ndimage import find_objects, gaussian_filter, generate_binary_structure, label, maximum_filter1d, binary_fill_holes
import numpy as np
import fill_voids
import sys
from multiprocessing.pool import Pool
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
    logger.info("Found %d slices" % len(slices))
    j = 0
    for i, slc in enumerate(slices[25000:]):
        if i%500 == 0:
            logger.info("Doing %d" % i)
        if slc is not None:
            msk = masks[slc] == (i + 1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            elif npix > 0:
                if msk.ndim == 3:
                    for k in range(msk.shape[0]):
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
    with Pool() as pool:
        pool.map(helper, zip(i, slc, minsize))


    # j = 0
    # for i, slc in enumerate(slices[25000:]):
    #     if i%500 == 0:
    #         logger.info("Doing %d" % i)
    #     if slc is not None:
    #         msk = masks[slc] == (i + 1)
    #         npix = msk.sum()
    #         if min_size > 0 and npix < min_size:
    #             masks[slc][msk] = 0
    #         elif npix > 0:
    #             if msk.ndim == 3:
    #                 for k in range(msk.shape[0]):
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
        msk = masks[slc] == (i + 1)
        npix = msk.sum()
        if min_size > 0 and npix < min_size:
            masks[slc][msk] = 0
        elif npix > 0:
            if msk.ndim == 3:
                for k in range(msk.shape[0]):
                    # msk[k] = binary_fill_holes(msk[k])
                    msk[k] = fill_voids.fill(msk[k], in_place=False)
                    # assert np.all(msk[k] == fv)
            else:
                # msk = binary_fill_holes(msk)
                msk = fill_voids.fill(msk, in_place=False)



if __name__ == "__main__":
    logger.info('Hello')
    masks = np.load("/home/dimitris/data/Mathieu/stitching_masks/stitched_masks.npy")
    logger.info('start')
    out = fill_holes_and_remove_small_masks_par(masks)
    logger.info('end')




