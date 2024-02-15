
import glob
import multiprocessing
import os
import pickle
import sys
from colorsys import rgb_to_hsv
from datetime import date
from enum import Enum
from functools import partial
from shutil import copy2, copyfile
from typing import List, Tuple

import cv2 as cv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from PIL import Image
from matplotlib.collections import PatchCollection
from tqdm import tqdm
from utils import get_slide_magnification


Image.MAX_IMAGE_PIXELS = None


class OTSU_METHOD(Enum):
    OTSU3_FLEXIBLE_THRESH = 0
    OTSU3_LOWER_THRESH = 1
    OTSU3_UPPER_THRESH = 2
    OTSU_REGULAR = 3



def make_grid(ROOT_DIR: str = './TCGA',
              tile_sz: int = 256,
              tissue_coverage: float = 0.5,
              desired_magnification: int = 10,
              num_workers: int = 1):
    """    
    :param ROOT_DIR: Root Directory for the data
    :param tile_sz: Desired tile size at desired magnification.
    :param tissue_coverage: tissue percent requirement for each tile in the grid.
    :return: 
    """""

    slides_data_file = os.path.join(ROOT_DIR, 'slides_data' + '.xlsx')

    slides_meta_data_DF = pd.read_excel(slides_data_file)
    files = slides_meta_data_DF['file'].tolist()

    meta_data_DF = pd.DataFrame(files, columns=['file'])

    slides_meta_data_DF.set_index('file', inplace=True)
    meta_data_DF.set_index('file', inplace=True)
    tile_nums = []
    total_tiles = []

    # Save the grid to file:
    grids_dir = os.path.join(ROOT_DIR, 'Grids_' + str(desired_magnification))
    grid_images_dir = os.path.join(ROOT_DIR, 'SegData',
                                   'GridImages_' + str(desired_magnification) + '_' + str(
                                       tissue_coverage))
    if not os.path.isdir(grids_dir):
        os.mkdir(grids_dir)
    if not os.path.isdir(grid_images_dir):
        os.mkdir(grid_images_dir)

    print('Starting Grid production...')

    with multiprocessing.Pool(num_workers) as pool:
        for tile_nums1, total_tiles1 in tqdm(pool.imap(partial(_make_grid_for_image,
                                                               meta_data_DF=slides_meta_data_DF,
                                                               ROOT_DIR=ROOT_DIR,
                                                               tissue_coverage=tissue_coverage,
                                                               tile_sz=tile_sz,
                                                               desired_magnification=desired_magnification,
                                                               grids_dir=grids_dir,
                                                               grid_images_dir=grid_images_dir),
                                                       files), total=len(files)):
            tile_nums.append(tile_nums1)
            total_tiles.append(total_tiles1)

    # Adding the number of tiles to the excel file:

    slide_usage = list(((np.array(tile_nums) / np.array(total_tiles)) * 100).astype(int))

    meta_data_DF.loc[
        files, 'Legitimate tiles - ' + str(tile_sz) + ' compatible @ X' + str(desired_magnification)] = tile_nums
    meta_data_DF.loc[
        files, 'Total tiles - ' + str(tile_sz) + ' compatible @ X' + str(desired_magnification)] = total_tiles
    meta_data_DF.loc[files, 'Slide tile usage [%] (for ' + str(tile_sz) + '^2 Pix/Tile) @ X' + str(
        desired_magnification)] = slide_usage
    meta_data_DF.loc[files, 'bad segmentation'] = ''

    meta_data_DF.to_excel(os.path.join(grids_dir, 'Grid_data.xlsx'))



def _make_grid_for_image(file, meta_data_DF, ROOT_DIR,
                         tissue_coverage, tile_sz, desired_magnification, grids_dir, grid_images_dir):
    filename = '.'.join(os.path.basename(file).split('.')[:-1])
    grid_file = os.path.join(grids_dir, filename + '--tlsz' + str(tile_sz) + '.data')
    segmap_file = os.path.join(ROOT_DIR, 'SegData', 'SegMaps',
                               filename + '_SegMap.jpg')

    if os.path.isfile(os.path.join(ROOT_DIR, file)) and os.path.isfile(segmap_file):  # make sure file exists
        
        slide = openslide.open_slide(os.path.join(ROOT_DIR, file))
        height = slide.dimensions[1]
        width = slide.dimensions[0]
        data_format = file.split('.')[-1]
        magnification = get_slide_magnification(slide, data_format)

        adjusted_tile_size_at_level_0 = int(tile_sz * (int(magnification) / desired_magnification))
        basic_grid = [(row, col) for row in range(0, height, adjusted_tile_size_at_level_0) for col in
                      range(0, width, adjusted_tile_size_at_level_0)]

        total_tiles = len(basic_grid)

        # We now have to check, which tiles of this grid are legitimate, meaning they contain enough tissue material.
        legit_grid, out_grid = _legit_grid(segmap_file,
                                           basic_grid,
                                           adjusted_tile_size_at_level_0,
                                           (height, width),
                                           desired_tissue_coverage=tissue_coverage)
        # create a list with number of tiles in each file
        tile_nums = len(legit_grid)


        thumb_file_png = os.path.join(ROOT_DIR, 'SegData', 'Thumbs',
                                      filename + '_thumb.jpg')

        grid_image_file = os.path.join(grid_images_dir, filename + '_GridImage.jpg')

        if os.path.isfile(thumb_file_png):
            thumb = np.array(Image.open(thumb_file_png))
            thumb_downsample = slide.dimensions[0] / thumb.shape[1]  # shape is transposed
            patch_size_thumb = adjusted_tile_size_at_level_0 / thumb_downsample

            fig, ax = plt.subplots()
            ax.imshow(thumb)
            patch_list = []
            for patch in out_grid:
                xy = (np.array(patch[::-1]) / thumb_downsample)
                rect = patches.Rectangle(xy, patch_size_thumb, patch_size_thumb, linewidth=1, edgecolor='none',
                                         facecolor='g', alpha=0.5)
                patch_list.append(rect)
            p = PatchCollection(patch_list, alpha=0.5, facecolors='g')
            ax.add_collection(p)

            plt.axis('off')
            plt.savefig(grid_image_file,
                        bbox_inches='tight', pad_inches=0, dpi=400)
            plt.close(fig)

        with open(grid_file, 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(legit_grid, filehandle)
    else:
        print('Grid was not computed for file {}'.format(file))
        if ~os.path.isfile(os.path.join(ROOT_DIR, file)):
            print('slide was not found')
        if ~os.path.isfile(segmap_file):
            print('seg map was not found')
        tile_nums = 0
        total_tiles = -1
    return tile_nums, total_tiles


def _legit_grid(image_file_name: str,
                grid: List[Tuple],
                adjusted_tile_size_at_level_0: int,
                size: tuple,
                desired_tissue_coverage: float = 0.5) -> List[Tuple]:
    """
    This function gets a file name, a basic grid and adjusted tile size and returns a list of legitimate grid locations.
    :param image_file_name: file name
    :param grid: basic grid
    :param adjusted_tile_size_at_level_0: adjusted tile size at level 0 of the slide
    :param size: size of original image (height, width)
    :param tissue_coverage: Coverage of tissue to make the slide legitimate
    :return:
    """
    non_legit_grid_tiles = []
    # Check if coverage is a number in the range (0, 1]
    if not (desired_tissue_coverage > 0 and desired_tissue_coverage <= 1):
        raise ValueError('Coverage Parameter should be in the range (0,1]')

    # open the segmentation map image from which the coverage will be calculated:
    segMap = np.array(Image.open(image_file_name))
    row_ratio = size[0] / segMap.shape[0]
    col_ratio = size[1] / segMap.shape[1]

    # the complicated next line only only rounds up the numbers
    tile_size_at_segmap_magnification = (
        int(-(-adjusted_tile_size_at_level_0 // row_ratio)), int(-(-adjusted_tile_size_at_level_0 // col_ratio)))
    # computing the compatible grid for the small segmentation map:
    idx_to_remove = []
    for idx, (row, col) in enumerate(grid):
        new_row = int(-(-(row // row_ratio)))
        new_col = int(-(-(col // col_ratio)))

        # collect the data from the segMap:
        tile = segMap[new_row: new_row + tile_size_at_segmap_magnification[0],
               new_col: new_col + tile_size_at_segmap_magnification[1]]
        num_tile_pixels = tile_size_at_segmap_magnification[0] * tile_size_at_segmap_magnification[1]
        tissue_coverage = tile.sum() / num_tile_pixels / 255
        if tissue_coverage < desired_tissue_coverage:
            idx_to_remove.append(idx)

    # We'll now remove items from the grid. starting from the end to the beginning in order to keep the indices correct:
    for idx in reversed(idx_to_remove):
        # grid.pop(idx)
        non_legit_grid_tiles.append(grid.pop(idx))

    return grid, non_legit_grid_tiles


def make_segmentations(data_path: str = './TCGA', magnification: int = 1, num_workers: int = 1):
    
    if not os.path.isdir(os.path.join(data_path, 'SegData')):
        os.mkdir(os.path.join(data_path, 'SegData'))
    if not os.path.isdir(os.path.join(data_path, 'SegData', 'Thumbs')):
        os.mkdir(os.path.join(data_path, 'SegData', 'Thumbs'))
    if not os.path.isdir(os.path.join(data_path, 'SegData', 'SegMaps')):
        os.mkdir(os.path.join(data_path, 'SegData', 'SegMaps'))
    
    slide_files_svs = glob.glob(os.path.join(data_path, '*.svs'))
    slide_files_ndpi = glob.glob(os.path.join(data_path, '*.ndpi'))
    slide_files_mrxs = glob.glob(os.path.join(data_path, '*.mrxs'))
    slide_files_jpg = glob.glob(os.path.join(data_path, '*.jpg'))
    slide_files_tiff = glob.glob(os.path.join(data_path, '*.tiff'))
    slide_files_tif = glob.glob(os.path.join(data_path, '*.tif'))
    slide_files = slide_files_svs + slide_files_ndpi + slide_files_mrxs + slide_files_jpg + slide_files_tiff + slide_files_tif
    print('found ' + str(len(slide_files)) + ' slides')

    slides_data_file = os.path.join(data_path, 'slides_data' + '.xlsx')
    slides_meta_data_DF = pd.read_excel(slides_data_file)
    slides_meta_data_DF.set_index('file', inplace=True)

    error_list = []

    with multiprocessing.Pool(num_workers) as pool:
        for error1 in tqdm(pool.imap(partial(_make_segmentation_for_image,
                                             data_path=data_path,
                                             magnification=magnification,
                                             slides_meta_data_DF=slides_meta_data_DF),
                                     slide_files), total=len(slide_files)):
            if error1 != []:
                error_list.append(error1)

    if len(error_list) != 0:
        # Saving all error data to excel file:
        error_DF = pd.DataFrame(error_list)
        print(error_list)
        error_DF.to_excel(os.path.join(data_path, 'Segmentation_Errors.xlsx'))
        print('Segmentation Process finished WITH EXCEPTIONS!!!!')
        print('Check \"{}\" file for details...'.format(os.path.join(data_path, 'Segmentation_Errors.xlsx')))
    else:
        print('Segmentation Process finished without exceptions!')


def _make_segmentation_for_image(file, data_path, slides_meta_data_DF, magnification):
    fn, data_format = os.path.splitext(os.path.basename(file))
    data_dir = os.path.dirname(data_path)
    pic1 = os.path.exists(os.path.join(data_path, 'SegData', 'Thumbs', fn + '_thumb.jpg'))
    pic2 = os.path.exists(os.path.join(data_path, 'SegData', 'SegMaps', fn + '_SegMap.jpg'))
    if pic1 and pic2:
        return []

    slide = None
    try:
        slide = openslide.open_slide(file)
    except:
        print('Cannot open slide at location: {}'.format(file))
    if slide is not None:
        # Get a thumbnail image to create the segmentation for:
        data_format = file.split('.')[-1]
        slide_magnification = get_slide_magnification(slide, data_format)

        height = slide.dimensions[1]
        width = slide.dimensions[0]
        try:
            thumb = slide.get_thumbnail(
                (width / (slide_magnification / magnification), height / (slide_magnification / magnification)))
        except openslide.lowlevel.OpenSlideError as err:
            error_dict = {}
            e = sys.exc_info()
            error_dict['File'] = file
            error_dict['Error'] = err
            error_dict['Error Details 1'] = e[0]
            error_dict['Error Details 2'] = e[1]
            print('Exception for file {}'.format(file))
            return error_dict

        otsu_method = OTSU_METHOD.OTSU_REGULAR
        thmb_seg_map, edge_image = _calc_segmentation_for_image(thumb, magnification, otsu_method=otsu_method)
        slide.close()
        thmb_seg_image = Image.blend(thumb, edge_image, 0.5)

        # Saving segmentation map, segmentation image and thumbnail:
        thumb.save(os.path.join(data_path, 'SegData', 'Thumbs', fn + '_thumb.jpg'))
        thmb_seg_map.save(os.path.join(data_path, 'SegData', 'SegMaps', fn + '_SegMap.jpg'))
    else:
        print('Error: Found no slide in path {}'.format(dir))
        error_dict = {}
        error_dict['File'] = file
        error_dict['Error'] = 'Slide not found'
        return error_dict
    return []


def otsu3(img):
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1, -1
    for i in range(1, 256):
        for j in range(i + 1, 256):
            p1, p2, p3 = np.hsplit(hist_norm, [i, j])  # probabilities
            q1, q2, q3 = Q[i], Q[j] - Q[i], Q[255] - Q[j]  # cum sum of classes
            if q1 < 1.e-6 or q2 < 1.e-6 or q3 < 1.e-6:
                continue
            b1, b2, b3 = np.hsplit(bins, [i, j])  # weights
            # finding means and variances
            m1, m2, m3 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2, np.sum(p3 * b3) / q3
            v1, v2, v3 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2, np.sum(
                ((b3 - m3) ** 2) * p3) / q3
            # calculates the minimization function
            fn = v1 * q1 + v2 * q2 + v3 * q3
            if fn < fn_min:
                fn_min = fn
                thresh = i, j
    return thresh


def _calc_segmentation_for_image(image: Image, magnification: int, otsu_method: OTSU_METHOD) -> (Image, Image):
    """
    This function creates a segmentation map for an Image
    """

    image_array = np.array(image.convert('CMYK'))[:, :, 1]

    # turn almost total black into total white to ignore black areas in otsu
    image_array_rgb = np.array(image)
    image_is_black = np.prod(image_array_rgb, axis=2) < 20 ** 3
    image_array[image_is_black] = 0

    # otsu Thresholding:
    if (otsu_method == OTSU_METHOD.OTSU3_FLEXIBLE_THRESH) or (otsu_method == OTSU_METHOD.OTSU3_LOWER_THRESH):
        # 3way binarization
        thresh = otsu3(image_array)
        _, seg_map = cv.threshold(image_array, thresh[0], 255, cv.THRESH_BINARY)
    elif otsu_method == OTSU_METHOD.OTSU3_UPPER_THRESH:
        thresh = otsu3(image_array)
        _, seg_map = cv.threshold(image_array, thresh[1], 255, cv.THRESH_BINARY)
    else:
        _, seg_map = cv.threshold(image_array, 0, 255, cv.THRESH_OTSU)

    # test median pixel color to inspect segmentation
    if (otsu_method == OTSU_METHOD.OTSU3_FLEXIBLE_THRESH):
        pixel_vec = image_array_rgb.reshape(-1, 3)[seg_map.reshape(-1) > 0]

        hue_vec = np.array([rgb_to_hsv(*pixel / 256)[0] * 360 for pixel in pixel_vec])
        median_hue = np.median(hue_vec)
        take_upper_otsu3_thresh = median_hue < 250
        if take_upper_otsu3_thresh:  # median seg hue is not purple/red or saturation is very low (white)
            # take upper threshold
            thresh = otsu3(image_array)
            _, seg_map = cv.threshold(image_array, thresh[1], 255, cv.THRESH_BINARY)

    # Smoothing the tissue segmentation imaqe:
    size = 10 * magnification
    kernel_smooth = np.ones((size, size), dtype=np.float32) / size ** 2
    seg_map_filt = cv.filter2D(seg_map, -1, kernel_smooth)

    th_val = 5
    seg_map_filt[seg_map_filt > th_val] = 255
    seg_map_filt[seg_map_filt <= th_val] = 0

    if np.sum(seg_map_filt) == 0: 
        seg_map_PIL = Image.fromarray(seg_map_filt)
        edge_image = seg_map_PIL.convert('RGB')
        return seg_map_PIL, edge_image

    # find small contours and delete them from segmentation map
    size_thresh = 5000
    contours, _ = cv.findContours(seg_map_filt, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contour_area = np.zeros(len(contours))
    rectangle_area = np.zeros(len(contours))

    for ii in range(len(contours)):
        contour_area[ii] = cv.contourArea(contours[ii])

        # enclosing rectangle area
        [[rectangle_min_x, rectangle_min_y]] = np.min(contours[ii], axis=0)
        [[rectangle_max_x, rectangle_max_y]] = np.max(contours[ii], axis=0)
        rectangle_area[ii] = (rectangle_max_y - rectangle_min_y) * (rectangle_max_x - rectangle_min_x)

    max_contour = np.max(contour_area)
    max_contour_ind = np.argmax(contour_area)
    if (otsu_method == OTSU_METHOD.OTSU3_FLEXIBLE_THRESH) and not take_upper_otsu3_thresh:
        fill_factor_largest_contour = contour_area[max_contour_ind] / rectangle_area[max_contour_ind]
        take_upper_otsu3_thresh = fill_factor_largest_contour > 0.97  # contour is a perfect rectangle
        if take_upper_otsu3_thresh:  # median seg hue is not purple/red or saturation is very low (white)
            # take upper threshold
            thresh = otsu3(image_array)
            _, seg_map = cv.threshold(image_array, thresh[1], 255, cv.THRESH_BINARY)

    small_contours_bool = (contour_area < size_thresh) & (contour_area < max_contour * 0.02)

    if otsu_method == OTSU_METHOD.OTSU_REGULAR:
        small_contours = [contours[ii] for ii in range(len(contours)) if small_contours_bool[ii] == True]
        seg_map_filt = cv.drawContours(seg_map_filt, small_contours, -1, (0, 0, 255),
                                       thickness=cv.FILLED)  # delete the small contours

    # check contour color only for large contours
    hsv_image = np.array(image.convert('HSV'))
    rgb_image = np.array(image)
    large_contour_ind = np.where(small_contours_bool == False)[0]
    white_mask = np.zeros(seg_map.shape, np.uint8)
    white_mask[np.any(rgb_image < 240, axis=2)] = 255
    gray_contours_bool = np.zeros(len(contours), dtype=bool)
    for ii in large_contour_ind:
        # get contour mean color
        mask = np.zeros(seg_map.shape, np.uint8)
        cv.drawContours(mask, [contours[ii]], -1, 255, thickness=cv.FILLED)
        contour_color, _ = cv.meanStdDev(rgb_image, mask=mask)
        contour_std = np.std(contour_color)
        if contour_std < 5:
            hist_mask = cv.bitwise_and(white_mask, mask)
            mean_col, _ = cv.meanStdDev(hsv_image, mask=hist_mask)
            mean_hue = mean_col[0]
            if mean_hue < 100:
                gray_contours_bool[ii] = True

    gray_contours = [contours[ii] for ii in large_contour_ind if gray_contours_bool[ii] == True]
    # delete the small contours
    seg_map_filt = cv.drawContours(seg_map_filt, gray_contours, -1, (0, 0, 255), thickness=cv.FILLED)

    # multiply seg_map with seg_map_filt
    seg_map *= (seg_map_filt > 0)

    seg_map_PIL = Image.fromarray(seg_map)

    edge_image = cv.Canny(seg_map, 1, 254)
    # Make the edge thicker by dilating:
    kernel_dilation = np.ones((3, 3))
    edge_image = Image.fromarray(cv.dilate(edge_image, kernel_dilation, iterations=magnification * 2)).convert('RGB')

    return seg_map_PIL, edge_image
