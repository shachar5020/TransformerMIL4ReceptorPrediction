import utils
import sys
import argparse
import torch
import os
import numpy as np
import random
from pathlib import Path
import imageio
import openslide
from PIL import Image
from skimage.morphology import disk, closing
from scipy.ndimage import gaussian_filter

parser = argparse.ArgumentParser(description='WSI_MIL Slide inference with visualization')
parser.add_argument('--heatmap_folder_path', type=str, default='./example_heatmap_files/', help='folder where the heatmap was saved')
args = parser.parse_args()

highlight_color = np.array([1,0.7,0]) #yellow

im_heatmap = np.array(Image.open(os.path.join(args.heatmap_folder_path,'ScoreHeatMap.png'))).astype(np.float64) / 65535.0
im_heatmap_valid = np.array(Image.open(os.path.join(args.heatmap_folder_path,'BackgroundMap.png'))).astype(np.float64) / 255.0
mask_valid = np.array(Image.open(os.path.join(args.heatmap_folder_path,'SegMap.png')).resize((im_heatmap.shape[1], im_heatmap.shape[0]), Image.NEAREST)).astype(np.float64) / 255.0



SE = disk(10)
binary_mask = im_seg > (150 / 255.0)
im_seg_filled = closing(binary_mask, SE)
mask_valid = im_seg_filled & (im_heatmap_valid == 0)

n_bits = 16; 
precision = 1/2**n_bits
percision_half = precision/2
zero_vals_before_discr = percision_half
ones_vals_before_discr = 1-percision_half
im_heatmap_fixed = np.copy(im_heatmap)
im_heatmap_fixed[im_heatmap == 0] = zero_vals_before_discr
im_heatmap_fixed[im_heatmap == 1] = ones_vals_before_discr
im_heatmap_seg_before_sigmoid = np.log(im_heatmap_fixed / (1 - im_heatmap_fixed))

S = 20
im_heatmap_scaled = im_heatmap_seg_before_sigmoid / S + 0.5
im_heatmap_scaled = np.power(im_heatmap_scaled, 4)
im_heatmap_scaled = np.clip(im_heatmap_scaled, 0, 1)

S_smooth = 5
im_heatmap_smooth = gaussian_filter(im_heatmap_scaled * mask_valid, S_smooth) / gaussian_filter(mask_valid.astype(np.float64), S_smooth)
im_heatmap_smooth[~mask_valid] = im_heatmap_scaled[~mask_valid]
#these numbers work for the provided example and only help it look better, it can be done for other slides manually but is not mandatory
Min_C = 0.4
Max_C = 1
im_heatmap_adj = (im_heatmap_smooth-Min_C)/(Max_C-Min_C)
im_heatmap_adj[~mask_valid] = 0
alpha = (np.clip(im_heatmap_adj, 0, 1) * 255).astype(np.uint8)

#this cropping is adjusted to the example image, without it there would be a larger blank patch that can either be cropped manually later
#or through finding the right coordiantes for a desired image
ROI_x = 600
ROI_y = 2000
ROI_s_x = 1200
ROI_s_y = 400
alpha_cropped = alpha[ROI_y-1:ROI_y+ROI_s_y-1, ROI_x-1:ROI_x+ROI_s_x-1]


highlight_color_im = (np.tile(highlight_color, (alpha_cropped.shape[0], alpha_cropped.shape[1], 1)) * 255).astype(np.uint8)
highlight_image = Image.fromarray(highlight_color_im)
highlight_image.putalpha(Image.fromarray(alpha_cropped))
highlight_image.save(os.path.join(args.heatmap_folder_path, 'final_heatmap.png'), 'PNG')










im_heatmap = np.array(Image.open(os.path.join(args.heatmap_folder_path,'ScoreHeatMap.png'))).astype(np.float64) / 65535.0
mask_valid = np.array(Image.open(os.path.join(args.heatmap_folder_path,'SegMap.png')).resize((im_heatmap.shape[1], im_heatmap.shape[0]), Image.NEAREST)).astype(np.float64)


n_bits = 16; 
precision = 1/2**n_bits
percision_half = precision/2
zero_vals_before_discr = percision_half
ones_vals_before_discr = 1-percision_half
im_heatmap_fixed = np.copy(im_heatmap)
im_heatmap_fixed[im_heatmap == 0] = zero_vals_before_discr
im_heatmap_fixed[im_heatmap == 1] = ones_vals_before_discr
im_heatmap_seg_before_sigmoid = np.log(im_heatmap_fixed / (1 - im_heatmap_fixed))

S = 20
im_heatmap_scaled = im_heatmap_seg_before_sigmoid / S + 0.5
im_heatmap_scaled = np.power(im_heatmap_scaled, 4)
im_heatmap_scaled = np.clip(im_heatmap_scaled, 0, 1)

S_smooth = 5
im_heatmap_smooth = gaussian_filter(im_heatmap_scaled * mask_valid, S_smooth) / gaussian_filter(mask_valid.astype(np.float64), S_smooth)
im_heatmap_smooth[~mask_valid] = im_heatmap_scaled[~mask_valid]
#these numbers work for the provided example and only help it look better, it can be done for other slides manually but is not mandatory
Min_C = 0.4
Max_C = 1
im_heatmap_adj = (im_heatmap_smooth-Min_C)/(Max_C-Min_C)
im_heatmap_adj[~mask_valid] = 0
alpha = (np.clip(im_heatmap_adj, 0, 1) * 255).astype(np.uint8)


highlight_color_im = (np.tile(highlight_color, (alpha.shape[0], alpha.shape[1], 1)) * 255).astype(np.uint8)
highlight_image = Image.fromarray(highlight_color_im)
highlight_image.putalpha(Image.fromarray(alpha))
highlight_image.save(os.path.join(args.heatmap_folder_path, 'final_heatmap.png'), 'PNG')