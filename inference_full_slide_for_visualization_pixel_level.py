import utils
from utils import get_patches_with_overlap
import sys
import argparse
from models import preact_resnet
import torch
import os
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path
import imageio
import openslide
from PIL import Image
from skimage.morphology import disk, closing
from scipy.ndimage import gaussian_filter

random.seed(0)
parser = argparse.ArgumentParser(description='WSI_MIL Slide inference with visualization')
parser.add_argument('--checkpoint_path', type=str, default='./runs/Exp_1-ER-TestFold_1/Model_CheckPoints/model_data_Last_Epoch.pt', help='path to the model checkpoint')
parser.add_argument('--slide_path', type=str, default='./example_data_dir/TCGA-A8-A09T-01A-01-TS1.46733e61-04d6-4b8f-97ba-d115e9208e9b.svs', help='slide to use')
parser.add_argument('--segmap_path', type=str, default='./example_data_dir/SegData/SegMaps/TCGA-A8-A09T-01A-01-TS1.46733e61-04d6-4b8f-97ba-d115e9208e9b_SegMap.jpg', help='path to the slide segmentation map')
parser.add_argument('--output_path', type=str, default='./Visualizations', help='folder to save the heatmap')

args = parser.parse_args()

slide_file = args.slide_path
segMap_file = args.segmap_path

DEVICE = utils.device_gpu_cpu()
downsample_rate = 8
frame_size = int(64 / downsample_rate)

model = preact_resnet.PreActResNet50()

# loading model parameters from the specific epoch
model_data_loaded = torch.load(os.path.join(args.checkpoint_path), map_location='cpu')
model.load_state_dict(model_data_loaded['model_state_dict'])
model.eval()

model.is_HeatMap = True


new_slide = True


# Create folders to save the data:

path_for_output = os.path.join(args.output_path,args.slide_path.split('/')[-1])
print('path is:' + path_for_output)
Path(path_for_output).mkdir(parents=True, exist_ok=True)

slide = openslide.OpenSlide(slide_file)
downsample = slide.level_downsamples[2]


with torch.no_grad():
    for batch_idx, (data, window_top_left_in_level, window_size, equivalent_grid_size) in enumerate(tqdm(get_patches_with_overlap(slide_file, segMap_file, rate = downsample_rate, size = 2048))):

        if new_slide:
            equivalent_slide_heat_map = np.ones((equivalent_grid_size)) * (-1)  # This heat map should be filled with the weights.
            new_slide = False
        data = data.to(DEVICE)

        model.to(DEVICE)
        output = torch.sigmoid(model(data)).squeeze().cpu().numpy()

        
        taken_window_top_left = (int(window_top_left_in_level[0]//downsample) + frame_size, int(window_top_left_in_level[1]//downsample) + frame_size)
        taken_window = output[frame_size:-frame_size, frame_size:-frame_size]
        taken_window_size = taken_window.shape
        
        try:
            equivalent_slide_heat_map[taken_window_top_left[0]:taken_window_top_left[0] + taken_window_size[0], taken_window_top_left[1]:taken_window_top_left[1] + taken_window_size[1]] = taken_window
        except:
            shape = equivalent_slide_heat_map[taken_window_top_left[0]:taken_window_top_left[0] + taken_window_size[0], taken_window_top_left[1]:taken_window_top_left[1] + taken_window_size[1]].shape
            taken_window = taken_window[:shape[0],:shape[1]]
            equivalent_slide_heat_map[taken_window_top_left[0]:taken_window_top_left[0] + taken_window_size[0], taken_window_top_left[1]:taken_window_top_left[1] + taken_window_size[1]] = taken_window
            




background_map = (equivalent_slide_heat_map == -1)
equivalent_slide_heat_map[equivalent_slide_heat_map == -1] = 0


height = slide.dimensions[1]
width = slide.dimensions[0]
mag_thumb = 5
height_thumb = np.min((int(height / mag_thumb), 10000))
width_thumb = int(width / height * height_thumb)

segmap = np.array(Image.open(segMap_file).resize((background_map.shape[1], background_map.shape[0])))
segmap = segmap * (background_map == 0)

heatmap_file = os.path.join(path_for_output,'ScoreHeatMap.png')
seg_file = os.path.join(path_for_output,'SegMap.png')
imageio.imwrite(heatmap_file, (equivalent_slide_heat_map * 65535).astype(np.uint16))
imageio.imwrite(seg_file, segmap)


print('Done !')