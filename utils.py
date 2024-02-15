import numpy as np
from PIL import Image
from matplotlib import image as plt_image
import os
import pandas as pd
import glob
from random import sample, seed
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
from typing import List, Tuple
from xlrd.biffh import XLRDError
from zipfile import BadZipFile
import matplotlib.pyplot as plt
from math import isclose
from argparse import Namespace as argsNamespace
from shutil import copyfile
from datetime import date
import inspect
import torch.nn.functional as F
import multiprocessing
from tqdm import tqdm
import sys
from PIL import ImageFile
from fractions import Fraction
import openslide
import logging
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def unnormalize(patches):
    mean = torch.tensor([0.8998, 0.8253, 0.9357]).reshape(1,3,1,1)
    std = torch.tensor([0.1125, 0.1751, 0.0787]).reshape(1,3,1,1)
    return ((patches*std+mean))
    
def normalize(patches):
    mean = torch.tensor([0.8998, 0.8253, 0.9357])
    std = torch.tensor([0.1125, 0.1751, 0.0787])
    transform = transforms.Compose([transforms.Normalize(
                                                  mean=(mean[0], mean[1], mean[2]),
                                                  std=(std[0], std[1], std[2]))
                                              ])
    return transform(patches)


def get_slide_magnification(slide, data_format):
    mag_dict = {'svs': 'aperio.AppMag', 'ndpi': 'hamamatsu.SourceLens', 'mrxs': 'openslide.objective-power',
                'tiff': 'tiff.Software'}
    try:
        mag = int(
                float(slide.properties[mag_dict[data_format]]))
    except:
        logging.info(f'{data_format}')
        logging.info(f'{slide.properties}')
        a=sksk
        mag = 10
        logging.info('failed to get magnification from slide, using default value')
    return mag
        

def chunks(list: List, length: int):
    new_list = [list[i * length:(i + 1) * length] for i in range((len(list) + length - 1) // length)]
    return new_list


def get_optimal_slide_level(slide, magnification, desired_mag, tile_size):
    # downsample needed for each dimension (reflected by level_downsamples property)
    desired_downsample = magnification / desired_mag

    if desired_downsample < 1:  # upsample
        best_slide_level = 0
        level_0_tile_size = int(desired_downsample * tile_size)
        adjusted_tile_size = level_0_tile_size
    else:
        level = -1
        best_next_level = np.argmin(slide.level_downsamples)
        level_downsample = slide.level_downsamples[best_next_level]
        for index, downsample in enumerate(slide.level_downsamples):
            if isclose(desired_downsample, downsample, rel_tol=1e-3):
                level = index
                level_downsample = 1
                break

            elif downsample < desired_downsample:
                best_next_level = index
                level_downsample = desired_downsample / slide.level_downsamples[best_next_level]

        adjusted_tile_size = int(tile_size * level_downsample)
        best_slide_level = level if level > best_next_level else best_next_level
        level_0_tile_size = int(desired_downsample * tile_size)

    return best_slide_level, adjusted_tile_size, level_0_tile_size


def _choose_data(grid_list: list,
                 slide: openslide.OpenSlide,
                 how_many: int,
                 magnification: int | float,
                 tile_size: int = 256,
                 print_timing: bool = False,
                 desired_mag: int = 20,
                 random_shift: bool = True):
    """
    This function choose and returns data to be held by DataSet.
    The function is in the PreLoad Version. It works with slides already loaded to memory.

    :param grid_list: A list of all grids for this specific slide
    :param slide: An OpenSlide object of the slide.
    :param how_many: how_many tiles to return from the slide.
    :param magnification: The magnification of level 0 of the slide
    :param tile_size: Desired tile size from the slide at the desired magnification
    :param print_timing: Do or don't collect timing for this procedure
    :param desired_mag: Desired Magnification of the tiles/slide.
    :return:
    """
    
    

    best_slide_level, adjusted_tile_size, level_0_tile_size = get_optimal_slide_level(slide, magnification, desired_mag,
                                                                                      tile_size)

    # Choose locations from the grid list:
    loc_num = len(grid_list)
    try:
        idxs = sample(range(loc_num), how_many)
    except ValueError:
        raise ValueError('Requested more tiles than available by the grid list')

    locs = [grid_list[idx] for idx in idxs]
    image_tiles, time_list, labels = _get_tiles(slide=slide,
                                                locations=locs,
                                                tile_size_level_0=level_0_tile_size,
                                                adjusted_tile_sz=adjusted_tile_size,
                                                output_tile_sz=tile_size,
                                                best_slide_level=best_slide_level,
                                                print_timing=print_timing,
                                                random_shift=random_shift)

    return image_tiles, time_list, labels, locs


def _get_tiles(slide: openslide.OpenSlide,
               locations: List[Tuple],
               tile_size_level_0: int,
               adjusted_tile_sz: int,
               output_tile_sz: int,
               best_slide_level: int,
               print_timing: bool = False,
               random_shift: bool = False,
               oversized_HC_tiles: bool = False):
    """
    This function extract tiles from the slide.
    :param slide: OpenSlide object containing a slide
    :param locations: locations of te tiles to be extracted
    :param tile_size_level_0: tile size adjusted for level 0
    :param adjusted_tile_sz: tile size adjusted for best_level magnification
    :param output_tile_sz: output tile size needed
    :param best_slide_level: best slide level to get tiles from
    :param print_timing: collect time profiling data ?
    :return:
    """

    # preallocate list of images
    empty_image = Image.fromarray(np.uint8(np.zeros((output_tile_sz, output_tile_sz, 3))))
    tiles_PIL = [empty_image] * len(locations)

    start_gettiles = time.time()

    if oversized_HC_tiles:
        adjusted_tile_sz *= 2
        output_tile_sz *= 2
        tile_shifting = (tile_size_level_0 // 2, tile_size_level_0 // 2)

    # get localized labels
    labels = np.zeros(len(locations)) - 1

    for idx, loc in enumerate(locations):
        if random_shift:
            tile_shifting = sample(range(-tile_size_level_0 // 2, tile_size_level_0 // 2), 2)

        if random_shift or oversized_HC_tiles:
            new_loc_init = {'Top': loc[0] - tile_shifting[0],
                            'Left': loc[1] - tile_shifting[1]}
            new_loc_end = {'Bottom': new_loc_init['Top'] + tile_size_level_0,
                           'Right': new_loc_init['Left'] + tile_size_level_0}
            if new_loc_init['Top'] < 0:
                new_loc_init['Top'] += abs(new_loc_init['Top'])
            if new_loc_init['Left'] < 0:
                new_loc_init['Left'] += abs(new_loc_init['Left'])
            if new_loc_end['Bottom'] > slide.dimensions[1]:
                delta_Height = new_loc_end['Bottom'] - slide.dimensions[1]
                new_loc_init['Top'] -= delta_Height
            if new_loc_end['Right'] > slide.dimensions[0]:
                delta_Width = new_loc_end['Right'] - slide.dimensions[0]
                new_loc_init['Left'] -= delta_Width
        else:
            new_loc_init = {'Top': loc[0],
                            'Left': loc[1]}

        try:
            image = slide.read_region((new_loc_init['Left'], new_loc_init['Top']), best_slide_level,
                                      (adjusted_tile_sz, adjusted_tile_sz)).convert('RGB')
        except:
            logging.info('failed to read slide {} in location {},{}'.format(slide._filename, loc[1], loc[0]))
            logging.info('taking blank patch instead')
            image = Image.fromarray(np.zeros([adjusted_tile_sz, adjusted_tile_sz, 3], dtype=np.uint8))


        if adjusted_tile_sz != output_tile_sz:
            image = image.resize((output_tile_sz, output_tile_sz))

        tiles_PIL[idx] = image

    end_gettiles = time.time()

    if print_timing:
        time_list = [0, (end_gettiles - start_gettiles) / len(locations)]
    else:
        time_list = [0]

    return tiles_PIL, time_list, labels


def device_gpu_cpu():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info('Using CUDA')
    else:
        device = torch.device('cpu')
        logging.info('Using cpu')
    return device


def get_cpu():
    cpu = len(os.sched_getaffinity(0))
    return cpu


def run_data(experiment: str = None,
             test_fold: int = 1,
             transform_type: str = 'none',
             tile_size: int = 256,
             tiles_per_bag: int = 50,
             num_bags: int = 1,
             DataSet_name: list = ['TCGA'],
             DataSet_test_name: list = [None],
             DataSet_size: tuple = None,
             DataSet_Slide_magnification: int = None,
             epoch: int = None,
             model: str = None,
             transformation_string: str = None,
             Receptor: str = None,
             MultiSlide: bool = False,
             test_mean_auc: float = None,
             is_per_patient: bool = False,
             is_last_layer_freeze: bool = False,
             is_repeating_data: bool = False,
             data_limit: int = None,
             free_bias: bool = False,
             carmel_only: bool = False,
             CAT_only: bool = False,
             Remark: str = '',
             Class_Relation: float = None,
             learning_rate: float = -1,
             weight_decay: float = -1,
             censored_ratio: float = -1,
             combined_loss_weights: list = [],
             receptor_tumor_mode: int = -1,
             is_domain_adaptation: bool = False):
    """
    This function writes the run data to file
    :param experiment:
    :param from_epoch:
    :param MultiSlide: Describes if tiles from different slides with same class are mixed in the same bag
    :return:
    """

    location_prefix = './'
    run_file_name = 'runs/run_data.xlsx'


    if os.path.isfile(run_file_name):
        read_success = False
        read_attempts = 0
        while (not read_success) and (read_attempts < 10):
            try:
                run_DF = pd.read_excel(run_file_name)
                read_success = True
            except (XLRDError, BadZipFile):
                print('Couldn\'t open file {}, check if file is corrupt'.format(run_file_name))
                return
            except ValueError:
                print('run_data file is being used, retrying in 5 seconds')
                read_attempts += 1
                time.sleep(5)
        if not read_success:
            print('Couldn\'t open file {} after 10 attempts'.format(run_file_name))
            return

        try:
            run_DF.drop(labels='Unnamed: 0', axis='columns', inplace=True)
        except KeyError:
            pass

        run_DF_exp = run_DF.set_index('Experiment', inplace=False)
    else:
        run_DF = pd.DataFrame()

    # If a new experiment is conducted:
    if experiment is None:
        if os.path.isfile(run_file_name):
            experiment = run_DF_exp.index.values.max() + 1
        else:
            experiment = 1

        location = os.path.join(os.path.abspath(os.getcwd()), 'runs',
                                'Exp_' + str(experiment) + '-' + Receptor + '-TestFold_' + str(test_fold))
        if type(DataSet_name) is not list:
            DataSet_name = [DataSet_name]

        if type(DataSet_test_name) is not list:
            DataSet_test_name = [DataSet_test_name]

        run_dict = {'Experiment': experiment,
                    'Start Date': str(date.today()),
                    'Test Fold': test_fold,
                    'Transformations': transform_type,
                    'Tile Size': tile_size,
                    'Tiles Per Bag': tiles_per_bag,
                    'MultiSlide Per Bag': MultiSlide,
                    'No. of Bags': num_bags,
                    'Location': location,
                    'DataSet': ' / '.join(DataSet_name),
                    'Test Set (DataSet)': ' / '.join(DataSet_test_name) if DataSet_test_name[0] != None else None,
                    'Receptor': Receptor,
                    'Model': 'None',
                    'Last Epoch': 0,
                    'Transformation String': 'None',
                    'Desired Slide Magnification': DataSet_Slide_magnification,
                    'Per Patient Training': is_per_patient,
                    'Last Layer Freeze': is_last_layer_freeze,
                    'Repeating Data': is_repeating_data,
                    'Data Limit': data_limit,
                    'Free Bias': free_bias,
                    'Carmel Only': carmel_only,
                    'Using Feature from CAT model alone': CAT_only,
                    'Remark': Remark,
                    'Class Relation': Class_Relation,
                    'Learning Rate': learning_rate,
                    'Weight Decay': weight_decay,
                    'Censor Ratio': censored_ratio,
                    'Combined Loss Weights': [Fraction(combined_loss_weights[item]).limit_denominator() for item in
                                              range(len(combined_loss_weights))],
                    'Receptor + is_Tumor Train Mode': receptor_tumor_mode,
                    'Trained with Domain Adaptation': is_domain_adaptation
                    }
        run_DF = run_DF.append([run_dict], ignore_index=True)
        if not os.path.isdir('runs'):
            os.mkdir('runs')

        if not os.path.isdir(location):
            os.mkdir(location)

        run_DF.to_excel(run_file_name)
        print('Created a new Experiment (number {}). It will be saved at location: {}'.format(experiment, location))

        # backup for run_data
        backup_dir = os.path.join(os.path.abspath(os.getcwd()), 'runs', 'run_data_backup')
        print(backup_dir)
        if not os.path.isdir(backup_dir):
            os.mkdir(backup_dir)
            print('backup dir created')
        try:
            run_DF.to_excel(os.path.join(backup_dir, 'run_data_exp' + str(experiment) + '.xlsx'))
        except:
            raise IOError('failed to back up run_data, please check there is enough storage')

        return {'Location': location,
                'Experiment': experiment
                }

    elif experiment is not None and epoch is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Last Epoch'] = epoch
        run_DF.to_excel(run_file_name)

    elif experiment is not None and model is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Model'] = model
        run_DF.to_excel(run_file_name)

    elif experiment is not None and transformation_string is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Transformation String'] = transformation_string
        run_DF.to_excel(run_file_name)

    elif experiment is not None and DataSet_size is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Train DataSet Size'] = DataSet_size[0]
        run_DF.at[index, 'Test DataSet Size'] = DataSet_size[1]
        run_DF.to_excel(run_file_name)

    elif experiment is not None and DataSet_Slide_magnification is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Desired Slide Magnification'] = DataSet_Slide_magnification
        run_DF.to_excel(run_file_name)

    elif experiment is not None and test_mean_auc is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'TestSet Mean AUC'] = test_mean_auc
        run_DF.to_excel(run_file_name)

    # In case we want to continue from a previous training session
    else:
        location = run_DF_exp.loc[[experiment], ['Location']].values[0][0]
        test_fold = int(run_DF_exp.loc[[experiment], ['Test Fold']].values[0][0])
        transformations = run_DF_exp.loc[[experiment], ['Transformations']].values[0][0]
        tile_size = int(run_DF_exp.loc[[experiment], ['Tile Size']].values[0][0])
        tiles_per_bag = int(run_DF_exp.loc[[experiment], ['Tiles Per Bag']].values[0][0])
        num_bags = int(run_DF_exp.loc[[experiment], ['No. of Bags']].values[0][0])
        DataSet_name = str(run_DF_exp.loc[[experiment], ['DataSet']].values[0][0])
        Receptor = str(run_DF_exp.loc[[experiment], ['Receptor']].values[0][0])
        MultiSlide = str(run_DF_exp.loc[[experiment], ['MultiSlide Per Bag']].values[0][0])
        model_name = str(run_DF_exp.loc[[experiment], ['Model']].values[0][0])
        Desired_Slide_magnification = int(run_DF_exp.loc[[experiment], ['Desired Slide Magnification']].values[0][0])
        try:
            receptor_tumor_mode = run_DF_exp.loc[[experiment], ['Receptor + is_Tumor Train Mode']].values[0][0]
            receptor_tumor_mode = convert_value_to_integer(receptor_tumor_mode)
            free_bias = bool(run_DF_exp.loc[[experiment], ['Free Bias']].values[0][0])
            CAT_only = bool(run_DF_exp.loc[[experiment], ['Using Feature from CAT model alone']].values[0][0])
            Class_Relation = float(run_DF_exp.loc[[experiment], ['Class Relation']].values[0][0])
        except:
            receptor_tumor_mode = np.nan
            free_bias = np.nan
            CAT_only = np.nan
            Class_Relation = np.nan

        if sys.platform == 'linux':
            if location.split('/')[0] == 'runs':
                location = location_prefix + location

        return {'Location': location,
                'Test Fold': test_fold,
                'Transformations': transformations,
                'Tile Size': tile_size,
                'Tiles Per Bag': tiles_per_bag,
                'Num Bags': num_bags,
                'Dataset Name': DataSet_name,
                'Receptor': Receptor,
                'MultiSlide': MultiSlide,
                'Model Name': model_name,
                'Desired Slide Magnification': Desired_Slide_magnification,
                'Free Bias': free_bias,
                'CAT Only': CAT_only,
                'Class Relation': Class_Relation,
                'Receptor + is_Tumor Train Mode': receptor_tumor_mode
                }



def save_code_files(args: argsNamespace, train_DataSet):
    """
    This function saves the code files and argparse data to a Code directory within the run path.
    :param args: argsparse Namespace of the run.
    :return:
    """
    code_files_path = os.path.join(args.output_dir, 'Code')
    # Get the filename that called this function
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    full_filename = module.__file__
    args.run_file = full_filename.split('/')[-1]

    args_dict = vars(args)

    # Add Grid Data:
    data_dict = args_dict
    if train_DataSet is not None:

        if train_DataSet.train_type != 'Features':
            grid_meta_data_file = os.path.join(train_DataSet.DataSet_location,
                                               'Grids_' + str(train_DataSet.desired_magnification),
                                               'production_meta_data.xlsx')
            if os.path.isfile(grid_meta_data_file):
                grid_data_DF = pd.read_excel(grid_meta_data_file)
                grid_dict = grid_data_DF.to_dict('split')
                grid_dict['dataset'] = train_DataSet.DataSet_location
                grid_dict.pop('index')
                grid_dict.pop('columns')
                data_dict['grid'] = grid_dict

    data_DF = pd.DataFrame([data_dict]).transpose()

    if not os.path.isdir(code_files_path):
        os.mkdir(code_files_path)
    data_DF.to_excel(os.path.join(code_files_path, 'run_arguments.xlsx'))
    py_files = glob.glob('*.py') # Get all .py files in the code path
    for _, file in enumerate(py_files):
        copyfile(file, os.path.join(code_files_path, os.path.basename(file)))



def get_label(target, multi_target=False):
    if multi_target:
        label = []
        for t in target:
            label.append(get_label(t))
        return label
    else:
        if target == 'Positive':
            return [1]
        elif target == 'Negative':
            return [0]
        elif ((isinstance(target, int) or isinstance(target, float)) and not np.isnan(target)) or (
                (target.__class__ == str) and (target.isnumeric())):  # support multiclass
            return [int(target)]
        else:  # unknown
            return [-1]



def start_log(args, to_file=False):
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    if to_file:
        logfile = os.path.join(args.output_dir, 'log.txt')
        os.makedirs(args.output_dir, exist_ok=True)
        handlers = [stream_handler,
                    logging.FileHandler(filename=logfile)]
    else:
        handlers = [stream_handler]
    logging.basicConfig(format='%(message)s',
                        level=logging.INFO,
                        handlers=handlers)
    logging.info('*** START ARGS ***')
    for k, v in vars(args).items():
        logging.info('{}: {}'.format(k, v))
    logging.info('*** END ARGS ***')
