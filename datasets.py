import logging
import os
import pickle
import sys
import time
from glob import glob
from random import choices, sample, seed
from typing import List

import cv2
import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from transformations import define_transformations

from utils import (_choose_data, _get_tiles,
                   chunks, get_label,
                   get_optimal_slide_level, get_slide_magnification)


class WSI_Master_Dataset(Dataset):

    def __init__(self,
                 DataSet_location: str = './TCGA',
                 tile_size: int = 256,
                 bag_size: int = 50,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 infer_folds: List = [None],
                 train: bool = True,
                 print_timing: bool = False,
                 transform_type: str = 'flip',
                 train_type: str = 'MASTER',
                 color_param: float = 0.1,
                 n_tiles: int = 10,
                 test_time_augmentation: bool = False,
                 desired_slide_magnification: int = 10,
                 slide_repetitions: int = 1,
                 RAM_saver: bool = False):



        logging.info(
            'Initializing {} DataSet....'.format('Train' if train else 'Test'))
        self.DataSet_location = DataSet_location
        self.desired_magnification = desired_slide_magnification
        self.tile_size = tile_size
        self.target_kind = target_kind
        self.test_fold = 'test' if test_fold == 0 else test_fold
        self.bag_size = bag_size
        self.train = train
        self.print_time = print_timing
        self.train_type = train_type
        self.color_param = color_param
        locations_list = []



        slide_meta_data_file = os.path.join(DataSet_location,
                                            'slides_data.xlsx')
        grid_meta_data_file = os.path.join(
            DataSet_location, 'Grids_' + str(self.desired_magnification),
            'Grid_data.xlsx')

        slide_meta_data_DF = pd.read_excel(slide_meta_data_file)
        grid_meta_data_DF = pd.read_excel(grid_meta_data_file)

        self.meta_data_DF = pd.DataFrame({
            **slide_meta_data_DF.set_index('file').to_dict(),
            **grid_meta_data_DF.set_index('file').to_dict()
        })


        self.meta_data_DF.reset_index(inplace=True)
        self.meta_data_DF.rename(columns={'index': 'file'}, inplace=True)
        
        if self.target_kind == 'OR':
            PR_targets = list(self.meta_data_DF['PR status'])
            ER_targets = list(self.meta_data_DF['ER status'])
            all_targets = ['Missing Data'] * len(ER_targets)
            for ii, (PR_target,
                     ER_target) in enumerate(zip(PR_targets, ER_targets)):
                if (PR_target == 'Positive' or ER_target == 'Positive'):
                    all_targets[ii] = 'Positive'
                elif (PR_target == 'Negative'
                      or ER_target == 'Negative'):  # avoid 'Missing Data'
                    all_targets[ii] = 'Negative'

        else:
            all_targets = list(self.meta_data_DF[self.target_kind +
                                                 ' status'])

        excess_block_slides = set()

        # We'll use only the valid slides - the ones with a Negative or Positive label. (Some labels have other values)
        # Let's compute which slides are these:
        all_targets_string = []
        for target in all_targets:
            if (type(target) == int
                    or type(target) == float) and not np.isnan(target):
                all_targets_string.append(str(int(target)))
            else:
                all_targets_string.append(str(target))

        valid_slide_indices1 = \
            np.where(np.isin(np.array(all_targets_string),
                             ['Positive', 'Negative']) == True)[0]
        valid_slide_indices2 = np.where(
            np.isin(np.array(all_targets_string),
                    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            == True)[0]
        valid_slide_indices = np.hstack(
            (valid_slide_indices1, valid_slide_indices2))

        # inference on unknown labels in case of (blind) test inference or Batched_Full_Slide_Inference_Dataset
        if len(valid_slide_indices) == 0:
            valid_slide_indices = np.arange(
                len(all_targets))  # take all slides

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF[
            'Total tiles - ' + str(self.tile_size) + ' compatible @ X' +
            str(self.desired_magnification)] == -1])
        # Remove slides with 0 tiles:
        slides_with_0_tiles = set(self.meta_data_DF.index[self.meta_data_DF[
            'Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X' +
            str(self.desired_magnification)] == 0])

        if 'bad segmentation' in self.meta_data_DF.columns:
            slides_with_bad_seg = set(self.meta_data_DF.index[
                self.meta_data_DF['bad segmentation'] == 1])
        else:
            slides_with_bad_seg = set()

        # Define number of tiles to be used
        if train_type == 'REG':
            n_minimal_tiles = n_tiles
        else:
            n_minimal_tiles = self.bag_size

        slides_with_few_tiles = set(self.meta_data_DF.index[self.meta_data_DF[
            'Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X' +
            str(self.desired_magnification)] < n_minimal_tiles])
        # FIXME: find a way to use slides with less than the minimal amount of slides. and than delete the following if.
        if len(slides_with_few_tiles) > 0:
            logging.info(
                '{} Slides were excluded from DataSet because they had less than {} available tiles or are non legitimate for training'
                .format(len(slides_with_few_tiles), n_minimal_tiles))

        valid_slide_indices = np.array(
            list(
                set(valid_slide_indices) - slides_without_grid -
                slides_with_few_tiles - slides_with_0_tiles -
                slides_with_bad_seg -
                excess_block_slides))

        if RAM_saver:
            # randomly select 1/4 of the slides
            shuffle_factor = 4
            valid_slide_indices = np.random.choice(
                valid_slide_indices,
                round(len(valid_slide_indices) / shuffle_factor),
                replace=False)

        # The train set should be a combination of all sets except the test set and validation set:
        fold_column_name = 'test fold idx'
        forbidden_values = ['1', '2', '3', '4', '5']
        values_in_folds = list(self.meta_data_DF[fold_column_name].unique())
        
        for item in values_in_folds:
            if item in forbidden_values:
                raise ValueError(f'fold column contains forbidden values: {values_in_folds}') 
        if self.train_type in ['REG', 'MIL']:
            if self.train:
                folds = list(self.meta_data_DF[fold_column_name].unique())
                if test_fold != -1:
                    folds.remove(self.test_fold)
                if 'test' in folds:
                    folds.remove('test')
                if 'val' in folds:
                    folds.remove('val')
                if 'Missing Data' in folds:
                    folds.remove('Missing Data')
                for item in folds:
                    if item not in [1,2,3,4,5,1.0,2.0,3.0,4.0,5.0]:
                        folds.remove(item)
            else:
                if test_fold != -1:
                    folds = [self.test_fold, 'val']
                else:
                    folds = []

        elif self.train_type == 'Infer':
            if 0 in infer_folds:
                infer_folds[infer_folds.index(0)] = 'test'
            folds = infer_folds
        elif self.train_type == 'Infer_All_Folds':
            folds = list(self.meta_data_DF[fold_column_name].unique())
        else:
            raise ValueError('Variable train_type is not defined')

        self.folds = folds

        if type(folds) is int:
            folds = [folds]

        correct_folds = self.meta_data_DF[fold_column_name][
            valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])
        
        all_image_file_names = list(self.meta_data_DF['file'])

        all_in_fold = list(self.meta_data_DF[fold_column_name])
        all_tissue_tiles = list(
            self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) +
                              ' compatible @ X' +
                              str(self.desired_magnification)])

        if train_type in ['Infer']:

            self.valid_slide_indices = valid_slide_indices
            self.all_tissue_tiles = all_tissue_tiles
            self.all_image_file_names = all_image_file_names
            self.all_targets = all_targets

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.slides = []
        self.grid_lists = []
        



        for _, index in enumerate(tqdm(valid_slide_indices)):
            self.image_file_names.append(all_image_file_names[index])
            self.image_path_names.append(
                self.DataSet_location)
            self.in_fold.append(all_in_fold[index])
            self.tissue_tiles.append(all_tissue_tiles[index])
            self.target.append(all_targets[index])

            # Preload slides - improves speed during training.
            grid_file = []
            image_file = []
            try:
                image_file = os.path.join(
                    self.DataSet_location,
                    all_image_file_names[index])
                if self.train_type in ['Infer']:
                    self.slides.append(image_file)
                else:
                    self.slides.append(
                        openslide.open_slide(image_file))

                basic_file_name = '.'.join(
                    all_image_file_names[index].split('.')[:-1])

                grid_file = os.path.join(
                    self.DataSet_location,
                    'Grids_' + str(self.desired_magnification),
                    basic_file_name + '--tlsz' + str(self.tile_size) +
                    '.data')
                with open(grid_file, 'rb') as filehandle:
                    grid_list = pickle.load(filehandle)
                    self.grid_lists.append(grid_list)
            except FileNotFoundError:
                raise FileNotFoundError(
                    'Couldn\'t open slide {} or its Grid file {}'.format(
                        image_file, grid_file))

        norm_type = 'standard'
        self.transform = define_transformations(transform_type, self.train,
                                                self.tile_size,
                                                self.color_param, norm_type)
        if train_type == 'REG':
            self.factor = n_tiles
            self.real_length = int(self.__len__() / self.factor)
        elif train_type == 'MIL':
            self.factor = 1
            self.real_length = self.__len__()
        if train is False and test_time_augmentation:
            self.factor = 4
            self.real_length = int(self.__len__() / self.factor)


        self.random_shift = True if (self.train) else False

    def __len__(self):
        return len(self.target) * self.factor

    def __getitem__(self, idx):
        start_getitem = time.time()
        idx = idx % self.real_length
        slide = self.slides[idx]
        data_format = self.image_file_names[idx].split('.')[-1]

        tiles, time_list, label, _ = _choose_data(
            grid_list=self.grid_lists[idx],
            slide=slide,
            how_many=self.bag_size,
            magnification=get_slide_magnification(slide, data_format),
            tile_size=self.tile_size,
            print_timing=self.print_time,
            desired_mag=self.desired_magnification,
            random_shift=self.random_shift)

        label = get_label(self.target[idx], False)
        label = torch.LongTensor(label)

        # X will hold the images after all the transformations
        X = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])
        

        start_aug = time.time()
        for i in range(self.bag_size):
            X[i] = self.transform(tiles[i])

        images = 0

        aug_time = (time.time() - start_aug) / self.bag_size
        total_time = time.time() - start_getitem

        if self.print_time:
            time_list = (time_list[0], time_list[1], aug_time, total_time)
            time_dict = {
                'Average time to extract a tile': time_list[1],
                'Augmentation time': aug_time,
                'Total time': total_time
            }
        else:
            time_list = [0]
            time_dict = {
                'Average time to extract a tile': 0,
                'Augmentation time': 0,
                'Total time': 0
            }


        target_binary = [-1]

        target_binary = torch.LongTensor(target_binary)

        return {
            'Data':
            X,
            'Target':
            label,
            'Time List':
            time_list,
            'Time dict':
            time_dict,
            'File Names':
            self.image_file_names[idx],
            'Images':
            images,
            'Target Binary':
            target_binary,
            'is_Train':
            self.train
        }


class WSI_REGdataset(WSI_Master_Dataset):

    def __init__(self,
                 DataSet_location: str = './TCGA',
                 tile_size: int = 256,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 print_timing: bool = False,
                 transform_type: str = 'flip',
                 color_param: float = 0.1,
                 n_tiles: int = 10,
                 desired_slide_magnification: int = 10,
                 RAM_saver: bool = False):
        super(WSI_REGdataset, self).__init__(
            DataSet_location=DataSet_location,
            tile_size=tile_size,
            bag_size=1,
            target_kind=target_kind,
            test_fold=test_fold,
            train=train,
            print_timing=print_timing,
            transform_type=transform_type,
            train_type='REG',
            color_param=color_param,
            n_tiles=n_tiles,
            desired_slide_magnification=desired_slide_magnification,
            RAM_saver=RAM_saver)



    def __getitem__(self, idx):
        data_dict = super(WSI_REGdataset, self).__getitem__(idx=idx)
        X = data_dict['Data']
        X = torch.reshape(X, (3, self.tile_size, self.tile_size))
        

        return {
            'Data': X,
            'Target': data_dict['Target'],
            'Target Binary': data_dict['Target Binary'],
            'Time List': data_dict['Time List'],
            'Time dict': data_dict['Time dict'],
            'File Names': data_dict['File Names'],
            'Images': data_dict['Images'],
            'is_Train': data_dict['is_Train']
        }


class Infer_Dataset(WSI_Master_Dataset):

    def __init__(self,
                 DataSet_location: str = './TCGA',
                 tile_size: int = 256,
                 tiles_per_iter: int = 500,
                 target_kind: str = 'ER',
                 folds: List = [1],
                 num_tiles: int = 500,
                 desired_slide_magnification: int = 10,
                 resume_slide: int = 0,
                 chosen_seed: int = None):
        super(Infer_Dataset, self).__init__(
            DataSet_location=DataSet_location,
            tile_size=tile_size,
            bag_size=None,
            target_kind=target_kind,
            test_fold=1,
            infer_folds=folds,
            train=True,
            print_timing=False,
            transform_type='none',
            train_type='Infer',
            desired_slide_magnification=desired_slide_magnification)

        self.tiles_per_iter = tiles_per_iter
        self.folds = folds
        self.num_tiles = []
        self.slide_grids = []
        self.grid_lists = []

        ind = 0
        slide_with_not_enough_tiles = 0

        self.valid_slide_indices = self.valid_slide_indices[resume_slide:]
        self.tissue_tiles = self.tissue_tiles[resume_slide:]
        self.image_file_names = self.image_file_names[resume_slide:]
        self.image_path_names = self.image_path_names[resume_slide:]
        self.slides = self.slides[resume_slide:]
        self.target = self.target[resume_slide:]

        if chosen_seed is not None:
            seed(chosen_seed)
            np.random.seed(chosen_seed)
            torch.manual_seed(chosen_seed)
        for _, slide_num in enumerate(self.valid_slide_indices):
            if num_tiles <= self.all_tissue_tiles[
                    slide_num] and self.all_tissue_tiles[slide_num] > 0:
                self.num_tiles.append(num_tiles)
            else:
                self.num_tiles.append(int(
                    self.all_tissue_tiles[slide_num]))
                slide_with_not_enough_tiles += 1

            which_patches = sample(range(int(self.tissue_tiles[ind])),
                                   self.num_tiles[-1])
            patch_ind_chunks = chunks(which_patches, self.tiles_per_iter)
            self.slide_grids.extend(patch_ind_chunks)

            basic_file_name = '.'.join(
                self.image_file_names[ind].split('.')[:-1])
            grid_file = os.path.join(
                self.image_path_names[ind],
                'Grids_' + str(self.desired_magnification),
                basic_file_name + '--tlsz' + str(self.tile_size) +
                '.data')
            with open(grid_file, 'rb') as filehandle:
                grid_list = pickle.load(filehandle)
                self.grid_lists.append(grid_list)

            ind += 1

        print('There are {} slides with less than {} tiles'.format(
            slide_with_not_enough_tiles, num_tiles))

        # The following properties will be used in the __getitem__ function
        self.tiles_to_go = None
        self.slide_num = -1
        self.current_file = None


        print(
            '{} Slides, with X{} magnification. {} tiles per iteration, {} iterations to complete full inference'
            .format(len(self.image_file_names), self.desired_magnification,
                    self.tiles_per_iter, self.__len__()))

    def __len__(self):
        return int(
            np.ceil(np.array(self.num_tiles) / self.tiles_per_iter).sum())

    def __getitem__(self, idx):
        start_getitem = time.time()
        if self.tiles_to_go is None:
            self.slide_num += 1
            self.tiles_to_go = self.num_tiles[self.slide_num]
            self.slide_name = self.image_file_names[self.slide_num]
            self.current_slide = openslide.open_slide(
                self.slides[self.slide_num])

            self.initial_num_patches = self.num_tiles[self.slide_num]
            data_format = self.slide_name.split('.')[-1]
            self.best_slide_level, self.adjusted_tile_size, self.level_0_tile_size = \
                get_optimal_slide_level(self.current_slide, get_slide_magnification(self.current_slide, data_format),
                                        self.desired_magnification, self.tile_size)

        label = get_label(self.target[self.slide_num], False)
        label = torch.LongTensor(label)

        locs = [
            self.grid_lists[self.slide_num][loc]
            for loc in self.slide_grids[idx]
        ]
        tiles, time_list, _ = _get_tiles(
            slide=self.current_slide,
            locations=locs,
            tile_size_level_0=self.level_0_tile_size,
            adjusted_tile_sz=self.adjusted_tile_size,
            output_tile_sz=self.tile_size,
            best_slide_level=self.best_slide_level,
            random_shift=False)

        if self.tiles_to_go <= self.tiles_per_iter:
            self.tiles_to_go = None
        else:
            self.tiles_to_go -= self.tiles_per_iter

        X = torch.zeros([len(tiles), 3, self.tile_size, self.tile_size])

        start_aug = time.time()
        for i in range(len(tiles)):
            X[i] = self.transform(tiles[i])

        aug_time = time.time() - start_aug
        total_time = time.time() - start_getitem
        if self.print_time:
            time_list = (time_list[0], time_list[1], aug_time, total_time)
        else:
            time_list = [0]
        if self.tiles_to_go is None:
            last_batch = True
        else:
            last_batch = False

        return {
            'Data': X,
            'Label': label,
            'Time List': time_list,
            'Is Last Batch': last_batch,
            'Initial Num Tiles': self.initial_num_patches,
            'Slide Filename': self.slide_name,
            'Patch Loc': locs,
        }


class WSI_MILdataset(WSI_Master_Dataset):

    def __init__(self,
                 DataSet_location: str = 'TCGA',
                 tile_size: int = 256,
                 bag_size: int = 100,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 print_timing: bool = False,
                 transform_type: str = 'flip',
                 color_param: float = 0.1,
                 desired_slide_magnification: int = 10):
        super(WSI_MILdataset, self).__init__(
            DataSet_location=DataSet_location,
            tile_size=tile_size,
            bag_size=bag_size,
            target_kind=target_kind,
            test_fold=test_fold,
            train=train,
            print_timing=print_timing,
            transform_type=transform_type,
            train_type='MIL',
            color_param=color_param,
            desired_slide_magnification=desired_slide_magnification)

        logging.info(
            'Initiation of WSI({}) {} {} DataSet for {} is Complete. Magnification is X{}, {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}.'
            .format(self.train_type, 'Train' if self.train else 'Test',
                    self.DataSet, self.target_kind, self.desired_magnification,
                    self.real_length, self.tile_size, self.bag_size,
                    'Without' if transform_type == 'none' else 'With',
                    self.test_fold))


class Features_MILdataset(Dataset):

    def __init__(
            self,
            dataset_location: str = r'./TCGA',
            data_location:
        str = r'./TCGA_Features',
            bag_size: int = 100,
            minimum_tiles_in_slide: int = 50,
            is_all_tiles: bool = False,
            # if True than all slide tiles will be used (else only bag_tiles number of tiles)
            fixed_tile_num: int = None,
            is_repeating_tiles: bool = True,
            # if True than the tile pick from each slide/patient is with repeating elements
            target: str = 'ER',
            is_train: bool = False,
            data_limit: int = None,
            # if 'None' than there will be no data limit. If a number is specified than it'll be the data limit
            print_timing: bool = False,
            sample_tiles: bool = True,
            slide_magnification: int = 10
        # slide_repetitions: int = 1
    ):
        self.magnification = slide_magnification
        self.tile_idx = None
        self.sample_tiles = sample_tiles
        self.is_all_tiles, self.is_repeating_tiles = is_all_tiles, is_repeating_tiles
        self.bag_size = bag_size
        self.train_type = 'Features'
        target = target.replace(
            '_Features', '', 1
        ) if len(target.split('_')) in [2, 3] and target.split(
            '_'
        )[-1] == 'Features' else target  # target.split('_')[0] if len(target.split('_')) == 2 and target.split('_')[1] == 'Features' else target

        self.slide_names = []
        self.labels = []
        self.targets = []
        self.features = []
        self.num_tiles = []
        self.scores = []
        self.tile_scores = []
        self.tile_location = []
        self.fixed_tile_num = fixed_tile_num  # This instance variable indicates what is the number of fixed tiles to be used. if "None" than all tiles will be used. This feature is used to check the necessity in using more than 500 feature tiles for training
        total_slides, bad_num_of_good_tiles = 0, 0
        slides_with_not_enough_tiles, slides_with_bad_segmentation = 0, 0


        data_files = glob(os.path.join(data_location, '*.data'))

        print('Loading data from files in location: {}'.format(data_location))

        

        grid_DF = pd.read_excel(dataset_location + fr'/Grids_{self.magnification}/Grid_data.xlsx')
        slide_data_DF = pd.read_excel(dataset_location + r'/slides_data.xlsx')

        grid_DF.set_index('file', inplace=True)
        slide_data_DF.set_index('file', inplace=True)
        

        
        for file_idx, file in enumerate(tqdm(data_files)):
            with open(file, 'rb') as filehandle:
                inference_data = pickle.load(filehandle)
            if len(inference_data) == 7:
                labels, targets, scores, patch_scores, slide_names, features, tile_location = inference_data
            else:
                continue


            try:
                num_slides, max_tile_num = features.shape[0], features.shape[2]
            except UnboundLocalError:
                print('File Index: {}, File Name: {}, File content length: {}'.
                      format(file_idx, file, len(inference_data)))
                print(features.shape)
            for slide_num in range(num_slides):
                total_slides += 1
                feature_1 = features[slide_num, :, :, 0]
                nan_indices = np.argwhere(np.isnan(feature_1)).tolist()
                first_nan_index = nan_indices[0][1] if bool(
                    nan_indices) else max_tile_num
                tiles_in_slide = first_nan_index  # check if there are any nan values in feature_1


                column_title = f'Legitimate tiles - 256 compatible @ X{self.magnification}'

                try:
                    tiles_in_slide_from_grid_data = int(
                        grid_DF.loc[slide_names[slide_num], column_title])
                except TypeError:
                    raise Exception('Debug')
                except KeyError:
                    print(
                        f'warning: slide {slide_names[slide_num]} not found in grid_data. skipping'
                    )
                    total_slides -= 1
                    continue
                except:
                    print(f'{slide_names[slide_num]}')
                    raise

                # Checking that the number of tiles in Grid_data.xlsx is equall to the one found in the actual data
                if tiles_in_slide_from_grid_data < tiles_in_slide:
                    bad_num_of_good_tiles += 1
                    tiles_in_slide = tiles_in_slide_from_grid_data

                # Limit the number of feature tiles according to argument "data_limit
                if data_limit is not None and is_train and tiles_in_slide > data_limit:
                    tiles_in_slide = data_limit

                # Checking that the slide has a minimum number of tiles to be useable
                if tiles_in_slide < minimum_tiles_in_slide:
                    slides_with_not_enough_tiles += 1
                    continue

                # Now we decide how many feature tiles will be taken w.r.t self.fixed_tile_num parameter
                if self.fixed_tile_num is not None:
                    tiles_in_slide = tiles_in_slide if tiles_in_slide <= self.fixed_tile_num else self.fixed_tile_num

                self.num_tiles.append(tiles_in_slide)
                self.features.append(features[
                    slide_num, :, :tiles_in_slide, :].squeeze(0).astype(
                        np.float32))
                if len(patch_scores.shape) == 2:
                    self.tile_scores.append(
                        patch_scores[slide_num, :tiles_in_slide])
                elif len(patch_scores.shape) == 1:
                    self.tile_scores.append(patch_scores[:tiles_in_slide])

                self.slide_names.append(slide_names[slide_num])
                self.labels.append(int(labels[slide_num]))
                self.targets.append(int(targets[slide_num]))
                self.scores.append(scores[slide_num])
                # self.tile_location.append(tile_location[slide_num, :tiles_in_slide, :])
                self.tile_location.append(
                    tile_location[slide_num, :tiles_in_slide])


        print(
            'There are {}/{} slides whose tile amount in Grid_data.xlsx is lower than amount found in the feature files'
            .format(bad_num_of_good_tiles, total_slides))
        print('There are {}/{} slides with \"bad segmentation\" '.format(
            slides_with_bad_segmentation, total_slides))
        print('There are {}/{} slides with less than {} tiles '.format(
            slides_with_not_enough_tiles, total_slides,
            minimum_tiles_in_slide))

        
        print('Initialized Dataset with {} feature slides'.format(
            self.__len__()))

    def __len__(self):
        return len(self.slide_names)


    def __getitem__(self, item):
        if self.sample_tiles:  # else we assume tile_idx is given from outside
            num_tiles = self.num_tiles[item]
            self.tile_idx = list(
                range(num_tiles)) if self.is_all_tiles else choices(
                    range(num_tiles), k=self.bag_size)
        tile_idx = self.tile_idx
        return {
            'labels':
            self.labels[item],
            'targets':
            self.targets[item],
            'scores':
            self.scores[item],
            'tile scores':
            self.tile_scores[item][tile_idx],
            'slide name':
            self.slide_names[item],
            'features':
            self.features[item][tile_idx],
            'num tiles':
            self.num_tiles[item],
            'tile locations':
            self.tile_location[item][tile_idx] if hasattr(
                self, 'tile_location') else None
        }
