import utils as utils
from torch.utils.data import DataLoader
import torch
import datasets
import numpy as np
from sklearn.metrics import roc_curve
import os
import sys, platform
import argparse
from tqdm import tqdm
import pickle
from collections import OrderedDict
from models import preact_resnet
import logging
import warnings

parser = argparse.ArgumentParser(description='WSI_REG Slide inference')
parser.add_argument('-ex', '--experiment', nargs='+', type=int, default=1, help='Use models from this experiment')
parser.add_argument('-fe', '--from_epoch', nargs='+', type=int, default=1000, help='Use this epoch models for inference') # use -1 for final epoch
parser.add_argument('-nt', '--num_tiles', type=int, default=10, help='Number of tiles to use')
parser.add_argument('-ds', '--dataset', type=str, default='ABCTB', help='DataSet location')
parser.add_argument('-f', '--folds', type=list, nargs="+", default=1, help='folds to infer')
parser.add_argument('--mag', type=int, default=10, help='desired magnification of patches')
parser.add_argument('--save_features', action='store_true', help='save features')
parser.add_argument('--resume', type=int, default=0, help='resume a failed feature extraction')
parser.add_argument('-sd', '--subdir', type=str, default='', help='output sub-dir')
parser.add_argument('-se', '--seed', type=int, help='use for deterministic patch sampling')
parser.add_argument('-tar', '--target', type=str, help='label: Her2/ER/PR/EGFR/PDL1')
args = parser.parse_args()

args.folds = list(map(int, args.folds[0]))

utils.start_log(args)

# If args.experiment contains 1 number than all epochs are from the same experiments,
# BUT if it is bigger than 1 than all the length of args.experiment should be equal to args.from_epoch
if len(args.experiment) > 1:
    if len(args.experiment) != len(args.from_epoch):
        raise Exception("number of from_epoch(-fe) should be equal to number of experiment(-ex)")
    else:
        different_experiments = True
        Output_Dirs = []
else:
    different_experiments = False

DEVICE = utils.device_gpu_cpu()

logging.info('Loading pre-saved models:')
models = []

# else, if there only one epoch, take it. otherwise take epoch 1000
if args.save_features:
    if len(args.from_epoch) > 1:
        try:
            feature_epoch_ind = (args.from_epoch).index(1000)
        except ValueError:
            feature_epoch_ind = (args.from_epoch).index(2000)  # If 1000 is not on the list, take epoch 2000
    elif len(args.from_epoch) == 1:
        feature_epoch_ind = 0

for counter in range(len(args.from_epoch)):
    epoch = args.from_epoch[counter]
    experiment = args.experiment[counter] if different_experiments else args.experiment[0]
    logging.info('  Exp. {} and Epoch {}'.format(experiment, epoch))
    # Basic meta data will be taken from the first model (ONLY if all inferences are done from the same experiment)
    if counter == 0:
        run_data_output = utils.run_data(experiment=experiment)
        output_dir, TILE_SIZE, model_name =\
            run_data_output['Location'], run_data_output['Tile Size'],\
            run_data_output['Model Name']
        if args.target is None:
             args.target = run_data_output['Receptor']
        if different_experiments:
            Output_Dirs.append(output_dir)
    elif counter > 0 and different_experiments:
        run_data_output = utils.run_data(experiment=experiment)
        output_dir, target, model_name =\
            run_data_output['Location'], run_data_output['Receptor'],\
            run_data_output['Model Name']
        Output_Dirs.append(output_dir)




        # Verifying that the target receptor is not changed:
        if counter > 1 and args.target != target:
            raise Exception("Target Receptor is changed between models - DataSet cannot support this action")



    # loading basic model type
    model = eval(model_name)
    # loading model parameters from the specific epoch
    if epoch==-1:
        model_to_load = os.path.join(output_dir, 'Model_CheckPoints',
                                                'model_data_Last_Epoch.pt')
    else:
        model_to_load = os.path.join(output_dir, 'Model_CheckPoints',
                                                'model_data_Epoch_' + str(epoch) + '.pt')
    model_data_loaded = torch.load(model_to_load, map_location='cpu')
    # Making sure that the size of the linear layer of the loaded model, fits the basic model.
    model.linear = torch.nn.Linear(in_features=model_data_loaded['model_state_dict']['linear.weight'].size(1),
                                   out_features=model_data_loaded['model_state_dict']['linear.weight'].size(0))
    model.load_state_dict(model_data_loaded['model_state_dict'])
    model.eval()
    models.append(model)


N_classes = models[0].linear.out_features  # for resnets and such
TILE_SIZE = 256
tiles_per_iter = 100




if args.save_features:
    logging.info('features will be taken from model {}'.format(str(args.from_epoch[feature_epoch_ind])))

slide_num = args.resume

inf_dset = datasets.Infer_Dataset(DataSet_location=args.dataset,
                                  tile_size=TILE_SIZE,
                                  tiles_per_iter=tiles_per_iter,
                                  target_kind=args.target,
                                  folds=args.folds,
                                  num_tiles=args.num_tiles,
                                  desired_slide_magnification=args.mag,
                                  resume_slide=slide_num,
                                  chosen_seed=args.seed)

inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

new_slide = True

NUM_MODELS = len(models)
NUM_SLIDES = len(inf_dset.image_file_names)
NUM_SLIDES_SAVE = 50
logging.info('NUM_SLIDES: {}'.format(NUM_SLIDES))


all_targets = []
total_pos, total_neg = 0, 0
all_labels = np.zeros((NUM_SLIDES, NUM_MODELS))
all_scores = np.zeros((NUM_SLIDES, NUM_MODELS))
patch_scores = np.empty((NUM_SLIDES, NUM_MODELS, args.num_tiles))

correct_pos = np.zeros(NUM_MODELS)
correct_neg = np.zeros(NUM_MODELS)

patch_locs_all = np.empty((NUM_SLIDES, args.num_tiles, 2))
if args.save_features:
    features_all = np.empty((NUM_SLIDES_SAVE, 1, args.num_tiles, 512))
    features_all[:] = np.nan
all_slide_names = np.zeros(NUM_SLIDES, dtype=object)
patch_scores[:] = np.nan
patch_locs_all[:] = np.nan

if args.resume:
    # load the inference state
    resume_file_name = os.path.join(output_dir, 'Inference', args.subdir,
                                    'Exp_' + str(args.experiment[0])
                                    + '-Folds_' + str(args.folds) + '_' + str(
                                        args.target) + '-Tiles_' + str(
                                        args.num_tiles) + '_resume_slide_num_' + str(slide_num) + '.data')
    with open(resume_file_name, 'rb') as filehandle:
        resume_data = pickle.load(filehandle)
    all_labels, all_targets, all_scores, total_pos, correct_pos, total_neg, \
    correct_neg, patch_scores, all_slide_names, NUM_SLIDES, patch_locs_all = resume_data
else:
    resume_file_name = 0

if not os.path.isdir(os.path.join(output_dir, 'Inference')):
    os.mkdir(os.path.join(output_dir, 'Inference'))

if not os.path.isdir(os.path.join(output_dir, 'Inference', args.subdir)):
    os.mkdir(os.path.join(output_dir, 'Inference', args.subdir))
    
with torch.no_grad():
    for batch_idx, MiniBatch_Dict in enumerate(tqdm(inf_loader)):

        # Unpacking the data:
        data = MiniBatch_Dict['Data']
        target = MiniBatch_Dict['Label']
        last_batch = MiniBatch_Dict['Is Last Batch']
        slide_file = MiniBatch_Dict['Slide Filename']
        patch_locs = MiniBatch_Dict['Patch Loc']
          
        
        if new_slide:
            n_tiles = inf_loader.dataset.num_tiles[slide_num - args.resume]

            current_slide_tile_scores = [np.zeros((n_tiles, N_classes)) for ii in range(NUM_MODELS)]
            patch_locs_1_slide = np.zeros((n_tiles, 2))
            if args.save_features:
                feature_arr = [np.zeros((n_tiles, 512))]
            target_current = target
            slide_batch_num = 0
            new_slide = False
            

        data = data.squeeze(0)
        

        data, target = data.to(DEVICE), target.to(DEVICE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            patch_locs_1_slide[slide_batch_num * tiles_per_iter: slide_batch_num * tiles_per_iter + len(data),:] = np.array(patch_locs)
            

        for model_ind, model in enumerate(models):
            model.to(DEVICE)

            scores, features = model(data)

            scores = torch.nn.functional.softmax(scores, dim=1)

            current_slide_tile_scores[model_ind][slide_batch_num * tiles_per_iter: slide_batch_num * tiles_per_iter + len(data), :] = scores.cpu().detach().numpy()

            if args.save_features:
                if model_ind == feature_epoch_ind:
                    feature_arr[0][slide_batch_num * tiles_per_iter: slide_batch_num * tiles_per_iter + len(data), :] = features.cpu().detach().numpy()

        slide_batch_num += 1

        if last_batch:
            new_slide = True

            all_targets.append(target.cpu().numpy()[0][0])
            if target == 1:
                total_pos += 1
            else:
                total_neg += 1


            patch_locs_all[slide_num, :len(patch_locs_1_slide), :] = patch_locs_1_slide
            if args.save_features:
                features_all[slide_num % NUM_SLIDES_SAVE, 0, :len(feature_arr[0])] = feature_arr[0]
            for model_ind in range(NUM_MODELS):
                predicted = current_slide_tile_scores[model_ind].mean(0).argmax()
                patch_scores[slide_num, model_ind, :n_tiles] = current_slide_tile_scores[model_ind][:, 1]
                all_scores[slide_num, model_ind] = current_slide_tile_scores[model_ind][:, 1].mean()
                if target == 1 and predicted == 1:
                    correct_pos[model_ind] += 1
                elif target == 0 and predicted == 0:
                    correct_neg[model_ind] += 1
                all_labels[slide_num, model_ind] = np.squeeze(predicted)
            all_slide_names[slide_num] = slide_file[0]

            slide_num += 1

            # save features every NUM_SLIDES_SAVE slides
            if slide_num % NUM_SLIDES_SAVE == 0:
                # save the inference state
                prev_resume_file_name = resume_file_name
                resume_file_name = os.path.join(output_dir, 'Inference', args.subdir,
                                                 'Exp_' + str(args.experiment[0])
                                                 + '-Folds_' + str(args.folds) + '_' + str(
                                                     args.target) + '-Tiles_' + str(
                                                     args.num_tiles) + '_resume_slide_num_' + str(slide_num) + '.data')
                resume_data = [all_labels, all_targets, all_scores,
                                  total_pos, correct_pos, total_neg, correct_neg,
                                  patch_scores, all_slide_names, NUM_SLIDES, patch_locs_all]

                with open(resume_file_name, 'wb') as filehandle:
                    pickle.dump(resume_data, filehandle)
                # delete previous resume file
                if os.path.isfile(prev_resume_file_name):
                    os.remove(prev_resume_file_name)

                # save features
                if args.save_features:
                    feature_file_name = os.path.join(output_dir, 'Inference', args.subdir,
                                                     'Model_Epoch_' + str(args.from_epoch[feature_epoch_ind])
                                                     + '-Folds_' + str(args.folds) + '_' + str(
                                                         args.target) + '-Tiles_' + str(args.num_tiles) + '_features_slides_' + str(slide_num) + '.data')
                    inference_data = [all_labels[slide_num-NUM_SLIDES_SAVE:slide_num, feature_epoch_ind],
                                      all_targets[slide_num-NUM_SLIDES_SAVE:slide_num],
                                      all_scores[slide_num-NUM_SLIDES_SAVE:slide_num, feature_epoch_ind],
                                      np.squeeze(patch_scores[slide_num-NUM_SLIDES_SAVE:slide_num, feature_epoch_ind, :]),
                                      all_slide_names[slide_num-NUM_SLIDES_SAVE:slide_num],
                                      features_all,
                                      patch_locs_all[slide_num-NUM_SLIDES_SAVE:slide_num]]
                    with open(feature_file_name, 'wb') as filehandle:
                        pickle.dump(inference_data, filehandle)
                    logging.info('saved output for {} slides'.format(str(slide_num)))
                    features_all = np.empty((NUM_SLIDES_SAVE, 1, args.num_tiles, 512))
                    features_all[:] = np.nan

# save features for last slides
if args.save_features and slide_num % NUM_SLIDES_SAVE != 0:
    feature_file_name = os.path.join(output_dir, 'Inference', args.subdir,
                                     'Model_Epoch_' + str(args.from_epoch[feature_epoch_ind])
                                     + '-Folds_' + str(args.folds) + '_' + str(
                                         args.target) + '-Tiles_' + str(args.num_tiles) + '_features_slides_last.data')
    last_save = slide_num // NUM_SLIDES_SAVE * NUM_SLIDES_SAVE
    inference_data = [all_labels[last_save:slide_num, feature_epoch_ind],
                      all_targets[last_save:slide_num],
                      all_scores[last_save:slide_num, feature_epoch_ind],
                      np.squeeze(patch_scores[last_save:slide_num, feature_epoch_ind, :]),
                      all_slide_names[last_save:slide_num],
                      features_all[:slide_num-last_save],
                      patch_locs_all[last_save:slide_num]]
    with open(feature_file_name, 'wb') as filehandle:
        pickle.dump(inference_data, filehandle)
    logging.info('saved output for {} slides'.format(str(slide_num)))

for model_num in range(NUM_MODELS):
    if different_experiments:
        output_dir = Output_Dirs[model_num]

    # remove targets = -1 from auc calculation
    try:
        scores_arr = all_scores[:, model_num]
        targets_arr = np.array(all_targets)
        scores_arr = scores_arr[targets_arr >= 0]
        targets_arr = targets_arr[targets_arr >= 0]
        fpr, tpr, _ = roc_curve(targets_arr, scores_arr)
    except ValueError:
        fpr, tpr = 0, 0  # if all labels are unknown

    # Save roc_curve to file:
    file_name = os.path.join(output_dir, 'Inference', args.subdir, 'Model_Epoch_' + str(args.from_epoch[model_num])
                             + '-Folds_' + str(args.folds) + '_' + str(args.target) + '-Tiles_' + str(args.num_tiles) + '.data')
    inference_data = [fpr, tpr, all_labels[:, model_num], all_targets, all_scores[:, model_num],
                      total_pos, correct_pos[model_num], total_neg, correct_neg[model_num], NUM_SLIDES,
                      np.squeeze(patch_scores[:, model_num, :]), all_slide_names,
                      np.squeeze(patch_locs_all)]

    with open(file_name, 'wb') as filehandle:
        pickle.dump(inference_data, filehandle)

    experiment = args.experiment[model_num] if different_experiments else args.experiment[0]
    logging.info('For model from Experiment {} and Epoch {}: {} / {} correct classifications'
          .format(experiment,
                  args.from_epoch[model_num],
                  int(len(all_labels[:, model_num]) - np.abs(np.array(all_targets) - np.array(all_labels[:, model_num])).sum()),
                  len(all_labels[:, model_num])))
logging.info('Done!')

# delete last resume file
if os.path.isfile(resume_file_name):
    os.remove(resume_file_name)

