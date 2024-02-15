import utils
import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import pandas as pd
from sklearn.utils import resample
import psutil
from models import preact_resnet
from datetime import datetime
import re
import logging
import wandb
import sys
from torchvision import transforms
sys.path.insert(0, os.path.abspath('../'))

DEFAULT_BATCH_SIZE = 18
parser = argparse.ArgumentParser(description='WSI_REG Training of PathNet Project')
parser.add_argument('-tf', '--test_fold', default=1, type=int, help='fold to be as VALIDATION FOLD, if -1 there is no validation.')
parser.add_argument('-e', '--epochs', default=1001, type=int, help='Epochs to run')
parser.add_argument('-ds', '--dataset', type=str, default='./TCGA', help='Dataset location')
parser.add_argument('-time', dest='time', action='store_true', help='save train timing data ?')
parser.add_argument('-tar', '--target', default='Survival_Time', type=str, help='label: Her2/ER/PR')
parser.add_argument('--n_patches_test', default=1, type=int, help='# of patches at test time')
parser.add_argument('--n_patches_train', default=10, type=int, help='# of patches at train time')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-5, type=float, help='L2 penalty')
parser.add_argument('-balsam', '--balanced_sampling', dest='balanced_sampling', action='store_true', help='balanced_sampling')
parser.add_argument('--transform_type', default='pcbnfrsc', type=str)
parser.add_argument('--batch_size', default=DEFAULT_BATCH_SIZE, type=int, help='size of batch')
parser.add_argument('--model', default='preact_resnet.PreActResNet50()', type=str, help='net to use')
parser.add_argument('--bootstrap', action='store_true', help='use bootstrap to estimate test AUC error')
parser.add_argument('--mag', type=int, default=10, help='desired magnification of patches')
parser.add_argument('--eval_rate', type=int, default=5, help='Evaluate validation set every # epochs')
parser.add_argument('--c_param', default=0.1, type=float, help='color jitter parameter')
parser.add_argument('--RAM_saver', action='store_true', help='use only a quarter of the slides + reshuffle every 100 epochs')
parser.add_argument('--wnb', type=str, default='', help='wandb project name for model diagnosis. disabled if empty string')

args = parser.parse_args()
config = vars(args)
EPS = 1e-7


def train(model: nn.Module, dloader_train: DataLoader, dloader_test: DataLoader, DEVICE, optimizer, criterion, print_timing: bool=False, wnb: bool=False):
    """
    This function trains the model
    :return:
    """
    
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    if not os.path.isdir(os.path.join(args.output_dir, 'Model_CheckPoints')):
        os.mkdir(os.path.join(args.output_dir, 'Model_CheckPoints'))
    writer_folder = os.path.join(args.output_dir, 'writer')
    all_writer = SummaryWriter(os.path.join(writer_folder, 'all'))
    test_auc_list = []

    if from_epoch == 0:
        all_writer.add_text('Experiment No.', str(experiment))
        all_writer.add_text('Train type', 'Regular')
        all_writer.add_text('Model type', str(type(model)))
        all_writer.add_text('Train Folds', str(dloader_train.dataset.folds).strip('[]'))
        all_writer.add_text('Test Folds', str(dloader_test.dataset.folds).strip('[]'))
        all_writer.add_text('Transformations', str(dloader_train.dataset.transform))
        all_writer.add_text('Receptor Type', str(dloader_train.dataset.target_kind))

    if print_timing:
        time_writer = SummaryWriter(os.path.join(writer_folder, 'time'))

    print('Start Training...')
    previous_epoch_loss = 1e5
    
    for e in range(from_epoch, epoch):
        time_epoch_start = time.time()
        scores_train = np.zeros(0)
        true_targets_train = np.zeros(0)
        correct_pos, correct_neg = 0, 0
        total_pos_train, total_neg_train = 0, 0
        correct_labeling = 0
        train_loss, total = 0, 0

        slide_names = []
        logging.info('Epoch {}:'.format(e))

        process = psutil.Process(os.getpid())
        logging.info('RAM usage: {} GB, time: {}, exp: {}'.format(np.round(process.memory_info().rss/1e9),
                                                                  datetime.now(),
                                                                  str(experiment)))

        model.train()
        model.to(DEVICE)
        
        i=0
        
        for batch_idx, minibatch in enumerate(tqdm(dloader_train)):
            data = minibatch['Data']
            target = minibatch['Target']
            time_list =  minibatch['Time List']
            f_names = minibatch['File Names']

            train_start = time.time()
            data, target = data.to(DEVICE), target.to(DEVICE).squeeze(1)

            optimizer.zero_grad()
            if print_timing:
                time_fwd_start = time.time()

            outputs, _ = model(data)

            if print_timing:
                time_fwd = time.time() - time_fwd_start

            loss = criterion(outputs, target)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            scores_train = np.concatenate((scores_train, outputs[:, 1].cpu().detach().numpy()))
            
            true_targets_train = np.concatenate((true_targets_train, target.cpu().detach().numpy()))
            total_pos_train += target.eq(1).sum().item()
            total_neg_train += target.eq(0).sum().item()
            correct_labeling += predicted.eq(target).sum().item()
            correct_pos += predicted[target.eq(1)].eq(1).sum().item()
            correct_neg += predicted[target.eq(0)].eq(0).sum().item()
            total += target.size(0)

            if loss != 0:
                if print_timing:
                    time_backprop_start = time.time()

                loss.backward()

                optimizer.step()

                train_loss += loss.item()

                if print_timing:
                    time_backprop = time.time() - time_backprop_start

            slide_names_batch = [os.path.basename(f_name) for f_name in f_names]
            slide_names.extend(slide_names_batch)

            if DEVICE.type == 'cuda' and print_timing:
                res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                all_writer.add_scalar('GPU/gpu', res.gpu, batch_idx + e * len(dloader_train))
                all_writer.add_scalar('GPU/gpu-mem', res.memory, batch_idx + e * len(dloader_train))
            train_time = time.time() - train_start
            if print_timing:
                time_stamp = batch_idx + e * len(dloader_train)
                time_writer.add_scalar('Time/Train (iter) [Sec]', train_time, time_stamp)
                time_writer.add_scalar('Time/Forward Pass [Sec]', time_fwd, time_stamp)
                time_writer.add_scalar('Time/Back Propagation [Sec]', time_backprop, time_stamp)
                time_list = torch.stack(time_list, 1)
                if len(time_list[0]) == 4:
                    time_writer.add_scalar('Time/Open WSI [Sec]', time_list[:, 0].mean().item(), time_stamp)
                    time_writer.add_scalar('Time/Avg to Extract Tile [Sec]', time_list[:, 1].mean().item(), time_stamp)
                    time_writer.add_scalar('Time/Augmentation [Sec]', time_list[:, 2].mean().item(), time_stamp)
                    time_writer.add_scalar('Time/Total To Collect Data [Sec]', time_list[:, 3].mean().item(), time_stamp)
                else:
                    time_writer.add_scalar('Time/Avg to Extract Tile [Sec]', time_list[:, 0].mean().item(), time_stamp)
                    time_writer.add_scalar('Time/Augmentation [Sec]', time_list[:, 1].mean().item(), time_stamp)
                    time_writer.add_scalar('Time/Total To Collect Data [Sec]', time_list[:, 2].mean().item(), time_stamp)

        time_epoch = (time.time() - time_epoch_start)  # sec
        if print_timing:
            time_writer.add_scalar('Time/Full Epoch [min]', time_epoch / 60, e)

        train_loss /= len(dloader_train)  # normalize loss
        train_acc = 100 * correct_labeling / total
        balanced_acc_train = 100. * ((correct_pos + EPS) / (total_pos_train + EPS) + (correct_neg + EPS) / (total_neg_train + EPS)) / 2


        roc_auc_train = np.float64(np.nan)
        if len(np.unique(true_targets_train[true_targets_train >= 0])) > 1:  # more than one label
            fpr_train, tpr_train, _ = roc_curve(true_targets_train, scores_train)
            roc_auc_train = auc(fpr_train, tpr_train)
        all_writer.add_scalar('Train/Balanced Accuracy', balanced_acc_train, e)
        all_writer.add_scalar('Train/Roc-Auc', roc_auc_train, e)
        all_writer.add_scalar('Train/Accuracy', train_acc, e)
        all_writer.add_scalar('Train/Loss Per Epoch', train_loss, e)
        

        logging.info('Finished Epoch: {}, Loss: {:.4f}, Loss Delta: {:.3f}, Train AUC per patch: {:.2f} , Time: {:.0f} m {:.0f} s'
              .format(e,
                      train_loss,
                      previous_epoch_loss - train_loss,
                      roc_auc_train if roc_auc_train.size == 1 else roc_auc_train[0],
                      time_epoch // 60,
                      time_epoch % 60))
        previous_epoch_loss = train_loss

        # Save model to file:
        try:
            model_state_dict = model.module.state_dict()
        except AttributeError:
            model_state_dict = model.state_dict()
        torch.save({'epoch': e,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'tile_size': TILE_SIZE,
                    'tiles_per_bag': 1},
                   os.path.join(args.output_dir, 'Model_CheckPoints', 'model_data_Last_Epoch.pt'))

        if e % args.eval_rate == 0:
            # Update 'Last Epoch' at run_data.xlsx file:
            utils.run_data(experiment=experiment, epoch=e)
            
            if len(dloader_test) != 0:
                acc_test, bacc_test, roc_auc_test = check_accuracy(model, dloader_test, all_writer, DEVICE, e)

                patch_df = pd.DataFrame({'slide': slide_names, 'scores': scores_train, 'labels': true_targets_train})
                slide_mean_score_df = patch_df.groupby('slide').mean()
                roc_auc_slide = np.nan
                if not all(slide_mean_score_df['labels'] == slide_mean_score_df['labels'][0]):  #more than one label
                    roc_auc_slide = roc_auc_score(slide_mean_score_df['labels'], slide_mean_score_df['scores'])
                all_writer.add_scalar('Train/slide AUC', roc_auc_slide, e)

                test_auc_list.append(roc_auc_test)
                if len(test_auc_list) == 5:
                    test_auc_mean = np.mean(test_auc_list)
                    test_auc_list.pop(0)
                    utils.run_data(experiment=experiment, test_mean_auc=test_auc_mean)
            else:
                acc_test, bacc_test = None, None

            # Save model to file:
            try:
                model_state_dict = model.module.state_dict()
            except AttributeError:
                model_state_dict = model.state_dict()
            torch.save({'epoch': e,
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        'acc_test': acc_test,
                        'bacc_test': bacc_test,
                        'tile_size': TILE_SIZE,
                        'tiles_per_bag': 1},
                       os.path.join(args.output_dir, 'Model_CheckPoints', 'model_data_Epoch_' + str(e) + '.pt'))
            logging.info('saved checkpoint to {}'.format(args.output_dir))
    all_writer.close()
    if print_timing:
        time_writer.close()


def check_accuracy(model: nn.Module, data_loader: DataLoader, all_writer, DEVICE, epoch: int):
    scores_test = np.zeros(0)
    true_pos_test, true_neg_test = 0, 0
    total_pos_test, total_neg_test = 0, 0
    true_labels_test = np.zeros(0)
    correct_labeling_test = 0
    total_test = 0
    slide_names = []

    model.eval()

    with torch.no_grad():
        for batch_idx, minibatch in enumerate(data_loader):
            data = minibatch['Data']
            targets = minibatch['Target']
            f_names = minibatch['File Names']
            slide_names_batch = [os.path.basename(f_name) for f_name in f_names]
            slide_names.extend(slide_names_batch)

            data, targets = data.to(device=DEVICE), targets.to(device=DEVICE).squeeze(1)
            model.to(DEVICE)

            outputs, _ = model(data)

            outputs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            scores_test = np.concatenate((scores_test, outputs[:, 1].cpu().detach().numpy()))

            true_labels_test = np.concatenate((true_labels_test, targets.cpu().detach().numpy()))
            correct_labeling_test += predicted.eq(targets).sum().item()
            total_pos_test += targets.eq(1).sum().item()
            total_neg_test += targets.eq(0).sum().item()
            true_pos_test += predicted[targets.eq(1)].eq(1).sum().item()
            true_neg_test += predicted[targets.eq(0)].eq(0).sum().item()
            total_test += targets.size(0)

        #perform slide inference
        patch_df = pd.DataFrame({'slide': slide_names, 'scores': scores_test, 'labels': true_labels_test})
        slide_mean_score_df = patch_df.groupby('slide').mean()
        roc_auc_slide = np.nan
        if not all(slide_mean_score_df['labels'] == slide_mean_score_df['labels'][0]): #more than one label
            roc_auc_slide = roc_auc_score(slide_mean_score_df['labels'], slide_mean_score_df['scores'])

        if args.bootstrap:
            # load dataset
            # configure bootstrap
            n_iterations = 100

            # run bootstrap
            roc_auc_array = np.empty(n_iterations)
            slide_roc_auc_array = np.empty(n_iterations)
            roc_auc_array[:], slide_roc_auc_array[:] = np.nan, np.nan
            acc_array, bacc_array = np.empty(n_iterations), np.empty(n_iterations)
            acc_array[:], bacc_array[:] = np.nan, np.nan

            all_preds = np.array([int(score > 0.5) for score in scores_test])

            for ii in range(n_iterations):
                slide_names = np.array(slide_names)
                slide_choice = resample(np.unique(np.array(slide_names)))
                slide_resampled = np.concatenate([slide_names[slide_names == slide] for slide in slide_choice])
                scores_resampled = np.concatenate([scores_test[slide_names == slide] for slide in slide_choice])
                labels_resampled = np.concatenate([true_labels_test[slide_names == slide] for slide in slide_choice])
                preds_resampled = np.concatenate([all_preds[slide_names == slide] for slide in slide_choice])
                patch_df = pd.DataFrame({'slide': slide_resampled, 'scores': scores_resampled, 'labels': labels_resampled})

                num_correct_i = np.sum(preds_resampled == labels_resampled)
                true_pos_i = np.sum(labels_resampled + preds_resampled == 2)
                total_pos_i = np.sum(labels_resampled == 1)
                true_neg_i = np.sum(labels_resampled + preds_resampled == 0)
                total_neg_i = np.sum(labels_resampled == 0)
                tot = total_pos_i + total_neg_i
                acc_array[ii] = 100 * float(num_correct_i) / tot
                bacc_array[ii] = 100. * ((true_pos_i + EPS) / (total_pos_i + EPS) + (true_neg_i + EPS) / (total_neg_i + EPS)) / 2
                fpr, tpr, _ = roc_curve(labels_resampled, scores_resampled)
                if not all(labels_resampled == labels_resampled[0]): #more than one label
                    roc_auc_array[ii] = roc_auc_score(labels_resampled, scores_resampled)

                slide_mean_score_df = patch_df.groupby('slide').mean()
                if not all(slide_mean_score_df['labels'] == slide_mean_score_df['labels'][0]):  # more than one label
                    slide_roc_auc_array[ii] = roc_auc_score(slide_mean_score_df['labels'], slide_mean_score_df['scores'])

            roc_auc_std = np.nanstd(roc_auc_array)
            roc_auc_slide_std = np.nanstd(slide_roc_auc_array)
            acc_err = np.nanstd(acc_array)
            bacc_err = np.nanstd(bacc_array)

            all_writer.add_scalar('Test_errors/Accuracy error', acc_err, epoch)
            all_writer.add_scalar('Test_errors/Balanced Accuracy error', bacc_err, epoch)
            all_writer.add_scalar('Test_errors/Roc-Auc error', roc_auc_std, epoch)
            if args.n_patches_test > 1:
                all_writer.add_scalar('Test_errors/slide AUC error', roc_auc_slide_std, epoch)

        acc = 100 * correct_labeling_test / total_test
        bacc = 100. * ((true_pos_test + EPS) / (total_pos_test + EPS) + (true_neg_test + EPS) / (total_neg_test + EPS)) / 2
        

        roc_auc = np.float64(np.nan)
        if not all(true_labels_test == true_labels_test[0]):  # more than one label
            fpr, tpr, _ = roc_curve(true_labels_test, scores_test)
            roc_auc = auc(fpr, tpr)

        all_writer.add_scalar('Test/Accuracy', acc, epoch)
        all_writer.add_scalar('Test/Balanced Accuracy', bacc, epoch)
        all_writer.add_scalar('Test/Roc-Auc', roc_auc, epoch)
        if args.n_patches_test > 1:
            all_writer.add_scalar('Test/slide AUC', roc_auc_slide, epoch)

        if args.n_patches_test > 1:
            #print('Slide AUC of {:.2f} over Test set'.format(roc_auc_slide))
            logging.info('Slide AUC of {:.2f} over Test set'.format(roc_auc_slide))
        else:
            #print('Tile AUC of {:.2f} over Test set'.format(roc_auc))
            logging.info('Tile AUC of {:.2f} over Test set'.format(roc_auc))
    model.train()
    try:
        return acc, bacc, roc_auc
    except:
        return acc, bacc, None

########################################################################################################
########################################################################################################


if __name__ == '__main__':
        
    # Tile size definition:
    TILE_SIZE = 256
        
    # Saving/Loading run meta data to/from file:
    run_data_results = utils.run_data(test_fold=args.test_fold,
                                                 transform_type=args.transform_type,
                                                 tile_size=TILE_SIZE,
                                                 tiles_per_bag=1,
                                                 DataSet_name=args.dataset,
                                                 Receptor=args.target,
                                                 num_bags=args.batch_size)

    args.output_dir, experiment = run_data_results['Location'], run_data_results['Experiment']

    utils.start_log(args, to_file=True)

    # Device definition:
    DEVICE = utils.device_gpu_cpu()

    # Get number of available CPUs and compute number of workers:
    cpu_available = utils.get_cpu()
    num_workers = cpu_available

    logging.info('num CPUs = {}'.format(cpu_available))
    logging.info('num workers = {}'.format(num_workers))
    
    if args.wnb:
        print("profiling training with wnb")
    
    wnb_mode = "online" if args.wnb else "disabled"
    
    with wandb.init(project=args.wnb, config=config, mode=wnb_mode, entity="gipmed"):
        # Get data:
        train_dset = datasets.WSI_REGdataset(DataSet_location=args.dataset,
                                             tile_size=TILE_SIZE,
                                             target_kind=args.target,
                                             test_fold=args.test_fold,
                                             train=True,
                                             print_timing=args.time,
                                             transform_type=args.transform_type,
                                             n_tiles=args.n_patches_train,
                                             color_param=args.c_param,
                                             desired_slide_magnification=args.mag,
                                             RAM_saver=args.RAM_saver,
                                             )
        test_dset = datasets.WSI_REGdataset(DataSet_location=args.dataset,
                                            tile_size=TILE_SIZE,
                                            target_kind=args.target,
                                            test_fold=args.test_fold,
                                            train=False,
                                            print_timing=False,
                                            transform_type='none',
                                            n_tiles=args.n_patches_test,
                                            desired_slide_magnification=args.mag,
                                            RAM_saver=args.RAM_saver,
                                            )
        sampler = None
        do_shuffle = True
        if args.balanced_sampling:
            labels = pd.DataFrame(train_dset.target * train_dset.factor)
            n_pos = np.sum(labels == 'Positive').item()
            n_neg = np.sum(labels == 'Negative').item()
            weights = pd.DataFrame(np.zeros(len(train_dset)))
            weights[np.array(labels == 'Positive')] = 1 / n_pos
            weights[np.array(labels == 'Negative')] = 1 / n_neg
            do_shuffle = False  # the sampler shuffles
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights.squeeze(), num_samples=len(train_dset))

        train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=do_shuffle,
                                  num_workers=num_workers, pin_memory=True, sampler=sampler)
        test_loader  = DataLoader(test_dset, batch_size=args.batch_size*2, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)

        # Save transformation data to 'run_data.xlsx'
        transformation_string = ', '.join([str(train_dset.transform.transforms[i]) for i in range(len(train_dset.transform.transforms))])
        utils.run_data(experiment=experiment, transformation_string=transformation_string)

        # Load model
        model = eval(args.model)

        utils.run_data(experiment=experiment, model=model.model_name)
        utils.run_data(experiment=experiment, DataSet_size=(train_dset.real_length, test_dset.real_length))
        utils.run_data(experiment=experiment, DataSet_Slide_magnification=train_dset.desired_magnification)

        # Saving code files, args and main file name (this file) to Code directory within the run files.
        utils.save_code_files(args, train_dset)

        epoch = args.epochs
        from_epoch = 0

        

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


        if DEVICE.type == 'cuda':
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

            if args.time:
                import nvidia_smi
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

        criterion = nn.CrossEntropyLoss()

        if args.RAM_saver:
            shuffle_freq = 100  # reshuffle dataset every 200 epochs
            shuffle_epoch_list = np.arange(np.ceil((from_epoch+EPS) / shuffle_freq) * shuffle_freq, epoch, shuffle_freq).astype(int)
            shuffle_epoch_list = np.append(shuffle_epoch_list, epoch)

            epoch = shuffle_epoch_list[0]
            train(model, train_loader, test_loader, DEVICE=DEVICE, optimizer=optimizer, criterion=criterion, print_timing=args.time, wnb=args.wnb)

            for from_epoch, epoch in zip(shuffle_epoch_list[:-1], shuffle_epoch_list[1:]):
                print('Reshuffling dataset:')
                # shuffle train and test set to get new slides
                # Get data:
                train_dset = datasets.WSI_REGdataset(DataSet_location=args.dataset,
                                                     tile_size=TILE_SIZE,
                                                     target_kind=args.target,
                                                     test_fold=args.test_fold,
                                                     train=True,
                                                     print_timing=args.time,
                                                     transform_type=args.transform_type,
                                                     n_tiles=args.n_patches_train,
                                                     color_param=args.c_param,
                                                     desired_slide_magnification=args.mag,
                                                     RAM_saver=args.RAM_saver
                                                     )
                test_dset = datasets.WSI_REGdataset(DataSet_location=args.dataset,
                                                    tile_size=TILE_SIZE,
                                                    target_kind=args.target,
                                                    test_fold=args.test_fold,
                                                    train=False,
                                                    print_timing=False,
                                                    transform_type='none',
                                                    n_tiles=args.n_patches_test,
                                                    desired_slide_magnification=args.mag,
                                                    RAM_saver=args.RAM_saver
                                                    )
                sampler = None
                do_shuffle = True
                if args.balanced_sampling:
                    labels = pd.DataFrame(train_dset.target * train_dset.factor)
                    n_pos = np.sum(labels == 'Positive').item()
                    n_neg = np.sum(labels == 'Negative').item()
                    weights = pd.DataFrame(np.zeros(len(train_dset)))
                    weights[np.array(labels == 'Positive')] = 1 / n_pos
                    weights[np.array(labels == 'Negative')] = 1 / n_neg
                    do_shuffle = False  # the sampler shuffles
                    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights.squeeze(),
                                                                             num_samples=len(train_dset))

                train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=do_shuffle,
                                          num_workers=num_workers, pin_memory=True, sampler=sampler)
                test_loader = DataLoader(test_dset, batch_size=args.batch_size * 2, shuffle=False,
                                         num_workers=num_workers, pin_memory=True)

                print('resuming training with new dataset')
                train(model, train_loader, test_loader, DEVICE=DEVICE, optimizer=optimizer, criterion=criterion, print_timing=args.time, wnb=args.wnb)
        else:
            train(model, train_loader, test_loader, DEVICE=DEVICE, optimizer=optimizer, criterion=criterion, print_timing=args.time, wnb=args.wnb)
