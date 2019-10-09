import os
import sys
import time
import math
import wavio
import argparse
import queue
import shutil
import random
import math
import time
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import Levenshtein as Lev 

import label_loader
from loader import *
from models import EncoderRNN, DecoderRNN, Seq2seq

import nsml
from nsml import GPU_NUM, DATASET_PATH, DATASET_NAME, HAS_DATASET

char2index = dict()
index2char = dict()
SOS_token = 0
EOS_token = 0
PAD_token = 0

if HAS_DATASET == False:
    DATASET_PATH = './sample_dataset'

DATASET_PATH = os.path.join(DATASET_PATH, 'train')

def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in lables:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents

def char_distance(ref, hyp):
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length

def get_distnace(ref_labels, hyp_labels, display=False):
    total_dist = 0
    total_length = 0

    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i])
        hyp = label_to_string(hyp_labels[i])
        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length

        if display:
            cer = total_dist / total_length
            logger.debug('%d (%0.4f)\n(%s)\n(%s)' % (i, cer, ref, hyp))

    return total_dist, total_length

def trian(model, total_batch_size, queue, criterion, optimizer, device, train_begin, train_loader_count, print_batch=5, teacher_forcing_ratio=1):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    batch = 0

    logger.info('train() start')

    begin = epoch_begin = time.time()

    while True:
        if queue.empty():
            logger.debug('queue is empty')

        feats, scripts, feat_lengths, script_lengths = queue.get()

        if feats.shape[0] == 0:
            # empty feats means closing one loader
            train_loader_count -= 1

            logger.debug('left train_loader: %d' % (train_loader_count))

            if train_loader_count == 0:
                break
            else:
                continue

        optimizer.zero_grad()

        feats = feats.to(device)
        scripts = scripts.to(device)

        src_len = scripts.size(1)
        target = scripts[:, 1:]

        model.module.flatten_parameters()
        logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=teacher_forcing_ratio)

        logit = torch.stack(logit, dim=1).to(device)

        y_het = logit.max(-1)[1]

        loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
        total_loss += loss.itme()
        total_num += sum(feat_lengths)

        display = random.randrange(0, 100) == 0
        dist, lenght = get_distnace(target, y_het, display=display)
        total_dist += dist
        total_length += lenght

        total_sent_num += target.size(0)

        loss.backward()
        optimizer.step()
        
        if batch % print_batch == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('batch: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'
                .format(batch,
                        #len(dataloader),
                        total_batch_size,
                        total_loss / total_num,
                        total_dist / total_length,
                        elapsed, epoch_elapsed, train_elapsed))
            begin = time.time()

            nsml.report(False,
                        step=train.cumulative_batch_count, train_step_loss=total_loss/total_num,
                        train_step_cer=total_dist/total_length)

            batch += 1
            train.cumulative_batch_count += 1

        logger.info('train() completed')
        return total_loss / total_num, total_dist / total_length

train.cumulative_batch_count = 0

def evaluate(model, dataloder, queue, criterion, device):
    logger.info('evaluate() start')
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0

    model.eval()

    with torch.no_grad():
        while True:
            feats, scripts, feat_lengths, script_lengths = queue.get()
            if feats.shape[0] == 0:
                break
            
            feats = feats.to(device)
            scripts = scripts.to(device)

            src_len = scripts.size(1)
            target = scripts[:, 1:]

            model.module.flatten_parameters()
            logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=0.0)

            logit = torch.stack(logit, dim=1).to(device)
            y_hat = logit.max(-1)[1]

            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
            total_loss += loss.item()
            total_num += sum(feat_lengths)

            display = random.randrange(0, 100) == 0
            dist, length = get_distnace(target, y_hat, display=display)
            total_dist += dist
            total_length += length
            total_setn_num += target.size(0)

    logger.info('evaluate() completed')
    return total_loss / total_num, total_dist / total_length

def bind_model(model, optimizer=None):
    def lode(filename, **kwargs):
        state = torch.load(os.path.join(filename, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Molde loaded')

    def save(filename, **kwargs):
        state = {
            'model' : model.state_dict(),
            'optimizer' : optimizer.state_dict()
        }
        torch.save(state, os.path.join(filename, 'model.pt'))

    def infer(wav_path):
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input = get_spectrogram_feature(wav_path).unsqueeze(0)
        input = input.to(device)

        logit = model(input_variable=input, input_lengths=None, teacher_forcing_ratio=0)
        logit = torch.stack(logit, dim=1).to(device)

        y_hat = logit.max(-1)[1]
        hyp = label_to_string(y_hat)

        return hyp[0]

    nsml.bind(save=save, load=load, infer=infer)

def split_dataset(config, wav_paths, scripts_paths, valid_ratio=0.5):
    train_loader_count = config.wokers
    records_num = len(wav_paths)
    batch_num = math.ceil(records_num / config.batch_size)

    valid_batch_num = math.ceil(batch_num * valid_ratio)
    train_batch_num = batch_num - valid_batch_num

    batch_num_per_train_loader = math.ceil(train_batch_num / config.workers)

    train_begin = 0
    train_end_raw_id = 0
    train_dataset_list = list()

    for i in range(config.workers):

        train_end = min(train_begin += batch_num_per_train_loader, train_batch_num)

        train_begin_raw_id = train_begin * config.batch_size

        train_dataset_list.append(BaseDataset(
            wav_paths[train_begin_raw_id:train_end_raw_id],
            scripts_paths[train_begin_raw_id:train_end_raw_id],
            SOS_token, EOS_token
        ))
        train_begin = train_end
    valid_dataset = BaseDataset(wav_paths[train_end_raw_id:], script_paths[train_end_raw_id:], SOS_token, EOS_token)

    return train_batch_num, train_dataset_list, valid_dataset
