#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
# import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from code import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU', default=True)
    parser.add_argument('--CUDA_DEVISE', default='1', help='')

    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data', default=True)

    parser.add_argument('--datadir', type=str, default='../../data')
    # parser.add_argument('--dataset', type=str, default='AwA')
    # parser.add_argument('--dataset', type=str, default='ImageNet/ImNet_A')
    parser.add_argument('--dataset', type=str, default='NELL')


    parser.add_argument('-save', '--save_path', type=str)

    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=1024, type=int)
    parser.add_argument('-d', '--hidden_dim', default=100, type=int)
    parser.add_argument('-g', '--gamma', default=12, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true', default=True)
    parser.add_argument('-a', '--adversarial_temperature', default=1, type=float)
    parser.add_argument('-b', '--batch_size', default=256, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=8, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.00005, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)


    parser.add_argument('--max_steps', default=80000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_steps', default=1000, type=int)
    parser.add_argument('--valid_steps', default=1000, type=int)
    parser.add_argument('--print_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    return parser.parse_args(args)


def save_embeddings(model, step, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    file_name = 'entity_' + str(step)
    entity_embedding = model.entity_embedding.detach().cpu().numpy()

    np.save(
        os.path.join(args.save_path, file_name),
        entity_embedding
    )

    rel_file_name = 'relation_' + str(step)
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, rel_file_name),
        relation_embedding
    )


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        print('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_DEVISE

    args.data_path = os.path.join(args.datadir, args.dataset, 'onto_file')



    # if args.init_checkpoint:
    #     override_config(args)
    if args.data_path is None:
        raise ValueError('data_path and dataset must be choosed.')

    args.save_path = os.path.join(args.data_path, 'save_onto_embeds')

    # if args.do_train and args.save_path is None:
    #     raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation

    print('Model: %s' % args.model)
    # print('Data Path: %s' % args.data_path + "/" + args.dataset)
    print('#entity num: %d' % nentity)
    print('#relation num: %d' % nrelation)

    all_triples = read_triple(os.path.join(args.data_path, 'all_triples.txt'), entity2id,
                              relation2id)
    print('#total triples num: %d' % len(all_triples))


    # All true triples
    all_true_triples = all_triples

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )

    # logging.info('Model Parameter Configuration:')
    # for name, param in kge_model.named_parameters():
    #     logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()

    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(all_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(all_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    print('Ramdomly Initializing %s Model...' % args.model)

    # step = init_step

    print('------ Start Training...')
    print('batch_size = %d' % args.batch_size)
    print('negative sample size = %d' % args.negative_sample_size)
    print('hidden_dim = %d' % args.hidden_dim)
    print('gamma = %f' % args.gamma)
    print('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))

    if args.negative_adversarial_sampling:
        print('adversarial_temperature = %f' % args.adversarial_temperature)

    print("learning rate = %f" % current_learning_rate)

    # Set valid dataloader as it would be evaluated during training

    if args.do_train:

        train_losses = []

        # Training Loop
        for step in range(1, args.max_steps + 1):

            loss_values = kge_model.train_step(kge_model, optimizer, train_iterator, args)

            train_losses.append(loss_values)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                print('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % args.print_steps == 0:
                pos_sample_loss = sum([losses['pos_sample_loss'] for losses in train_losses]) / len(train_losses)
                neg_sample_loss = sum([losses['neg_sample_loss'] for losses in train_losses]) / len(train_losses)
                loss1 = sum([losses['loss'] for losses in train_losses]) / len(train_losses)

                # log_metrics('Training average', step, metrics)
                print('Training Step: %d; average -> pos_sample_loss: %f; neg_sample_loss: %f; loss: %f' %
                      (step, pos_sample_loss, neg_sample_loss, loss1))
                train_losses = []

            if step % args.save_steps == 0:
                save_embeddings(kge_model, step, args)

            if args.evaluate_train and step % args.valid_steps == 0:
                print('------ Evaluating on Training Dataset...')
                metrics = kge_model.test_step(kge_model, all_triples, all_true_triples, args)
                log_metrics('Test', step, metrics)


if __name__ == '__main__':
    # random_seed = random.randint(1, 10000)
    # random_seed = 5487
    #
    # print("random seed:", random_seed)
    # random.seed(random_seed)
    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed)

    main(parse_args())
