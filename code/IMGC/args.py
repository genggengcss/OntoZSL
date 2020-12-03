import argparse
import os

import random
def loadArgums():
    parser = argparse.ArgumentParser()
    '''
    Data loading
    '''
    parser.add_argument('--DATADIR', default='../../data', help='path to dataset')
    parser.add_argument('--Workers', default=2, help='number of data loading workers')

    # parser.add_argument('--DATASET', default='AwA', help='for awa')


    parser.add_argument('--DATASET', default='ImageNet/ImNet_A', help='for imagenet')
    parser.add_argument('--SeenFeaFile', default='', help='imagenet seen samples for training model')
    parser.add_argument('--SeenTestFeaFile', default='', help='imagenet seen samples for testing model')
    parser.add_argument('--UnseenFeaFile', default='', help='')
    parser.add_argument('--SplitFile', default='', help='')


    parser.add_argument('--SemEmbed', default='o2v', help='the type of class embedding to input')

    # AwA
    # parser.add_argument('--SemFile', default='o2v-55000-text120.mat', help='the file to store class embedding')
    # ImNet-A
    # parser.add_argument('--SemFile', default='o2v-45000-text120.mat', help='the file to store class embedding')
    # ImNet-O
    # parser.add_argument('--SemFile', default='o2v-45000-text130.mat', help='the file to store class embedding')

    parser.add_argument('--SemSize', type=int, default=100, help='size of semantic features')
    parser.add_argument('--NoiseSize', type=int, default=100, help='size of semantic features')
    parser.add_argument('--FeaSize', default=2048, help='size of visual features')


    parser.add_argument('--Unseen_NSample', type=int, help='extract the subset of unseen samples, for testing model')
    parser.add_argument('--PerClassAcc', action='store_true', default=False, help='testing the accuracy of each class')

    '''
    Generator and Discriminator
    '''
    parser.add_argument('--NetG_Path', default='', help='path to netG (to continue training)')
    parser.add_argument('--NetD_Path', default='', help='path to netD (to continue training)')
    parser.add_argument('--Pretrained_Classifier', default='', help='path to pretrain classifier (to continue training)')
    parser.add_argument('--NetG_Name', default='MLP_G', help='')
    parser.add_argument('--NetD_Name', default='MLP_CRITIC', help='')
    parser.add_argument('--NGH', default=4096, help='size of the hidden units in generator')
    parser.add_argument('--NDH', default=4096, help='size of the hidden units in discriminator')
    parser.add_argument('--Critic_Iter', default=5, help='critic iteration of discriminator, default=5, following WGAN-GP setting')
    parser.add_argument('--GP_Weight', type=float, default=10, help='gradient penalty regularizer, default=10, the completion of Lipschitz Constraint in WGAN-GP')
    parser.add_argument('--Cls_Weight', default=0.01, help='loss weight for the supervised classification loss')
    parser.add_argument('--SynNum', default=300, help='number of features generating for each unseen class; awa_default = 300')
    parser.add_argument('--SeenSynNum', default=300, help='number of features for training seen classifier when testing')

    '''
    Training Parameter
    '''
    parser.add_argument('--GZSL', action='store_true', default=False, help='enable generalized zero-shot learning')
    parser.add_argument('--PreProcess', default=True, help='enbale MinMaxScaler on visual features, default=True')
    parser.add_argument('--Standardization', default=False, help='')
    parser.add_argument('--Cross_Validation', default=False, help='enable cross validation mode')
    parser.add_argument('--Cuda', default=True, help='')
    parser.add_argument('--NGPU', default=1, help='number of GPUs to use')
    parser.add_argument('--CUDA_DEVISE', default='3', help='')
    parser.add_argument('--ManualSeed', default=9416, type=int, help='')  #
    parser.add_argument('--BatchSize', default=4096, help='')
    parser.add_argument('--Epoch', default=100, help='')
    parser.add_argument('--LR', default=0.0001, help='learning rate to train GAN')
    parser.add_argument('--Cls_LR', default=0.001, help='after generating unseen features, the learning rate for training softmax classifier')
    parser.add_argument('--Ratio', default=0.1, help='ratio of easy samples')
    parser.add_argument('--Beta', default=0.5, help='beta for adam, default=0.5')

    parser.add_argument('--OutFolder', default='./checkpoint/', help='folder to output data and model checkpoints')
    parser.add_argument('--OutName', default='imagenet', help='folder to output data and model checkpoints')
    parser.add_argument('--SaveEvery', default=100, help='')
    parser.add_argument('--PrintEvery', default=1, help='')
    parser.add_argument('--ValEvery', default=1, help='')
    parser.add_argument('--StartEvery', default=0, help='')

    args = parser.parse_args()

    if args.DATASET == 'AwA':
        args.FeaFile = 'res101.mat'
        args.SplitFile = 'binaryAtt_splits.mat'
        if args.SemEmbed == 'o2v':
            args.SemFile = os.path.join('onto_file', 'embeddings', args.SemFile)
            args.SemSize = 200
            args.NoiseSize = 100
        else:
            args.SemSize = 85
            args.NoiseSize = 85

    else:
        args.SplitFile = 'split.mat'
        args.SeenFeaFile = 'Res101_Features/ILSVRC2012_train'
        args.SeenTestFeaFile = 'Res101_Features/ILSVRC2012_val'
        args.UnseenFeaFile = 'Res101_Features/ILSVRC2011'

        if args.SemEmbed == 'o2v':
            args.SemFile = os.path.join('onto_file', 'embeddings', args.SemFile)
            args.SemSize = 200
            args.NoiseSize = 100
        else:
            args.SemEmbed = 'w2v'
            args.SemFile = 'w2v.mat'
            args.SemSize = 500
            args.NoiseSize = 500


    if args.ManualSeed is None:
        args.ManualSeed = random.randint(1, 10000)

    print("Random Seed: ", args.ManualSeed)
