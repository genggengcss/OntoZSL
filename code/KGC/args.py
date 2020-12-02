# -*- coding: utf-8 -*-
import argparse
import random
def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="/home/gyx/ZSL_KGR/DATA/", type=str)
    parser.add_argument("--dataset", default="NELL", type=str)
    parser.add_argument("--embed_model", default='DistMult', type=str)
    parser.add_argument("--RansomSplit", action='store_true', default=True)

    # embedding dimension
    parser.add_argument("--embed_dim", default=100, type=int, help='dimension of triple embedding')
    parser.add_argument("--ep_dim", default=200, type=int, help='dimension of entity pair embedding')
    parser.add_argument("--fc1_dim", default=400, type=int, help='dimension of entity pair embedding')
    parser.add_argument("--noise_dim", default=15, type=int)

    # feature extractor pretraining related
    parser.add_argument("--pretrain_batch_size", default=64, type=int)
    parser.add_argument("--pretrain_few", default=30, type=int)
    parser.add_argument("--pretrain_subepoch", default=20, type=int)
    parser.add_argument("--pretrain_margin", default=10.0, type=float, help='pretraining margin loss')
    parser.add_argument("--pretrain_times", default=16000, type=int, help='total training steps for pretraining')
    parser.add_argument("--pretrain_loss_every", default=500, type=int)

    # adversarial training related
    # batch size
    parser.add_argument("--D_batch_size", default=256, type=int)
    parser.add_argument("--G_batch_size", default=256, type=int)
    parser.add_argument("--gan_batch_rela", default=2, type=int)
    # learning rate
    parser.add_argument("--lr_G", default=0.0001, type=float)
    parser.add_argument("--lr_D", default=0.0001, type=float)
    parser.add_argument("--lr_E", default=0.0005, type=float)
    # training times
    parser.add_argument("--train_times", default=6500, type=int)
    parser.add_argument("--D_epoch", default=5, type=int)
    parser.add_argument("--G_epoch", default=1, type=int)
    # log
    # parser.add_argument("--log_every", default=1000, type=int)
    parser.add_argument("--loss_every", default=50, type=int)
    parser.add_argument("--eval_every", default=500, type=int)
    # hyper-parameter
    parser.add_argument("--test_sample", default=20, type=int, help='number of synthesized samples')
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument('--REG_W', default=0.001, type=float)
    parser.add_argument('--REG_Wz', default=0.0001, type=float)
    parser.add_argument("--max_neighbor", default=50, type=int, help='neighbor number of each entity')
    parser.add_argument("--grad_clip", default=5.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)

    parser.add_argument("--fine_tune", action='store_true')
    parser.add_argument("--aggregate", default='max', type=str)
    parser.add_argument("--no_meta", action='store_true')

    # switch
    parser.add_argument("--generate_text_embedding", action='store_true')
    parser.add_argument("--pretrain_feature_extractor", action='store_true')
    parser.add_argument("--load_trained_embed", action='store_true', help='load well trained kg embeddings, such as DistMult')
    parser.add_argument("--trained_embed_path", default='')


    # parser.add_argument("--semantic_of_rel", default='Expri_DATA/rela_matrix_transe_55000_0919.npz')

    parser.add_argument("--semantic_of_rel", default='Expri_DATA/rela_matrix_distmult_15000.npz')

    # parser.add_argument("--semantic_of_rel", default='Expri_DATA/rela_matrix.npz')
    # parser.add_argument("--semantic_of_rel", default='Expri_DATA/multimodal/multimodal_text_nei_tfidf_con/rela_matrix_multimodal_140.npz')
    # parser.add_argument("--semantic_of_rel", default='Expri_DATA/rela_matrix_transe_generalizations_50000.npz')


    # parser.add_argument("--semantic_of_rel", default='Expri_DATA/rela_matrix_transe_52000.npz')
    parser.add_argument("--input_dim", default=600, type=int)
    parser.add_argument("--train_data", default='')
    parser.add_argument("--splitname", default='')


    parser.add_argument("--seed", type=int, default=6096)
    parser.add_argument('--device', type=int, default=1, help='device to use for iterate data, -1 means cpu [default: 0]')

    args = parser.parse_args()


    if args.RansomSplit:
        # args.splitname = 'four'
        args.save_path = args.datadir + args.dataset + '/Expri_DATA/models_train_split/'
        args.trained_embed_path = 'Embed_used_train_split/'
        args.train_data = 'datasplit/' + args.splitname+'_train_tasks.json'

    else:
        args.save_path = args.datadir + args.dataset + '/Expri_DATA/models_train_1004/'
        args.trained_embed_path = 'Embed_used_train_1004/'
        args.train_data = 'ori_train_tasks.json'



    if args.seed is None:
        args.seed = random.randint(1, 10000)

    # print("------HYPERPARAMETERS-------")
    # for k, v in vars(args).items():
    #     print(k + ': ' + str(v))
    # print("----------------------------")

    print("------HYPERPARAMETERS-------")
    print('training contains validation set: ' + str(args.TrainVal))
    print('relation embedding: ' + str(args.semantic_of_rel))
    print('input rel embedding dimension: ' + str(args.input_dim))
    print('data split: ' + str(args.splitname))
    print('random seed: ' + str(args.seed))

    print("----------------------------")

    return args

if __name__ == "__main__":
    read_options()

