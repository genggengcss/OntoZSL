# numpy data L2 nomarlization
import os
import json
import sys
import pickle as pkl
import numpy as np
import scipy.io as scio


def readTxt(file_name):
    class_list = list()
    wnids = open(file_name, 'rU')
    try:
        for line in wnids:
            line = line[:-1]
            class_list.append(line)
    finally:
        wnids.close()
    print(len(class_list))
    return class_list


###########################

def loadDict(file_name):
    entities = list()
    wnids = open(file_name, 'rU')
    try:
        for line in wnids:
            line = line[:-1]
            index, cls = line.split('\t')
            entities.append(cls)
    finally:
        wnids.close()
    print(len(entities))
    return entities



if __name__ == '__main__':

    datadir = '../../data'
    
    # dataset = 'AwA'
    dataset = 'ImageNet/ImNet_A'

    DATASET_DIR = os.path.join(datadir, dataset)
    DATA_DIR = os.path.join(datadir, dataset, 'onto_file')



    entity_file = os.path.join(DATA_DIR, 'entities_names.dict')
    relation_file = os.path.join(DATA_DIR, 'relations.dict')


    # load entity dict
    entities = loadDict(entity_file)
    relations = loadDict(relation_file)

    embed_dir = os.path.join(DATA_DIR, 'save_onto_embeds')

    ent_embed_file = embed_dir + '/entity_55000.npy'
    rel_embed_file = embed_dir + '/relation_55000.npy'

    ent_embeds = np.load(ent_embed_file)
    print(ent_embeds.shape)

    rel_embeds = np.load(rel_embed_file)
    print(rel_embeds.shape)

    embed_dict = dict()
    for i, ent in enumerate(entities):
        embed_dict[ent] = ent_embeds[i].astype('float32')
    for i, rel in enumerate(relations):
        embed_dict[rel] = rel_embeds[i].astype('float32')

    print(len(embed_dict.keys()))

    save_name = open(os.path.join(DATA_DIR, 'embeddings', 'Onto_TransE.pkl'), 'wb')
    pkl.dump(embed_dict, save_name)



