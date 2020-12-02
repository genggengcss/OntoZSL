import os
import json
import sys
import pickle as pkl
import numpy as np
import scipy.io as scio




# AwA dataset split (id & name)

train_wnid = ["n02071294", "n02363005", "n02110341", "n02123394", "n02106662", "n02123597", "n02445715", "n01889520", "n02129604", "n02398521", "n02128385", "n02493793", "n02503517", "n02480855", "n02403003", "n02481823", "n02342885", "n02118333", "n02355227", "n02324045", "n02114100", "n02085620", "n02441942", "n02444819", "n02410702", "n02391049", "n02510455", "n02395406", "n02129165", "n02134084", "n02106030", "n02403454", "n02430045", "n02330245", "n02065726", "n02419796", "n02132580", "n02391994", "n02508021", "n02432983"]
test_wnid = ["n02411705", "n02068974", "n02139199", "n02076196", "n02064816", "n02331046", "n02374451", "n02081571", "n02439033", "n02127482"]

train_name = ["killer+whale", "beaver", "dalmatian", "persian+cat", "german+shepherd", "siamese+cat", "skunk", "mole", "tiger", "hippopotamus", "leopard", "spider+monkey", "elephant", "gorilla", "ox", "chimpanzee", "hamster", "fox", "squirrel", "rabbit", "wolf", "chihuahua", "weasel", "otter", "buffalo", "zebra", "giant+panda", "pig", "lion", "polar+bear", "collie", "cow", "deer", "mouse", "humpback+whale", "antelope", "grizzly+bear", "rhinoceros", "raccoon", "moose"]
test_name = ["sheep", "dolphin", "bat", "seal", "blue+whale", "rat", "horse", "walrus", "giraffe", "bobcat"]




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

def load_class():
    seen = readTxt(seen_file)
    unseen = readTxt(unseen_file)
    return seen, unseen

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


def save_embed(filename):

    # load embeddings
    embeds = np.load(filename)


    # save to .mat file
    matcontent = scio.loadmat(os.path.join(DATASET_DIR, 'att_splits.mat'))
    all_names = matcontent['allclasses_names'].squeeze().tolist()



    embed_size = embeds.shape[1]
    o2v = np.zeros((len(all_names), embed_size), dtype=np.float)



    for i in range(len(all_names)):
        name = all_names[i][0]
        wnid = wnids[names.index(name)]
        o2v[i] = embeds[entities.index(wnid)]

    print(o2v.shape)

    o2v_file = os.path.join(DATA_DIR, save_file)
    scio.savemat(o2v_file, {'o2v': o2v})



wnids = train_wnid + test_wnid
names = train_name + test_name

if __name__ == '__main__':

    dataset = 'AwA'

    datadir = '../../data'

    DATASET_DIR = os.path.join(datadir, dataset)
    DATA_DIR = os.path.join(datadir, dataset, 'onto_file')


    # load entity dict
    entity_file = os.path.join(DATA_DIR, 'entities.dict')
    entities = loadDict(entity_file)

    embed_dir = os.path.join(DATA_DIR, 'save_onto_embeds')

    embed_file = embed_dir + '/entity_55000.npy'


    save_file = 'o2v-55000.mat'

    save_embed(embed_file)












