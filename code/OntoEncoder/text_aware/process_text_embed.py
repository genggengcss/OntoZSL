# -*- coding: utf-8 -*-
import sys, os, re
import numpy as np
from collections import namedtuple, defaultdict
from itertools import count
from copy import deepcopy

import pickle
import json
from PIL import ImageFile
import codecs

import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable

from nltk.corpus import stopwords
import string
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

ImageFile.LOAD_TRUNCATED_IMAGES = True



train_wnid = ["n02071294", "n02363005", "n02110341", "n02123394", "n02106662", "n02123597", "n02445715", "n01889520", "n02129604", "n02398521", "n02128385", "n02493793", "n02503517", "n02480855", "n02403003", "n02481823", "n02342885", "n02118333", "n02355227", "n02324045", "n02114100", "n02085620", "n02441942", "n02444819", "n02410702", "n02391049", "n02510455", "n02395406", "n02129165", "n02134084", "n02106030", "n02403454", "n02430045", "n02330245", "n02065726", "n02419796", "n02132580", "n02391994", "n02508021", "n02432983"]
test_wnid = ["n02411705", "n02068974", "n02139199", "n02076196", "n02064816", "n02331046", "n02374451", "n02081571", "n02439033", "n02127482"]

train_name = ["killer+whale", "beaver", "dalmatian", "persian+cat", "german+shepherd", "siamese+cat", "skunk", "mole", "tiger", "hippopotamus", "leopard", "spider+monkey", "elephant", "gorilla", "ox", "chimpanzee", "hamster", "fox", "squirrel", "rabbit", "wolf", "chihuahua", "weasel", "otter", "buffalo", "zebra", "giant+panda", "pig", "lion", "polar+bear", "collie", "cow", "deer", "mouse", "humpback+whale", "antelope", "grizzly+bear", "rhinoceros", "raccoon", "moose"]
test_name = ["sheep", "dolphin", "bat", "seal", "blue+whale", "rat", "horse", "walrus", "giraffe", "bobcat"]

wnids = train_wnid + test_wnid
names = train_name + test_name





def get_embedding(entity_str, word_vectors, get_vector):
    try:
        feat = get_vector(word_vectors, entity_str)
        return feat
    except:
        feat = np.zeros(WORD_VEC_LEN)

    str_set = filter(None, re.split("[ \-_]+", entity_str))
    str_set = list(str_set)
    cnt_word = 0
    for i in range(len(str_set)):
        temp_str = str_set[i]
        try:
            now_feat = get_vector(word_vectors, temp_str)
            feat = feat + now_feat
            cnt_word = cnt_word + 1
        except:
            continue

    if cnt_word > 0:
        feat = feat / cnt_word
    return feat

def generate_text_embedding(ent2doc):


    # all_feats = list()

    has = 0
    cnt_missed = 0
    missed_list = []
    entities2vec = dict()
    for ent, doc in ent2doc.items():
        feat = np.zeros(shape=WORD_VEC_LEN, dtype='float32')

        doc = doc.replace('_', ' ')
        doc = doc.replace('-', ' ')
        # print(doc)

        options = doc.split()
        cnt_word = 0

        for option in options:
            now_feat = get_embedding(option.strip(), word_vectors, get_vector)
            if np.abs(now_feat.sum()) > 0:
                cnt_word += 1
                feat += now_feat
        if cnt_word > 0:
            feat = feat / cnt_word

        if cnt_word != len(options):
            print(ent, 'count:', cnt_word)

        if np.abs(feat.sum()) == 0:
            # print('cannot find word ' + class_name)
            cnt_missed = cnt_missed + 1
            missed_list.append(ent + "###" + doc)
            feat = feat

        else:
            has += 1
            feat = feat / (np.linalg.norm(feat) + 1e-6)

        # all_feats.append(feat)
            entities2vec[ent] = feat




    # all_feats = np.array(all_feats)
    # print(all_feats.shape)
    for each in missed_list:
        print(each)
    print('does not have semantic embedding: ', cnt_missed, 'has: ', has)


    return entities2vec
    # entities2vec = dict()
    # for i, ent in enumerate(ent_list):
    #     entities2vec[ent] = ent_matrix[i]
    #     # print(ent_matrix[i])
    #
    # return entities2vec




def glove_google(word_vectors, word):
    return word_vectors[word]

def get_glove_dict(txt_dir):
    print('load glove word embedding')
    txt_file = os.path.join(txt_dir, 'glove.6B.300d.txt')
    word_dict = {}
    feat = np.zeros(WORD_VEC_LEN)
    with open(txt_file) as fp:
        for line in fp:
            words = line.split()
            assert len(words) - 1 == WORD_VEC_LEN
            for i in range(WORD_VEC_LEN):
                feat[i] = float(words[i+1])
            feat = np.array(feat)
            word_dict[words[0]] = feat
    print('loaded to dict!')
    return word_dict


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

def load_domain_range(triples_file):
    text_file = codecs.open(triples_file, "r", "utf-8")
    lines = text_file.readlines()
    triples = list()
    for line in lines:
        line_arr = line.rstrip("\r\n").split("\t")
        head = line_arr[0]
        rel = line_arr[1]
        tail = line_arr[2]
        triples.append((head, rel, tail))
    return triples

if __name__ == "__main__":

    dataset = 'AwA'
    datadir = '../../data'

    DATASET_DIR = os.path.join(datadir, dataset)
    DATA_DIR = os.path.join(datadir, dataset, 'onto_file')



    entity_text_file = os.path.join(DATA_DIR, 'entities_names.dict')
    entity_file = os.path.join(DATA_DIR, 'entities.dict')
    triples_file = os.path.join(DATA_DIR, 'all_triples_names.txt')


    entities = loadDict(entity_file)
    entities_names = loadDict(entity_text_file)



    WORD_VEC_LEN = 300
    word_vectors = get_glove_dict('../../data/glove')
    get_vector = glove_google


    ent2doc = dict()
    with open(entity_text_file) as f_doc:
        lines = f_doc.readlines()

        for i in range(len(lines)):
            entity_text = lines[i].strip().split('\t')[1].strip()
            if entity_text == 'bodyshape':
                ent2doc[entity_text] = 'body shape'
            elif entity_text == 'bodypart':
                ent2doc[entity_text] = 'body part'
            elif entity_text == 'bodysize':
                ent2doc[entity_text] = 'body size'
            elif entity_text == 'hairanimal':
                ent2doc[entity_text] = 'hair animal'
            elif entity_text == 'nestspot':
                ent2doc[entity_text] = 'nest spot'
            elif entity_text == 'quadrapedal':
                ent2doc[entity_text] = 'quadrupedal'
            elif entity_text == 'leporid':
                ent2doc[entity_text] = 'rabbits and hares'
            elif entity_text == 'procyonid':
                ent2doc[entity_text] = 'raccoons coatis kinkajous olingos olinguitos ringtails and cacomistles'
            elif entity_text == 'lagomorph':
                ent2doc[entity_text] = 'gnawing mammal'
            elif entity_text == 'proboscidean':
                ent2doc[entity_text] = 'massive herbivorous mammals having tusks and a long trunk'
            elif entity_text == 'musteline_mammal':
                ent2doc[entity_text] = 'weasel badger otter mink marten polecat wolverine'
            else:
                ent2doc[entity_text] = entity_text





    entities2vec = generate_text_embedding(ent2doc)
    print(len(entities2vec.keys()))



    save_name = open(os.path.join(DATA_DIR, 'embeddings', 'Onto_Text_Embed.pkl'), 'wb')
    pickle.dump(entities2vec, save_name)









