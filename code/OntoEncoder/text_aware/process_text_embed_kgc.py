import sys, os, re
import numpy as np
from collections import namedtuple, defaultdict
import pickle
import codecs

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer



def generate_matrix(WordMatrix, corpus, corpus_tfidf, rela_list, w_dim=300):
    doc_Matrix = np.zeros(shape=(len(corpus), w_dim), dtype='float32')
    for num in range(len(corpus)):
        doc = corpus[num]
        doc_tfidf = corpus_tfidf[num]
        tmp_vec = np.zeros(shape=(w_dim,), dtype='float32')
        non_repeat = list()
        for i in range(len(doc)):
            # print '1', WordMatrix[doc[i]]
            # print doc_tfidf[i]
            # print '2', WordMatrix[doc[i]] * doc_tfidf[i]
            if doc[i] not in non_repeat:
                non_repeat.append(doc[i])
                tmp_vec += WordMatrix[doc[i]] * doc_tfidf[i]
        # print(rela_list[num], len(non_repeat), len(doc))
        tmp_vec = tmp_vec / float(len(non_repeat))
        doc_Matrix[num] = tmp_vec
        # break

    return doc_Matrix


def get_vocabulary(rela2text):
    vocab = defaultdict(float)
    for rela, text in rela2text.items():
        text_ = text.split()
        for word in text_:
            vocab[word] += 1
    return vocab





def load_wordembedding_300(vocab):
    word2id, word_vecs = dict(), dict()

    print('load glove word embedding')
    txt_file = os.path.join(txt_embed_dir, 'glove.6B.300d.txt')


    with open(txt_file) as fp:
        for line in fp:
            words = line.split()
            assert len(words) - 1 == WORD_VEC_LEN
            if words[0] in list(vocab.keys()):
                feat = np.zeros(WORD_VEC_LEN)
                for i in range(WORD_VEC_LEN):
                    feat[i] = float(words[i + 1])
                feat = np.array(feat)
                word_vecs[words[0]] = feat

    W = np.zeros(shape=(len(word_vecs), WORD_VEC_LEN), dtype='float32')
    i = 0
    for word in word_vecs:
        W[i] = word_vecs[word]
        word2id[word] = i
        i += 1

    return word2id, W


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets.
    Every dataset is lower cased
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_OOV(rela2doc, word2id):
    rela_list = list()
    corpus = list()
    corpus_text = list()
    for rela, doc in rela2doc.items():
        rela_list.append(rela)
        doc = doc.split()
        cleaned_doc_id = list()
        cleaned_doc = list()
        for word in doc:
            if word in word2id.keys():  # word2id.has_key(word):
                cleaned_doc_id.append(word2id[word])
                cleaned_doc.append(word)
            # else:
            #    print word,
        corpus.append(cleaned_doc_id)
        corpus_text.append(' '.join(cleaned_doc))
        # print '\n'
    return rela_list, corpus_text




def calculate_tfidf(ent_list, corpus, word2id):
    tfidf_vec = TfidfVectorizer(stop_words=stopwords.words('english'))
    # transformer=TfidfTransformer(stop_words=stopwords.words('english'))
    tfidf = tfidf_vec.fit_transform(corpus)
    word = tfidf_vec.get_feature_names()  # list, num of words
    weight = tfidf.toarray()  # (181, num of words)
    weight = weight.astype('float32')

    corpus_tfidf = list()
    corpus_new = list()
    for num in range(len(ent_list)):
        word2tfidf = zip(word, list(weight[num]))
        word2tfidf = dict(word2tfidf)
        assert len(word) == len(list(weight[num]))
        doc_tfidf = list()
        doc_ids = list()
        word_list = corpus[num].split()
        for w in word_list:
            if w in word:
                # if word2tfidf[w] < 0.05:
                #    continue
                doc_tfidf.append(word2tfidf[w])
                doc_ids.append(word2id[w])
                # if rela_list[num] == 'concept:athletebeatathlete':
                # print '  (', w, word2tfidf[w], ')  ',
        corpus_tfidf.append(doc_tfidf)
        corpus_new.append(doc_ids)

    assert len(corpus_tfidf) == len(ent_list)

    return corpus_tfidf, word, corpus_new

def load_triples(triples_file):
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

def generate_text_embedding(doc_file):
    # read NELL.ENT.DESC.txt

    ent2doc = dict()
    with open(doc_file) as f_doc:
        lines = f_doc.readlines()
        if dataset == 'NELL':
            for num in range(1063):
                ent = lines[3 * num].strip().split('###')[0].strip()
                description = lines[3 * num + 1].strip().split('###')[1].strip()
                description = clean_str(description)
                ent2doc[ent] = description
        if dataset == 'Wiki':
            for num in range(3491):
                ent = lines[4 * num].strip().split('###')[0].strip()
                name = lines[4 * num + 1].strip().split('###')[1].strip()
                description = lines[4 * num + 2].strip().split('###')[1].strip()
                description = name + ' ' + description
                description = clean_str(description)
                ent2doc[ent] = description

    # load triple (relation's domain and range)
    triples = load_triples(triples_file)
    ent_neighbors = defaultdict(list)
    for (h, r, t) in triples:
        ent_neighbors[h].append(t)
        ent_neighbors[t].append(h)
    print(len(ent_neighbors.keys()))

    ent2doc_new = dict()
    for ent, desc in ent2doc.items():
        new_desc = desc
        if len(ent_neighbors[ent]) > 0:
            for nei in ent_neighbors[ent]:
                if nei in ent2doc:
                    new_desc = new_desc + ' ' + ent2doc[nei]

        ent2doc_new[ent] = new_desc

    # Generate NELL description vocabulary
    vocab = get_vocabulary(ent2doc_new)
    print('Onto description vocab size %d' % (len(vocab)))

    word2id, WordMatrix = load_wordembedding_300(vocab)


    ent_list, corpus_text = clean_OOV(ent2doc_new, word2id)



    corpus_tfidf, vocab_tfidf, corpus = calculate_tfidf(ent_list, corpus_text, word2id)

    entity_matrix = generate_matrix(WordMatrix, corpus, corpus_tfidf, ent_list)
    print(entity_matrix.shape)

    ent2vecs = dict()
    for i, ent in enumerate(ent_list):
        ent2vecs[ent] = entity_matrix[i]


    save_name = open(os.path.join(DATA_DIR, 'embeddings', 'Onto_Text_Embed.pkl'), 'wb')
    pickle.dump(ent2vecs, save_name)


if __name__ == "__main__":

    WORD_VEC_LEN = 300

    datadir = '../../data'
    dataset = 'NELL'

    txt_embed_dir = '../../data/glove'

    DATA_DIR = os.path.join(datadir, dataset, 'onto_file')

    document_file = os.path.join(DATA_DIR, 'ENT.DESC.txt')
    triples_file = os.path.join(DATA_DIR, 'all_triples.txt')

    generate_text_embedding(document_file)


