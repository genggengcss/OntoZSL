import os
import sys
import json

import numpy as np
import scipy.io as scio
import tensorflow as tf
import argparse

sys.path.append('../../..')
import code.OntoEncoder.text_aware.util as u


train_wnid = ["n02071294", "n02363005", "n02110341", "n02123394", "n02106662", "n02123597", "n02445715", "n01889520", "n02129604", "n02398521", "n02128385", "n02493793", "n02503517", "n02480855", "n02403003", "n02481823", "n02342885", "n02118333", "n02355227", "n02324045", "n02114100", "n02085620", "n02441942", "n02444819", "n02410702", "n02391049", "n02510455", "n02395406", "n02129165", "n02134084", "n02106030", "n02403454", "n02430045", "n02330245", "n02065726", "n02419796", "n02132580", "n02391994", "n02508021", "n02432983"]
test_wnid = ["n02411705", "n02068974", "n02139199", "n02076196", "n02064816", "n02331046", "n02374451", "n02081571", "n02439033", "n02127482"]

train_name = ["killer+whale", "beaver", "dalmatian", "persian+cat", "german+shepherd", "siamese+cat", "skunk", "mole", "tiger", "hippopotamus", "leopard", "spider+monkey", "elephant", "gorilla", "ox", "chimpanzee", "hamster", "fox", "squirrel", "rabbit", "wolf", "chihuahua", "weasel", "otter", "buffalo", "zebra", "giant+panda", "pig", "lion", "polar+bear", "collie", "cow", "deer", "mouse", "humpback+whale", "antelope", "grizzly+bear", "rhinoceros", "raccoon", "moose"]
test_name = ["sheep", "dolphin", "bat", "seal", "blue+whale", "rat", "horse", "walrus", "giraffe", "bobcat"]

wnids = train_wnid + test_wnid
names = train_name + test_name


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default='../../data', help='')
parser.add_argument('--dataset', default='AwA', help='')

parser.add_argument('--rel_str_embed', default=100, type=int, help='relation structural embeddings size')
parser.add_argument('--ent_str_embed', default=100, type=int, help='entity structural embeddings size')
parser.add_argument('--ent_text_embed', default=300, type=int, help='entity textual embeddings size')
parser.add_argument('--mapping_size', default=100, type=int, help='hidden layer size')
parser.add_argument('--dropout_ratio', default=0.0, type=float, help='')
parser.add_argument('--margin', default=10, type=int, help='')
parser.add_argument('--training_epochs', default=1000, type=int, help='')
parser.add_argument('--batch_size', default=100, type=int, help='')
parser.add_argument('--display_step', default=1, type=int, help='')
parser.add_argument('--initial_learning_rate', default=0.001, type=float, help='')
parser.add_argument('--activation_function', default='', help='')
param = parser.parse_args()


param.activation_function = tf.nn.tanh


# Loading the data
all_triples_file = os.path.join(param.datadir, param.dataset, 'onto_file', 'all_triples_names_htr.txt')
structural_embeddings_file = os.path.join(param.datadir, param.dataset, 'onto_file', 'embeddings', 'Onto_TransE.pkl')
entity_text_embeddings_file = os.path.join(param.datadir, param.dataset, 'onto_file', 'embeddings', 'Onto_Text_Embed.pkl')





DATASET_DIR = os.path.join(param.datadir, param.dataset)



# .... Loading the data ....
print("load all triples")
relation_embeddings = u.load_binary_file(structural_embeddings_file)  # "structure/FB_transE_100_norm.pkl"
entity_embeddings_txt = u.load_binary_file(structural_embeddings_file)  # "structure/FB_transE_100_norm.pkl"
entity_embeddings_img = u.load_binary_file(entity_text_embeddings_file)  # "multimodal/fb_vgg128_avg_fb_txt_normalized.pkl"


all_triples, entity_list = u.load_training_triples(all_triples_file)
print("#original entities: ", len(entity_list), "#original total triples", len(all_triples))

# filter the entity that not has multimodal embedding
entity_list_filtered = []
for e in entity_list:
    if e in entity_embeddings_img:
        entity_list_filtered.append(e)
entity_list = entity_list_filtered


triple_list_filtered = []
for h, t, r in all_triples:
    if h in entity_embeddings_txt and t in entity_embeddings_txt and h in entity_embeddings_img and t in entity_embeddings_img:
        triple_list_filtered.append((h, t, r))
all_triples = triple_list_filtered
# entity: 11757; triples:
# print(entity_list)
print("#filter no multimodal entities: ", len(entity_list), "#total triples", len(all_triples))
triples_set = [t[0] + "_" + t[1] + "_" + t[2] for t in all_triples]
triples_set = set(triples_set)


training_data = u.load_freebase_triple_data_multimodal(all_triples_file, entity_embeddings_txt,
                                                       entity_embeddings_img, relation_embeddings)


print("#training data", len(training_data))   # num: 285850


# random.seed(10)
# valid_data = random.sample(training_data, 100)
valid_data = training_data
valid_data = [(t[5], t[7], t[6]) for t in valid_data]
# valid_data = valid_data[:100]

print("valid_data",len(valid_data))

def max_norm_regulizer(threshold,axes=1,name="max_norm",collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights,clip_norm=threshold,axes=axes)
        clip_weights = tf.assign(weights,clipped,name=name)
        tf.add_to_collection(collection,clip_weights)
        return None
    return max_norm

max_norm_reg = max_norm_regulizer(threshold=1.0)

def my_dense(x, nr_hidden, scope, activation_fn=param.activation_function,reuse=None):
    with tf.variable_scope(scope):
        h = tf.contrib.layers.fully_connected(x, nr_hidden,
                                              activation_fn=activation_fn,
                                              reuse=reuse,
                                              scope=scope#, weights_regularizer= max_norm_reg
                                              )

        return h



# ........... Creating the model
with tf.name_scope('input'):
    # dim: 100
    r_input = tf.placeholder(dtype=tf.float32, shape=[None, param.rel_str_embed],name="r_input")

    # dim: 100
    h_pos_txt_input = tf.placeholder(dtype=tf.float32, shape=[None, param.ent_str_embed], name="h_pos_txt_input")
    h_neg_txt_input = tf.placeholder(dtype=tf.float32, shape=[None, param.ent_str_embed], name="h_neg_txt_input")

    # dim: 1000+128=1128
    h_pos_img_input = tf.placeholder(dtype=tf.float32, shape=[None, param.ent_text_embed],name="h_pos_img_input")
    h_neg_img_input = tf.placeholder(dtype=tf.float32, shape=[None, param.ent_text_embed], name="h_neg_img_input")

    t_pos_txt_input = tf.placeholder(dtype=tf.float32, shape=[None, param.ent_str_embed], name="t_pos_txt_input")
    t_pos_img_input = tf.placeholder(dtype=tf.float32, shape=[None, param.ent_text_embed], name="t_pos_img_input")

    t_neg_txt_input = tf.placeholder(dtype=tf.float32, shape=[None, param.ent_str_embed],   name="t_neg_txt_input")
    t_neg_img_input = tf.placeholder(dtype=tf.float32, shape=[None, param.ent_text_embed], name="t_neg_img_input")

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

with tf.name_scope('head_relation'):
    # structure
    r_mapped = my_dense(r_input, param.mapping_size, activation_fn=param.activation_function, scope="txt_proj", reuse=None)
    r_mapped = tf.nn.dropout(r_mapped, keep_prob)

    h_pos_txt_mapped = my_dense(h_pos_txt_input, param.mapping_size, activation_fn=param.activation_function,  scope="txt_proj", reuse=True)
    h_pos_txt_mapped = tf.nn.dropout(h_pos_txt_mapped, keep_prob)

    h_neg_txt_mapped = my_dense(h_neg_txt_input, param.mapping_size, activation_fn=param.activation_function,  scope="txt_proj", reuse=True)
    h_neg_txt_mapped = tf.nn.dropout(h_neg_txt_mapped, keep_prob)

    t_pos_txt_mapped = my_dense(t_pos_txt_input, param.mapping_size, activation_fn=param.activation_function,   scope="txt_proj", reuse=True)
    t_pos_txt_mapped = tf.nn.dropout(t_pos_txt_mapped, keep_prob)

    t_neg_txt_mapped = my_dense(t_neg_txt_input, param.mapping_size, activation_fn=param.activation_function,   scope="txt_proj", reuse=True)
    t_neg_txt_mapped = tf.nn.dropout(t_neg_txt_mapped, keep_prob)

    # Visual
    h_pos_img_mapped = my_dense(h_pos_img_input, param.mapping_size, activation_fn=param.activation_function, scope="img_proj", reuse=None)
    h_pos_img_mapped = tf.nn.dropout(h_pos_img_mapped, keep_prob)

    h_neg_img_mapped = my_dense(h_neg_img_input, param.mapping_size, activation_fn=param.activation_function, scope="img_proj", reuse=True)
    h_neg_img_mapped = tf.nn.dropout(h_neg_img_mapped, keep_prob)

    # Tail image ....
    t_pos_img_mapped = my_dense(t_pos_img_input, param.mapping_size, activation_fn=param.activation_function, scope="img_proj", reuse=True)
    t_pos_img_mapped = tf.nn.dropout(t_pos_img_mapped, keep_prob)

    t_neg_img_mapped = my_dense(t_neg_img_input, param.mapping_size, activation_fn=param.activation_function, scope="img_proj", reuse=True)
    t_neg_img_mapped = tf.nn.dropout(t_neg_img_mapped, keep_prob)




with tf.name_scope('cosine'):

    # Head model
    energy_ss_pos = tf.reduce_sum(abs(h_pos_txt_mapped + r_mapped - t_pos_txt_mapped), 1, keep_dims=True, name="pos_s_s")
    energy_ss_neg = tf.reduce_sum(abs(h_pos_txt_mapped + r_mapped - t_neg_txt_mapped), 1, keep_dims=True, name="neg_s_s")

    energy_is_pos = tf.reduce_sum(abs(h_pos_img_mapped + r_mapped - t_pos_txt_mapped), 1, keep_dims=True, name="pos_i_i")
    energy_is_neg = tf.reduce_sum(abs(h_pos_img_mapped + r_mapped - t_neg_txt_mapped), 1, keep_dims=True, name="neg_i_i")

    energy_si_pos = tf.reduce_sum(abs(h_pos_txt_mapped + r_mapped - t_pos_img_mapped), 1, keep_dims=True, name="pos_s_i")
    energy_si_neg = tf.reduce_sum(abs(h_pos_txt_mapped + r_mapped - t_neg_img_mapped), 1, keep_dims=True, name="neg_s_i")

    energy_ii_pos = tf.reduce_sum(abs(h_pos_img_mapped + r_mapped - t_pos_img_mapped), 1, keep_dims=True, name="pos_i_i")
    energy_ii_neg = tf.reduce_sum(abs(h_pos_img_mapped + r_mapped - t_neg_img_mapped), 1, keep_dims=True, name="neg_i_i")

    energy_concat_pos = tf.reduce_sum(abs((h_pos_txt_mapped + h_pos_img_mapped) + r_mapped - (t_pos_txt_mapped + t_pos_img_mapped)), 1, keep_dims=True, name="energy_concat_pos")
    energy_concat_neg = tf.reduce_sum(abs((h_pos_txt_mapped + h_pos_img_mapped) + r_mapped - (t_neg_txt_mapped + t_neg_img_mapped)), 1, keep_dims=True, name="energy_concat_neg")

    h_r_t_pos = tf.reduce_sum([energy_ss_pos, energy_is_pos, energy_si_pos, energy_ii_pos, energy_concat_pos], 0,   name="h_r_t_pos")
    h_r_t_neg = tf.reduce_sum([energy_ss_neg, energy_is_neg, energy_si_neg, energy_ii_neg, energy_concat_neg], 0, name="h_r_t_neg")


    # Tail model

    score_t_t_pos = tf.reduce_sum(abs(t_pos_txt_mapped - r_mapped - h_pos_txt_mapped), 1, keep_dims=True, name="pos_s_s")
    score_t_t_neg = tf.reduce_sum(abs(t_pos_txt_mapped - r_mapped - h_neg_txt_mapped), 1, keep_dims=True, name="neg_s_s")

    score_i_t_pos = tf.reduce_sum(abs(t_pos_img_mapped - r_mapped - h_pos_txt_mapped), 1, keep_dims=True, name="pos_i_i")
    score_i_t_neg = tf.reduce_sum(abs(t_pos_img_mapped - r_mapped - h_neg_txt_mapped), 1, keep_dims=True, name="neg_i_i")

    score_t_i_pos = tf.reduce_sum(abs(t_pos_txt_mapped - r_mapped - h_pos_img_mapped), 1, keep_dims=True, name="pos_s_i")
    score_t_i_neg = tf.reduce_sum(abs(t_pos_txt_mapped - r_mapped - h_neg_img_mapped), 1, keep_dims=True, name="neg_s_i")

    score_i_i_pos = tf.reduce_sum(abs(t_pos_img_mapped - r_mapped - h_pos_img_mapped), 1, keep_dims=True, name="pos_i_i")
    score_i_i_neg = tf.reduce_sum(abs(t_pos_img_mapped - r_mapped - h_neg_img_mapped), 1, keep_dims=True, name="neg_i_i")

    energy_concat_pos_tail = tf.reduce_sum(abs((t_pos_txt_mapped + t_pos_img_mapped) - r_mapped - (h_pos_txt_mapped + h_pos_img_mapped)), 1, keep_dims=True, name="energy_concat_pos_tail")
    energy_concat_neg_tail = tf.reduce_sum(abs((t_pos_txt_mapped + t_pos_img_mapped) - r_mapped - (h_neg_txt_mapped + h_neg_img_mapped)), 1, keep_dims=True, name="energy_concat_neg_tail")

    t_r_h_pos = tf.reduce_sum([score_t_t_pos, score_i_t_pos, score_t_i_pos, score_i_i_pos,energy_concat_pos_tail], 0, name="t_r_h_pos")
    t_r_h_neg = tf.reduce_sum( [score_t_t_neg, score_i_t_neg, score_t_i_neg, score_i_i_neg,energy_concat_neg_tail], 0,name="t_r_h_neg")


    kbc_loss1 = tf.maximum(0., param.margin - h_r_t_neg + h_r_t_pos)
    kbc_loss2 = tf.maximum(0., param.margin - t_r_h_neg + t_r_h_pos)


    kbc_loss = kbc_loss1 + kbc_loss2

    tf.summary.histogram("loss", kbc_loss)

#epsilon= 0.1
optimizer = tf.train.AdamOptimizer().minimize(kbc_loss)

summary_op = tf.summary.merge_all()

#..... start the training
saver = tf.train.Saver()

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


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

with tf.Session(config=sess_config) as sess:
    sess.run(tf.global_variables_initializer())

    initial_valid_loss = 100


    for epoch in range(param.training_epochs):

        np.random.shuffle(training_data)

       # training_data2 = training_data[:len(training_data)//3]
        training_loss = 0.
        total_batch = len(training_data) // param.batch_size +1

        for i in range(total_batch):

            batch_loss = 0
            start = i * param.batch_size
            end = (i + 1) * param.batch_size

            h_data_txt, h_data_img, r_data, t_data_txt, \
            t_data_img, t_neg_data_txt, t_neg_data_img, h_neg_data_txt, h_neg_data_img = u.get_batch_with_neg_heads_and_neg_tails_multimodal(
                training_data, triples_set, entity_list, start,
                end, entity_embeddings_txt, entity_embeddings_img)

            _, loss, summary = sess.run(
                [optimizer, kbc_loss, summary_op],
                feed_dict={r_input: r_data,
                           h_pos_txt_input: h_data_txt,
                           h_pos_img_input: h_data_img,

                           t_pos_txt_input: t_data_txt,
                           t_pos_img_input: t_data_img,

                           t_neg_txt_input: t_neg_data_txt,
                           t_neg_img_input: t_neg_data_img,

                           h_neg_txt_input: h_neg_data_txt,
                           h_neg_img_input: h_neg_data_img,

                           keep_prob: 1 - param.dropout_ratio#,
                           #learning_rate : param.initial_learning_rate
                           })
            #sess.run(clip_all_weights)
            # print(np.sum(loss))

            batch_loss = np.sum(loss)/param.batch_size

            training_loss += batch_loss


        training_loss = training_loss / total_batch

        # validating by sampling every epoch




        print("Epoch:", (epoch + 1), "loss=", str(round(training_loss, 4)))

        if (epoch + 1) >= 50 and (epoch + 1) % 10 == 0:

            # saver.save(sess, model_weights_best_valid_file)


            # test_in((epoch + 1), relation_embeddings, entity_embeddings_txt, entity_embeddings_img, entity_list, valid_data,
            #     all_triples)

            # mapping embeddings to ...

            entity_file = os.path.join(DATASET_DIR, 'onto_file', 'entities.dict')
            entity_name_file = os.path.join(DATASET_DIR, 'onto_file', 'entities_names.dict')

            entities = loadDict(entity_file)
            entities_names = loadDict(entity_name_file)

            # mapping structural embeddings
            w_dim = 100
            all_feats_txt = np.zeros((len(entities_names), 100), dtype=np.float)
            all_feats_img = np.zeros((len(entities_names), 300), dtype=np.float)

            for i, rel in enumerate(entities_names):
                # load embeddings
                if rel in entity_embeddings_img and rel in entity_embeddings_txt:
                    rel_embed_txt = entity_embeddings_txt[rel]
                    rel_embed_img = entity_embeddings_img[rel]
                    # all_feats_list.append(rel_embed_txt)
                    all_feats_txt[i] = rel_embed_txt
                    all_feats_img[i] = rel_embed_img
                else:
                    continue

            feats_maps_txt = sess.run(
                [h_pos_txt_mapped],
                feed_dict={
                    h_pos_txt_input: all_feats_txt,
                    keep_prob: 1 - param.dropout_ratio
                })
            feats_maps_img = sess.run(
                [h_pos_img_mapped],
                feed_dict={
                    h_pos_img_input: all_feats_img,
                    keep_prob: 1 - param.dropout_ratio
                })
            # print(feats_maps)
            # feats_maps = feats_maps[0]
            feats = np.hstack((feats_maps_txt[0], feats_maps_img[0]))
            print("mapped rela_matrix shape %s" % (str(feats.shape)))

            # save to .mat file

            matcontent = scio.loadmat(os.path.join(DATASET_DIR, 'att_splits.mat'))
            all_names = matcontent['allclasses_names'].squeeze().tolist()

            embed_size = feats.shape[1]
            o2v = np.zeros((len(all_names), embed_size), dtype=np.float)
            for i in range(len(all_names)):
                name = all_names[i][0]
                wnid = wnids[names.index(name)]
                o2v[i] = feats[entities.index(wnid)]

            # save wnids together
            save_name = 'o2v-55000-text' + str((epoch + 1)) + '.mat'
            save_dir = os.path.join(DATASET_DIR, 'onto_file', 'embeddings')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            n2v_file = os.path.join(save_dir, save_name)
            scio.savemat(n2v_file, {'o2v': o2v})


            # mapping both structural and multimodal embeddings
            # w_dim = 300
            # all_feats_txt = np.zeros((len(rel2id.keys()), w_dim), dtype=np.float)
            # all_feats_img = np.zeros((len(rel2id.keys()), w_dim), dtype=np.float)
            # for i, rel in id2rel.items():
            #     # load embeddings
            #     if rel in entity_embeddings_img and rel in entity_embeddings_txt:
            #         rel_embed_txt = entity_embeddings_txt[rel]
            #         rel_embed_img = entity_embeddings_img[rel]
            #         # all_feats_list.append(rel_embed_txt)
            #         all_feats_txt[i] = rel_embed_txt
            #         all_feats_img[i] = rel_embed_img
            #     else:
            #         continue
            # feats_maps_txt = sess.run(
            #     [h_pos_txt_mapped],
            #     feed_dict={
            #         h_pos_txt_input: all_feats_txt,
            #         keep_prob: 1 - param.dropout_ratio
            #     })
            # feats_maps_img = sess.run(
            #     [h_pos_img_mapped],
            #     feed_dict={
            #         h_pos_img_input: all_feats_img,
            #         keep_prob: 1 - param.dropout_ratio
            #     })
            #
            # # print(feats_maps)
            # feats = np.hstack((feats_maps_txt[0], feats_maps_img[0]))
            # print("mapped rela_matrix shape %s" % (str(feats.shape)))
            # save_name = 'rela_matrix_multimodal_' + str((epoch + 1)) + '.npz'
            # # np.savez(os.path.join('/home/gyx/ZSL_KGR/DATA/' + 'NELL' + '/Expri_DATA/multimodal_keywords', save_name),
            # #          relaM=feats)
            #
            # save_dir = '/home/gyx/ZSL_KGR/DATA/' + param.DATASET + '/Expri_DATA/multimodal_con'
            # if not os.path.exists(save_dir):
            #     os.mkdir(save_dir)
            # np.savez(os.path.join(save_dir, save_name), relaM=feats)

    saver.save(sess, param.model_current_weights_file)



