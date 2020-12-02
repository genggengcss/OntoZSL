import tensorflow as tf
import os


# for WN-IMG
relation_structural_embeddings_size = 100
mapping_size = 100
entity_structural_embeddings_size = 100
entity_multimodal_embeddings_size = 300
dropout_ratio = 0.0
margin = 10
training_epochs = 1000
batch_size = 100
display_step = 1
activation_function = tf.nn.tanh
initial_learning_rate = 0.001


# Loading the data

all_triples_file = '/home/gyx/ZSL_May/ZSL_DATA/AWA2/Expri_DATA/KGE_GAN/kge_data/all_triples_names_htr.txt'
structural_embeddings_file = '/home/gyx/ZSL_May/ZSL_DATA/AWA2/Expri_DATA/KGE_GAN/embeddings/Onto_TransE.pkl'
entity_multimodal_embeddings_file = '/home/gyx/ZSL_May/ZSL_DATA/AWA2/Expri_DATA/KGE_GAN/embeddings/Onto_Text_Embed.pkl'



model_id = "Onto_Train_text_con"

checkpoint_best_valid_dir = "best_"+model_id+"/"
checkpoint_current_dir ="current_"+model_id+"/"


if not os.path.exists(checkpoint_best_valid_dir):
    os.mkdir(checkpoint_best_valid_dir)

if not os.path.exists(checkpoint_current_dir):
    os.mkdir(checkpoint_current_dir)


model_current_weights_file = checkpoint_current_dir + model_id + "_current"
current_model_meta_file = checkpoint_current_dir + model_id + "_current.meta"


