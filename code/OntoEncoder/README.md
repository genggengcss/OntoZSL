

### Dataset Preparation
#### Images of Unseen Class

**AwA**: Download [AwA](http://cvml.ist.ac.at/AwA2/AwA2-data.zip) (13GB!) and uncompress it to the folder `'data/images/'`.
Note that we rename the awa class to its wordnet ID for conveniently training and testing.
```
python data/process_awa.py
```



#### for encoding AwA's ontological schema
```
python run.py --dataset AwA
```


then you need to run `process_structure_embed.py`, `process_text_embed.py` and `process_triple.py` in the folder text
```
python text_aware/train.py --rel_str_embed 100 --ent_str_embed 100 --ent_text_embed 300
```


#### for Wikidata-ZS
```
python gan_kgc.py --dataset Wiki --embed_model DistMult/TransE --embed_dim 50 --ep_dim 100 --fc1_dim 200 --pretrain_batch_size 128 --pretrain_subepoch 30 --pretrain_times 7000 --D_batch_size 64 --G_batch_size 64 --gan_batch_rela 8  --pretrain_feature_extractor --load_trained_embed
```
