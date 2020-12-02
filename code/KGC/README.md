

### Dataset Preparation
#### Images of Unseen Class

**AwA**: Download [AwA](http://cvml.ist.ac.at/AwA2/AwA2-data.zip) (13GB!) and uncompress it to the folder `'data/images/'`. 
Note that we rename the awa class to its wordnet ID for conveniently training and testing.   
```
python data/process_awa.py
```



#### for NELL-ZS
```
python gan_kgc.py --dataset NELL --embed_model DistMult/TransE --embed_dim 100 --ep_dim 200 --fc1_dim 400 --pretrain_batch_size 64 --pretrain_subepoch 20 --pretrain_times 16000 --D_batch_size 256 --G_batch_size 256 --gan_batch_rela 2 --pretrain_feature_extractor --load_trained_embed
```


#### for Wikidata-ZS
```
python gan_kgc.py --dataset Wiki --embed_model DistMult/TransE --embed_dim 50 --ep_dim 100 --fc1_dim 200 --pretrain_batch_size 128 --pretrain_subepoch 30 --pretrain_times 7000 --D_batch_size 64 --G_batch_size 64 --gan_batch_rela 8  --pretrain_feature_extractor --load_trained_embed
```
