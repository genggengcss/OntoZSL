## Running Command

### For NELL-ZS

#### With Pre-trained KG Embedding: TransE
- **original split**
```
python gan_kgc.py --dataset NELL --embed_model TransE --splitname ori  --embed_dim 100 --ep_dim 200 --fc1_dim 400 --pretrain_batch_size 64 --pretrain_subepoch 20 --pretrain_times 16000 --D_batch_size 256 --G_batch_size 256 --gan_batch_rela 2 --pretrain_feature_extractor --load_trained_embed --semantic_of_rel rela_matrix_onto_nell.npz
```
- **random split: one/two/three**
```
python gan_kgc.py --dataset NELL --embed_model TransE --splitname one  --embed_dim 100 --ep_dim 200 --fc1_dim 400 --pretrain_batch_size 64 --pretrain_subepoch 20 --pretrain_times 16000 --D_batch_size 256 --G_batch_size 256 --gan_batch_rela 2 --pretrain_feature_extractor --load_trained_embed --semantic_of_rel rela_matrix_onto_nell.npz
```

#### With Pre-trained KG Embedding: DistMult
- **original split**
```
python gan_kgc.py --dataset NELL --embed_model DistMult --splitname ori --embed_dim 100 --ep_dim 200 --fc1_dim 400 --pretrain_batch_size 64 --pretrain_subepoch 20 --pretrain_times 16000 --D_batch_size 256 --G_batch_size 256 --gan_batch_rela 2 --pretrain_feature_extractor --load_trained_embed --semantic_of_rel rela_matrix_onto_nell.npz
```
- **random split: one/two/three**
```
python gan_kgc.py --dataset NELL --embed_model DistMult --splitname one --embed_dim 100 --ep_dim 200 --fc1_dim 400 --pretrain_batch_size 64 --pretrain_subepoch 20 --pretrain_times 16000 --D_batch_size 256 --G_batch_size 256 --gan_batch_rela 2 --pretrain_feature_extractor --load_trained_embed --semantic_of_rel rela_matrix_onto_nell.npz
```


### For Wikidata-ZS

#### With Pre-trained KG Embedding: TransE
- **original split**
```
python gan_kgc.py --dataset Wiki --embed_model TransE --splitname ori --embed_dim 50 --ep_dim 100 --fc1_dim 200 --pretrain_batch_size 128 --pretrain_subepoch 30 --pretrain_times 7000 --D_batch_size 64 --G_batch_size 64 --gan_batch_rela 8  --pretrain_feature_extractor --load_trained_embed --semantic_of_rel rela_matrix_onto_wiki.npz
```
- **random split: one/two/three**
```
python gan_kgc.py --dataset Wiki --embed_model TransE --splitname one --embed_dim 50 --ep_dim 100 --fc1_dim 200 --pretrain_batch_size 128 --pretrain_subepoch 30 --pretrain_times 7000 --D_batch_size 64 --G_batch_size 64 --gan_batch_rela 8  --pretrain_feature_extractor --load_trained_embed --semantic_of_rel rela_matrix_onto_wiki.npz
```

*Other commands also follow similar settings.*