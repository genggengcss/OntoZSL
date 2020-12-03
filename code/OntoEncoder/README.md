
### Running Command

#### Default Ontology Encoder
Running the script `pretrain_struc.py` to pretrain the structural representation of Ontological Schema.

**For AwA & ImNet_A/O**
```
python pretrain_struc.py --dataset AwA/ImNet_A/ImNet_O --hidden_dim 100
```
**For NELL-ZS & Wikidata-ZS**
```
python pretrain_struc.py --dataset NELL/Wiki --hidden_dim 300
```

#### Text-aware Ontology Encoder
Running the scripts in the folder `text_aware` to learn text-aware ontology embedding.

- preprocess the structural/textual representation of ontology entities
```
python process_structure_embed.py
python process_text_embed.py/process_text_embed_kgc.py
python process_triple.py
```






#### for encoding AwA's ontological schema



then you need to run `process_structure_embed.py`, `process_text_embed.py` and `process_triple.py` in the folder text
```
python text_aware/train.py --dataset AwA --rel_str_embed 100 --ent_str_embed 100 --mapping_size 100
```


#### for encoding NELL's ontological schema
```
python run.py --dataset NELL --hidden_dim 300
```
then you need to run `process_structure_embed.py`, `process_text_embed.py` and `process_triple.py` in the folder text
```
python text_aware/train_kgc.py --dataset NELL --rel_str_embed 300 --ent_str_embed 300 --mapping_size 300
```



#### for encoding ImNet_A/O's ontological schema
```
python run.py --dataset ImageNet/ImNet_A
```
```
python run.py --dataset ImageNet/ImNet_O
```


then you need to run `process_structure_embed.py`, `process_text_embed.py` and `process_triple.py` in the folder text
```
python text_aware/train.py --dataset ImageNet/ImNet_A --rel_str_embed 100 --ent_str_embed 100  --mapping_size 100
```
