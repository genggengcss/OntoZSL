
## Running Command

### Default Ontology Encoder
Running the script `pretrain_struc.py` to pretrain the structural representation of Ontological Schema.

**For AwA & ImNet_A/O**
```
python pretrain_struc.py --dataset AwA --hidden_dim 100
python pretrain_struc.py --dataset ImNet_A --hidden_dim 100
python pretrain_struc.py --dataset ImNet_O --hidden_dim 100
```
**For NELL-ZS & Wikidata-ZS**
```
python pretrain_struc.py --dataset NELL --hidden_dim 300
python pretrain_struc.py --dataset Wiki --hidden_dim 300
```

### Text-aware Ontology Encoder
Running the scripts in the folder `text_aware/` to learn text-aware ontology embedding.

- Preprocess the structural/textual representation of ontology entities
```
python text_aware/process_structure_embed.py
python text_aware/process_text_embed.py or python text_aware/process_text_embed_kgc.py
python text_aware/process_triple.py
```

- Train the text-aware encoder model
**for AwA & ImNet_A/O**
```
python text_aware/train.py --dataset AwA --rel_str_embed 100 --ent_str_embed 100 --mapping_size 100
```

**for NELL-ZS & Wikidata-ZS**
```
python text_aware/train_kgc.py --dataset NELL --rel_str_embed 300 --ent_str_embed 300 --mapping_size 300
```

