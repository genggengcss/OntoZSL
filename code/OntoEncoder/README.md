

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
python text_aware/train.py --dataset AwA --rel_str_embed 100 --ent_str_embed 100 --ent_text_embed 300
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
python text_aware/train.py --dataset ImageNet/ImNet_A --rel_str_embed 100 --ent_str_embed 100 --ent_text_embed 300
```
