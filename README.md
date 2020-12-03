# OntoZSL

Code and Data for the paper: "OntoZSL: Ontology-enhanced Zero-shot Learning".

### Dataset Description

#### IMGC
|Dataset| # Classes (Total/Seen/Unseen) | # Ontology Schema (Triples/Concepts/Properties) |  
|:------:|:------:|:------:| 
|**AwA**|50/40/10| 1,256/180/12| 
|**ImNet-A**|80/25/55|563/227/19| 
|**ImNet-O**|35/10/25|222/115/8| 

#### KGC
|Dataset| # Relations (Total/Train/Val/Test) | # Ontology Schema (Triples/Concepts/Properties) |
|:------:|:------:|:------:|
|**NELL-ZS**|139/10/32| 3,055/1,186/4|
|**Wikidata-ZS**|469/20/48|10,399/3,491/8|


### Requirements
- `python 3.5`
- `PyTorch >= 1.0.0`

### Dataset Preparation

#### Pre-trained Word Embeddings
You need to download pretrained [Glove](http://nlp.stanford.edu/data/glove.6B.zip) word embedding dictionary, uncompress it and put all files to the folder `data/glove`.


#### AwA
Download public image features and dataset split for [AwA](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip), uncompress it and put the files in **AWA2** folder to our folder `data/AwA/`.


#### ImageNet (ImNet-A, ImNet-O)

The original ImageNet feature file is too large (>100GB!), and the subsets we used in the paper are not small to put on Github, so we will release them by Google Drive later if the paper is accepted.

The above downloaded AwA file also contains the dataset splits of ImNet-A/O, you can put `split.mat` and `w2v.mat` in the **ImageNet** folder to our folder `data/ImageNet/`.


#### NELL-ZS & Wikidata-ZS
You can download the two datasets from [here](https://github.com/Panda0406/Zero-shot-knowledge-graph-relational-learning) and put them to the corresponding data folder.


### OntoZSL Training
The first thing you need to do is to train the text-aware ontology encoder using the code in the folder `code/OntoEncoder`, you can read more details at [code/OntoEncoder/README.md](/X_ZSL/README.md).
Secondly, with well-trained ontology embedding, you can take it as the input of generative model, see the codes in the folders `code/IMGC` and `code/KGC`. The running commands are listed in the corresponding README.md files.
*You can skip the first step if you just want to use the ontology embedding we trained, the files are in the corresponding directories*.