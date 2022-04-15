# OntoZSL

Code and Data for the paper: "[**OntoZSL: Ontology-enhanced Zero-shot Learning**](https://dl.acm.org/doi/10.1145/3442381.3450042)".
Yuxia Geng, Jiaoyan Chen, Zhuo Chen, Jeff Z. Pan, Zhiquan Ye, and Huajun Chen.
The Web Conference (WWW) 2021 Research Track.
> In this work, we propose to utilize ontology and generative adversarial network to deal with the zero-shot learning problems in image classification and KG completion.

### Dataset Description

#### IMGC
|Dataset| # Classes (Total/Seen/Unseen) | # Ontology Schema (Triples/Concepts/Properties) |  
|:------:|:------:|:------:| 
|**AwA**|50 / 40 / 10| 1,256 / 180 / 12| 
|**ImNet-A**|80 / 28 / 52|563 / 227 / 19|
|**ImNet-O**|35 / 10 / 25|222 / 115 / 8| 

#### KGC
|Dataset| # Relations (Total/Train/Val/Test) | # Ontology Schema (Triples/Concepts/Properties) |
|:------:|:------:|:------:|
|**NELL-ZS**|139 / 10 / 32| 3,055 / 1,186/4|
|**Wikidata-ZS**|469 / 20 / 48|10,399 / 3,491/8|


### Requirements
- `python 3.5`
- `PyTorch >= 1.5.0`

### Dataset Preparation

#### Word Embeddings
You need to download pretrained [Glove](http://nlp.stanford.edu/data/glove.6B.zip) word embedding dictionary, uncompress it and put all files to the folder `data/glove/`.


#### AwA2
Download public image features and dataset split for [AwA2](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip), uncompress it and put the files in **AWA2** folder to our folder `data/AwA2/`.


#### ImageNet (ImNet-A, ImNet-O)

Download the image features and the word embeddings of ImageNet classes as well as their splits from [here](https://drive.google.com/drive/folders/1An6nLXRRvlKSCbJoKKlqTNDvgN7PyvvW?usp=sharing) and put them to the folder `data/ImageNet/`.


#### NELL-ZS & Wikidata-ZS
You can download these two datasets from [here](https://github.com/Panda0406/Zero-shot-knowledge-graph-relational-learning) and put them to the corresponding data folder.


### OntoZSL Training
The first thing you need to do is to train the text-aware ontology encoder using the code in the folder `code/OntoEncoder`, you can get more details at [code/OntoEncoder/README.md](code/OntoEncoder/README.md).

Secondly, with well-trained ontology embedding, you can take it as the input of generative model, see the codes in the folders `code/IMGC` and `code/KGC`. The running commands are listed in the corresponding README.md files.

*Note: you can skip the first step if you just want to use the ontology embedding we learned, the files are provided in the corresponding directories*.

### How to Cite
If you find this code useful, please consider citing the following paper.
```bigquery
@inproceedings{geng2021ontozsl,
  author    = {Yuxia Geng and
               Jiaoyan Chen and
               Zhuo Chen and
               Jeff Z. Pan and
               Zhiquan Ye and
               Zonggang Yuan and
               Yantao Jia and
               Huajun Chen},
  editor    = {Jure Leskovec and
               Marko Grobelnik and
               Marc Najork and
               Jie Tang and
               Leila Zia},
  title     = {OntoZSL: Ontology-enhanced Zero-shot Learning},
  booktitle = {{WWW} '21: The Web Conference 2021, Virtual Event / Ljubljana, Slovenia,
               April 19-23, 2021},
  pages     = {3325--3336},
  publisher = {{ACM} / {IW3C2}},
  year      = {2021},
  url       = {https://doi.org/10.1145/3442381.3450042},
  doi       = {10.1145/3442381.3450042},
  timestamp = {Thu, 14 Oct 2021 10:04:23 +0200},
  biburl    = {https://dblp.org/rec/conf/www/GengC0PYYJC21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
### Acknowledgement
We refer to the code of [LisGAN](https://github.com/lijin118/LisGAN) and [ZSGAN](https://github.com/Panda0406/Zero-shot-knowledge-graph-relational-learning). Thanks for their contributions.
