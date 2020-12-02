# OntoZSL

Code and Data for the paper:  
"OntoZSL: Ontology-enhanced Zero-shot Learning".  

### Dataset Description

#### IMGC
|Dataset| # Classes (Total/Seen/Unseen) | # Ontology Schema (Triples/Concepts/Properties) |  
|:------:|:------:|:------:| 
|**AwA**|50/40/10| 1,256/180/12| 
|**ImNet-A**|80/25/55|563/227/19| 
|**ImNet-O**|35/10/25|222/115/8| 

#### KGC

You can skip this step if you just want to use the AZSL model we trained.

### Dataset Preparation

#### Pre-trained Word Embeddings

You need to download pretrained [Glove](http://nlp.stanford.edu/data/glove.6B.zip) word embedding dictionary, uncompress it and put all files to the folder `'data/glove'`.


#### AwA
Download public data splits and features for [AwA](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip), uncompress it and put **AWA2** folder to our folder `'data/AwA/'`.


#### AwA
Download public data splits and features for [AwA](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip), uncompress it and put **AWA2** folder to our folder `'data/AwA/'`.


#### ImageNet (ImNet-A, ImNet-O)

Original Feature file is too large (>100GB!), we will release the subsets we used in paper by Google Drive if the paper is accepted.

The above downloaded AwA file also contains the dataset splits of ImNet-A/O, you can put `split.mat` and `w2v.mat` to our folder 'data/ImageNet/'.
