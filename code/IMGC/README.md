## Running Command

#### For AwA

- **in Standard ZSL setting:**
```
python gan_imgc.py --DATASET AwA2 --ManualSeed 9182 --BatchSize 64 --LR 0.00001 --SemFile o2v-awa.mat --SynNum 300
```
- **in Generalized ZSL setting:**
```
python gan_imgc.py --DATASET AwA2 --ManualSeed 9182 --BatchSize 64 --LR 0.00001 --SemFile o2v-awa.mat --GZSL --SynNum 1800
```

#### For ImNet-A

- **in Standard ZSL Setting:**
```
python gan_imgc.py --DATASET ImageNet/ImNet_A --ManualSeed 9416 --BatchSize 4096 --LR 0.0001 --SemFile o2v-imagenet-a.mat
```
- **in Generalized ZSL setting:**
```
python gan_imgc.py --DATASET ImageNet/ImNet_A --ManualSeed 9416 --BatchSize 4096 --LR 0.0001 --SemFile o2v-imagenet-a.mat --GZSL
```

#### For ImNet-O:
- **in Standard ZSL Setting:**
```
python gan_imgc.py --DATASET ImageNet/ImNet_O --ManualSeed 9416 --BatchSize 4096 --LR 0.0001 --SemFile o2v-imagenet-o.mat
```
- **in Generalized ZSL setting:**
```
python gan_imgc.py --DATASET ImageNet/ImNet_O --ManualSeed 9416 --BatchSize 4096 --LR 0.0001 --SemFile o2v-imagenet-o.mat --GZSL
```