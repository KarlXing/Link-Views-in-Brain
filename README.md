# Link-Views-in-Brain
This repository contains the code for training neural networks to convert views in the work of "Linking Global Top-Down Views to First-Person Views in the Brain".

## Requirement
Python 3  
PyTorch >= 1.4  
Numpy  

## Steps
### 1. Unzip Dataset
```
cd Link-Views-in-Brain/data
cat transform_data.tgz.* | tar -xzvf -
cat transform_data_seq.tgz* | tar -xzvf -
cat transform_data_bidirection.tgz* | tar -xzvf -
```

### 2. Conduct Training
Three variants of experiments were conducted.
1. Convert top-down views to first-person views and vice versa. For example, 
```
python  main_transform.py --data-path data/transform_data  --batch-size 100  --lr  0.0001  --epochs 20000  --tag c2s-latent100  --latent-size 100  --save-freq 5000  --save-path checkpoints/c2s-latent100.pt;   
python  main_transform.py --data-path data/transform_data  --batch-size 100  --lr  0.0001  --epochs 20000  --tag s2c-latent100  --latent-size 100  --save-freq 5000  --save-path checkpoints/s2c-latent100.pt  --sim2camera ;
```

2. Based on 1, consider temporal component by converting sequential views. For example, 
```
python  main_transform_seq.py --data-path data/transform_data_seq  --batch-size 100  --lr  0.0001  --epochs 20000  --tag c2s-latent100-seq  --latent-size 100  --save-freq 5000  --save-path checkpoints/c2s-latent100-seq.pt;   
python  main_transform_seq.py --data-path data/transform_data_seq  --batch-size 100  --lr  0.0001  --epochs 20000  --tag s2c-latent10-seq  --latent-size 100  --save-freq 5000  --save-path checkpoints/s2c-latent100-seq.pt --sim2camera ;
```


3. Based on 1, conduct two conversions at the same time with the same neural network. For example, 
```
python  main_transform_bidirection.py --data-path data/transform_data_bidirection  --batch-size 100  --lr  0.0001  --epochs 20000  --tag bidirection-latent100 --latent-size 100  --save-freq 5000  --save-path checkpoints/bidirection-latent100.pt;   
```
