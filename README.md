# MGMA-DTI

# Dependencies

Dependencies: <br>
``` 
- python 3.8+ 
- torch>=0.7.1  
- numpy>=1.20.2   
- pandas>=1.2.4    
- scikit-learn>=0.24.2      
- yacs~=0.1.8    
- rdkit~=2021.03.2    
```
# Install Guide
```
$ conda create --name mgmadti python==3.10
$ caonda activate mgmadti
# clone the source code of MGMA-DTI
$ git clone https://github.com/ChangLi-tt/MGMA-DTI.git
$ cd MGMA-DTI

```

# 
Parameters:<br>
- `--data`: select `<dataset name>` from `BindingDB`, `BioSNAP`, `Human`
- `--split`: select `<split>` from `random`, `cold`

```
$ python main.py --data [--data] [--split]
```

