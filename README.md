[TOC]



# iProL

> A novel predictor for identifying DNA promoters from sequence information based on Longformer pre-trained model



## iProL overview

![iproL_new](images/iproL_new.png)

Fig. 1 iProL overview. It includes (A) data construction, (B) five-fold cross-validation, and (C) model framework



## Dataset

Our experimental data came from RegulonDB and used the same benchmark dataset as BERT-Promoter. The dataset was originally provided by iPsW(2L)-PseKNC, and the complete data can be downloaded from https://regulondb.ccg.unam.mx/index.jsp.



## Longformer pre-trained model

The pre-trained model we use is named longformer-base-4096, which supports text sequences of up to 4096 length and can embed each word into a 768-dimensional vector. This pre-trained Longformer model can be downloaded from HuggingFace, specifically at https://huggingface.co/allenai/longformer-base-4096/tree/main.



## Run

```bash
cd ~/Documents/pbc/iProL/src
nohup jupyter notebook > ../jupyter_log/jupyter.log 2>&1 &
nohup ipython -c "%run ./main_iProL.ipynb" > ../jupyter_log/out.txt 2>&1 &
```



