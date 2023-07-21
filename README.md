# Clustered Federated Learning based on Nonconvex Pairwise Fusion

This repository contains the official PyTorch implementatio of **Clustered Federated Learning based on Nonconvex Pairwise Fusion**.
```
@artical{yu2022clustered,
  title={Clustered Federated Learning based on Nonconvex Pairwise Fusion}, 
    author={Xue Yu and Ziyi Liu and Wu Wang and Yifan Sun},
    year={2022},
    eprint={2211.04218},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
## Software requirements

- Python3
- Numpy
- scikit-learn
- matplotlib
- Pytorch
- pandas

## Downloading dependencies

```python
pip3 install -r requirements.txt  
```

## General Guidelines

- For most of the baselines, we use the open-source code provided by the authors.

- we evaluate FPFC algorithm on diverse synthetic data and real-world federated datasets.

- The details of FPFC on a multi-classification problem with synthetic data are provided in `Synthetic` folder.

- For the evaluation of FPFC on the real-world datasets, please see details in  `LABELSWAP` folder.

Please see the README.md file in corresponding folder to get instructions of how to run these experimnets.

## Cite
Please cite our paper if you use this code in your research work.

## Questions/Bugs
Please submit a Github issue or contact xueyu_2019@ruc.edu.cn if you have any questions or find any bugs.

