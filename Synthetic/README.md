# Clustered Federated Learning based on Nonconvex Pairwise Fusion

Implementation of FPFC on synthetic dataset.

## Running the experiments

- The `data/generate_synthetic_dataset.py` file is used to generate datasets, and the dataset will be saved in `output_synthetic/dataset.pth`.

- All the results will be automatically stored in `output_synthetic/results_XX.pickle`.

- There is a main file "main.py" which allows running all algorithms.

## Run Demos

- To produce the comparison experiments for FPFC using Synthetic dataset, here wo consider scenario S1 with 100 devices and 4 clusters.
run below commands with fine-tune parameter:
```
python main.py --dataset Synthetic --algorithm FPFC --batch_size 100 --learning_rate 0.1 --a 3.7 --lamda 0.6 --xi 10e-4 --threshold 0.1 --num_global_iters 2000 --local_epochs 10 --num_users 100 --ratio 0.3 --dimension 60 --project_dir output_synthetic --dataset_dir output_synthetic
```

```
python main.py --dataset Synthetic --algorithm FPFCL1 --batch_size 100 --learning_rate 0.1 --lamdaL1 0.2 --thresholdL1 4 --num_global_iters 2000 --local_epochs 10 --num_users 100 --ratio 0.3 --dimension 60 --project_dir output_synthetic --dataset_dir output_synthetic
```
  
- It is noted that each algorithm should be run at least 3 times and then the results are averaged.
  