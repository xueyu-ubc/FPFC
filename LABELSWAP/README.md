# Clustered Federated Learning based on Nonconvex Pairwise Fusion

Implementation of FPFC on MNIST/FMNIST experiments.

## Running the experiments

- The `data/generate_mnist_dataset.py` file is used to generate MNIST datasets, and the dataset will be saved in `mnist_output/dataset.pth`. Similar to FMNIST datasets, the dataset will be saved in `fmnist_output/dataset.pth`.

- All the results will be automatically stored in `mnist_output/results_XX.pickle` and `fmnist_output/results_XX.pickle`, respectively where `XX` is the algorithm name.

- There is a main file "main.py" which allows running all algorithms.

## Run Demos

- run below commands with fine-tune parameter:
```
python main.py --dataset MNIST --algorithm FPFC --batch_size 100 --learning_rate 0.01 --a 3.7 --lamda 0.4 --xi 10e-4 --num_global_iters 200 --local_epochs 10 --num_users 20 --ratio 0.5 --dimension 400 --project_dir mnist_output --dataset_dir mnist_output
```

```
python main.py --dataset MNIST --algorithm FPFCL1 --batch_size 100 --learning_rate 0.01 --lamdaL1 0.4 --num_global_iters 200 --local_epochs 10 --num_users 20 --ratio 0.5 --dimension 400 --project_dir mnist_output --dataset_dir mnist_output
```    
  
