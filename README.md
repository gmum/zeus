# ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data

Code repository for [https://arxiv.org/abs/2505.10704](https://arxiv.org/abs/2505.10704).

Repository is based on the first version of TabPFN. The license is located in the [legal](legal) folder. Link to TabPFN2 repository 
[https://github.com/PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN).

## Abstract
Clustering tabular data remains a significant open challenge in data analysis and machine learning. 
Unlike for image data, similarity between tabular records often varies across datasets, 
making the definition of clusters highly dataset-dependent. Furthermore, 
the absence of supervised signals complicates hyperparameter tuning in deep learning clustering methods, 
frequently resulting in unstable performance. To address these issues and reduce the need for per-dataset tuning,
we adopt an emerging approach in deep learning: zero-shot learning. We propose ZEUS, 
a self-contained model capable of clustering new datasets without any additional training or fine-tuning. 
It operates by decomposing complex datasets into meaningful components that can then be clustered effectively. 
Thanks to pre-training on synthetic datasets generated from a latent-variable prior, 
it generalizes across various datasets without requiring user intervention. To the best of our knowledge, 
ZEUS is the first zero-shot method capable of generating embeddings for tabular data in a fully unsupervised manner. 
Experimental results demonstrate that it performs on par with or better than traditional clustering algorithms 
and recent deep learning-based methods, while being significantly faster and more user-friendly.

## Setup
Setup with conda environment.

```shell
conda create -n zeus python=3.11
conda activate zeus
pip install -r requirements.txt
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## Experiments
Details of ZEUS configuration parameters can be found in the [zeus/configs.py](zeus/configs.py) file.

## Pre-training
Pre-training can be performed using the following command: 

```shell
python pretrain.py nr_epochs=300 dim=30 use_pca=True num_test_datasets=200 num_categorical=3 pca_dim=30 learning_rate=2e-5 inf_method=KMEANS
```

## Model checkpoint
ZEUS checkpoint is available at [Google Drive](https://drive.google.com/file/d/1D7uikacymUnmmMxjUjBuCNIomqhBWS67/view?usp=sharing).


## Evaluation
The evaluation of ZEUS can be executed as follows:

```shell
python .\evaluation.py model_path=zeus.pt inf_method=KMEANS eval_dataset=OPENML metric_type=ARI results_file=openml.csv
```
