# SLGCN_cancer_prognosis

## Setup

In order to prepare the data and train the model we provide two Docker containers:
* DockerfileCpu: Designed to run Pytorchonlyon CPU,
* DockerfileServer:  Designed to run Pytorch either on CPU or GPU usingCUDA (requires that your system has a GPU).

These are provided as two files in the root folder. To use them, first, run either 
```bash
docker build -t cdrscan -f DockerfileCpu .
```
or
```bash
docker build -t cdrscan -f DockerfileServer .
```

from the root folder. Then run the container with:

```bash
docker run -d -it --name=cdrscan -p 5001:6006 -p 5000:8888 --mount type=bind,source="$(pwd)",target=/workspaces/ cdrscan:latest
```

which will create a container running in the machine. You can then attach to it by using:

```bash
docker attach cdrscan
```

From there you can generate the data and train the models.


## Making a knowledge graph
```bash
python scripts/prepare_data.py
```

## Config File
The following describes the expected config.json file. By default this file should be placed in the root of the project and should have the following attributes:

* FOLDER: Folder to store the generated features.

* CANCER_TYPE (true/false): True will include the Cancer Type features in the Sample Features.

* GRAPH_DATA_FILE: Source for the Pathway Commons dataset.
* ENSEMBLE_TO_HGNC_DATA_FILE: Ensembl to HGNC dictionary file.
* VERTICES_DIC: file to save the vertices integer encoding dictionary.
* RELATIONSHIPS_DIC: file to save the edges integer encoding dictionary.
* OUTPUT_GRAPH_FILE: file to save the graph.
* OUTPUT_NODE_FEATURES_FILE: file to save the node features.
* OUTPUT_SAMPLE_FEATURES_FILE: file to save the sample features.
* OUTPUT_LABELS_FILE: file to save the labels.
* EMBEDDING_SIZE: Length of the deep-features extracted in the networks (should match NUM_NODE_FEATURE in the current networks, but a different architecture can allow different values).
* NUM_NODE_FEATURE: Length of the encoded node feature vectors (should match EMBEDDING_SIZE in the current networks, but a different architecture can allow different values).
* BATCH_SIZE: Size of the batches.
* CROSS_VALIDATION_SPLITS: Number of cross validation splits.
* EPOCHS:Numberofepochstotraineachpartitionfromthecross-validation split.
* SAVING_ROOT_PATH: Folder to save the training outputs (models and results).
* MODEL_KEYWORD_NAME: Keyword for the model currently being trained. Used for saving and plot titles.
* RANDOM_STATE: Random seed for the cross validation splitting.
* NUMBER_PRINCIPAL_COMPONENTS: To not use PCA set to null. In order to use it, select an integer which will define the number of principal components to use.
* GRAPH_TRAINING (true/false): True will simultaneously update the weights in the graph model while training the samples model. False will freeze the graph network.
* LOSS_CLIP: If not null, perform loss clipping as described in Section 1.2.3.2.
* num_workers: Number of workers to use in the dataloader.
* MODEL: Select which to train from the available types of models: “SampleNet”, “SampleNetDropout”, “SampleNetFullResidual”, “SampleNetResConv”; for Simple Fully Connected Network, Simple Fully Connected Activated with Dropout Network, Fully Connected Resnet, and Convolutional Network respectively.
* LOSS: Select which type of loss to use: “MSE”/“Shr”(Shrinkage Loss (Section 1.2.3.1))/“BCE”/“BCEWithLogit”.
* REDUCTION: Reduction for the loss function: “none”, “sum”, or “mean”.
* PATIENCE: We use reduce-on-plateau optimizer scheduler which reduces the learning rate every PATIENCE epochs were the R2 value does not improve.
* LR: Initial learning rate.
* CUDA (true/false): Whether to use CUDA or not.
* TRAIN_FROM_K_PARTITION: If you are resuming training after stopping for any reason, and you wish to begin from a partition K instead of the first partition 0, input the partition to resume. (not implemented?)
* EXPLANATION_PATH: output the explanation for prediction in the test dataset. This file is saved as a joblib serialization file using joblib library.
* EXPLANATION_STRATEGY: selects an a strategy for which sample explanation to output by choicing from "all"/"binary_class_positive"/"binary_class_pred_positive"/"binary_class_positive_equal"/integer
  - integer: the number of samples at random
  - "all": to compute all samples
  - "binary_class_positive": samples with positive labels
  - "binary_class_pred_positive": samples with positive prediction
  - "binary_class_positive_equal": samples with positive labels and positive prediction
 

## Commands

To make a knowledge graph:
```bash
python -m scripts.prepare_data
```

To train the Graph Network:
```bash
python -m lib.slgcn_graph_train
```

Finally, after running the 2 previous commands, you can train the sample model training with:
```bash
python -m lib.slgcn_sample_train
```
