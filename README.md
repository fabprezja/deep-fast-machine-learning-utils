# Deep Fast Machine Learning Utils
![Deep Fast Machine Learning Utils Logo](https://github.com/fabprezja/deep-fast-machine-learning-utils/assets/87379098/19075a60-524c-472c-bae2-1f101167f907)

Welcome to the Deep Fast Machine Learning Utils! This library is designed to streamline and expedite your machine learning prototyping process. It offers unique tools for model search and feature selection, which are not found in other ML libraries. The aim is to complement established libraries such as Tensorflow, Keras and Scikit-learn. Additionally, it provides extra tools for dataset management and visualization of training outcomes.

Documentation at: https://fabprezja.github.io/deep-fast-machine-learning-utils/

> **Note:** This library is in the early stages of development.

<a name="installation"></a>
## Installation

You can install the library directly using pip:

```bash
pip install deepfastmlu
```
<a name="citation"></a>
## Citation

If you find this library useful in your research, please consider citing:
```shell
@misc{fabprezja_2023_dfmlu,
  author = {Fabi Prezja},
  title = {Deep Fast Machine Learning Utils},
  month = sept,
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/fabprezja/deep-fast-vision}},
  doi = {10.5281/zenodo.8374468},
  url = {https://doi.org/10.5281/zenodo.8374468}
}
```
If you used the Adaptive Variance Threshold (AVT) class, please cite the following article:
```shell
Paper + Citation Comming Soon
```
> **Note:** When referencing, please consider additional attributions to Tensorflow, Scikit-learn, and Keras, as the library is built around them.

## Table of Contents

1. [Model Search](#neural-architecture-design)
    - [Principal Component Cascade Dense Neural Architecture Search (PCCDNAS)](#pccdnas)
2. [Feature Selection](#feature-selection)
    - [Adaptive Variance Threshold (AVT)](#adaptivevariancethreshold)
    - [Rank Aggregated Feature Selection](#rankaggregatedfs)
    - [Chained Feature Selection](#chainedfs)
    - [Mixing Feature Selection Approaches](#mixing-feature-selection-approaches)
3. [Extra Tools](#extra-tools)
    - [Data Management](#data-management)
        - [Data Splitter](#data-splitter)
        - [Data Sub Sampler (miniaturize)](#data-sub-sampler-miniaturize)
    - [Visualizing Results](#visualizing-results)
        - [Plot Validation Curves](#plot-validation-curves)
        - [Plot Generator Confusion Matrix](#plot-generator-confusion-matrix)

<a name="neural-architecture-design"></a>
## Model Search

<a name="pccdnas"></a>
### Principal Component Cascade Dense Neural Architecture Search (PCCDNAS)

PCCDNAS provides an automated method for designing dense neural networks. Using PCA (Principal Component Analysis), it systematically sets the number of neurons in each layer of the network. After applying PCA to the initial data, the neuron count for the first layer is determined based on the principal components (PCs) for a given variance threshold. Subsequently, the cascade mechanism ensures that the activations from each trained layer undergo PCA again. This process, in turn, determines the neuron count for the subsequent layers using the same principal component variance threshold criteria.

```shell
PCCDNAS Core Pseudo-Algorithm (Paper Comming Soon):

1. Initialize:
   - Create an empty neural network model.
   - Create an empty list to store the number of neurons for each layer.

2. Data Initialization:
   - Accept training data and labels.
   - Center or Normalize the data if required.

3. Initialize Model Search:
   - Set hyperparameters (e.g., number of layers, PCA variance threshold, etc.).

4. Build the Neural Network Model:
   - While not reached the desired number of layers:
     a. If at the first layer build stage, use the original training data.
     b. For subsequent layer build stage:
        - Train the model.
        - Extract the activations from the last layer (for each data point).
     c. Perform PCA on the data (original or activations).
     d. Determine the number of principal components that meet the variance threshold.
     e. Set the number of neurons in the next layer based on the determined principal components count.
     f. Add the layer to the model.
```

**Usage**:
```python
from deepfastmlu.model_search import PCCDNAS
from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the PCCDNAS object
pccdnas = PCCDNAS()

# Initialize data for the model search
pccdnas.data_init(X_train=X_train,
                  y_train=y_train,
                  validation=(X_val, y_val),
                  normalize=True,
                  unit=True)

# Initialize model search hyperparameters
pccdnas.initialize_model_search(
    epochs=10,  # Number of training epochs
    layers=3,  # Number of layers in the neural network
    activation='relu',  # Activation function for the layers
    pca_variance=[0.95,0.84,0.63],  # Desired explained variance for PCA for each layer
    loss='binary_crossentropy',  # Loss function for the model
    optimizer='adam',  # Optimizer for the model
    metrics=['accuracy'],  # List of metrics to be evaluated during training
    output_neurons=1,  # Number of neurons in the output layer
    out_activation='sigmoid',  # Activation function for the output layer
    stop_criteria='val_loss',  # Criteria for early stopping
    es_mode='min',  # Mode for early stopping (maximize the stop_criteria)
    dropout=0.2,  # Dropout rate for dropout layers
    regularize=('l2', 0.01),  # Regularization type ('l2') and value (0.01)
    batch_size=32,  # Batch size for training
    kernel_initializer='he_normal',  # Kernel initializer for the dense layers
    batch_norm=True,  # Whether to include batch normalization layers
    es_patience=5,  # Number of epochs with no improvement for early stopping
    verbose=1,  # Verbosity mode (1 = progress bar)
    learn_rate=0.001  # Learning rate for the optimizer
)
# Build the model
model, num_neurons = pccdnas.build()
print("Number of neurons in each layer:", num_neurons)
```
<a name="feature-selection"></a>
## Feature Selection

<a name="adaptivevariancethreshold"></a>
### Adaptive Variance Threshold (AVT)

Adaptive Variance Threshold is a feature selector that dynamically determines a variance threshold based on the provided percentile of the feature variances. Features with a variance below this threshold are dropped. Traditional (non-zero) variance-based feature selection uses a dataset dependent manual threshold, which is not optimal between datasets.

**Usage**:
```python
from sklearn.model_selection import train_test_split
from deepfastmlu.feature_select import AdaptiveVarianceThreshold

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize AdaptiveVarianceThreshold
avt = AdaptiveVarianceThreshold(percentile=1.5, verbose=True)

# Fit AVT to the training data
avt.fit(X_train)

# Transform both training and validation data
X_train_new = avt.transform(X_train)
X_val_new = avt.transform(X_val)
```
<a name="rankaggregatedfs"></a>
### Rank Aggregated Feature Selection

RankAggregatedFS is a feature selector that aggregates the rankings of features from multiple feature selection methods. It combines the scores or rankings of features from different methods to provide a unified ranking of features. This approach can be useful when there's uncertainty about which feature selection method to use, as it combines the strengths of multiple methods.

**Usage**:
```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler
from deepfastmlu.feature_select import RankAggregatedFS

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create feature selection methods
variance_selector = VarianceThreshold(threshold=0.0)
mi_selector = SelectKBest(score_func=mutual_info_classif, k=10)
f_classif_selector = SelectKBest(score_func=f_classif, k=10)

# Initialize RankAggregatedFS with multiple methods (excluding VarianceThreshold)
rank_aggregated_fs = RankAggregatedFS(methods=[mi_selector, f_classif_selector], k=10)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalize the data
    ('variance_threshold', variance_selector),  # Apply VarianceThreshold
    ('rank_aggregated_fs', rank_aggregated_fs)  # Apply RankAggregatedFS
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Transform the training and test data using the pipeline
X_train_new = pipeline.transform(X_train)
X_test_new = pipeline.transform(X_test)
```
<a name="chainedfs"></a>
### Chained Feature Selection

ChainedFS is a feature selector that sequentially applies a list of feature selection methods. This class allows for the chaining of multiple feature selection methods, where the output of one method becomes the input for the next. This can be particularly useful when one wants to combine the strengths of different feature selection techniques or when a sequence of operations is required to refine the feature set.

**Usage**:
```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from deepfastmlu.feature_select import ChainedFS

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create feature selection methods
variance_selector = VarianceThreshold(threshold=0.0)
k_best_selector = SelectKBest(score_func=mutual_info_classif, k=10)

# Initialize ChainedFS and create a pipeline
chained_fs = ChainedFS([variance_selector, k_best_selector])
pipeline = Pipeline([('feature_selection', chained_fs)])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Transform the training and test data using the pipeline
X_train_new = pipeline.transform(X_train)
X_test_new = pipeline.transform(X_test)
```
<a name="mixing-feature-selection-approaches"></a>
### Mixing Feature Selection Approaches

In this example we mix the previously shown methods.

**Usage**:
```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler
from deepfastmlu.feature_select import RankAggregatedFS,AdaptiveVarianceThreshold

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create feature selection methods
adaptive_variance_selector = AdaptiveVarianceThreshold(percentile=1.5)
mi_selector = SelectKBest(score_func=mutual_info_classif, k=10)
f_classif_selector = SelectKBest(score_func=f_classif, k=10)

# Initialize RankAggregatedFS with multiple methods
rank_aggregated_fs = RankAggregatedFS(methods=[mi_selector, f_classif_selector], k=10)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalize the data
    ('adaptive_variance_threshold', adaptive_variance_selector),  # Apply AdaptiveVarianceThreshold
    ('rank_aggregated_fs', rank_aggregated_fs)  # Apply RankAggregatedFS
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Transform the training and test data using the pipeline
X_train_new = pipeline.transform(X_train)
X_test_new = pipeline.transform(X_test)
```

<a name="extra-tools"></a>
## Extra Tools

<a name="data-management"></a>
### Data Management

<a name="data-splitter"></a>
#### Data Splitter

A class to split any folder based data instances into a partition format (train, val, test(s)). The splits are stratified.
```python
from deepfastmlu.extra.data_helpers import DatasetSplitter

# Define the paths to the original dataset and the destination directory for the split datasets
data_dir = 'path/to/original/dataset'
destination_dir = 'path/to/destination/directory'

# Instantiate the DatasetSplitter class with the desired train, validation, and test set ratios
splitter = DatasetSplitter(data_dir, destination_dir, train_ratio=0.7,
                           val_ratio=0.10, test_ratio=0.10, test_ratio_2=0.10, seed=42)

# Split the dataset into train, validation, and test sets
splitter.run()
```
<a name="data-sub-sampler-miniaturize"></a>
#### Data Sub Sampler (miniaturize)

A class to sub-sample (miniaturize) any folder based data instances give a ratio:
```python
from deepfastmlu.extra.data_helpers import DataSubSampler

# Define the paths to the original dataset and the destination directory for the subsampled dataset
data_dir = 'path/to/original/dataset'
subsampled_destination_dir = 'path/to/subsampled/dataset'

# Instantiate the DataSubSampler class with the desired fraction of files to sample
subsampler = DataSubSampler(data_dir, subsampled_destination_dir, fraction=0.5, seed=42)

# Create a smaller dataset by randomly sampling a fraction (in this case, 50%) of files from the original dataset
subsampler.create_miniature_dataset()
```
<a name="visualizing-results"></a>
### Visualizing Results

<a name="plot-validation-curves"></a>
#### Plot Validation Curves

Leverage the plot_history_curves function to visualize the training and validation metrics across epochs. This function displays the evolution of your model's performance but also highlights the minimum loss and maximum metric values to make insights clearer.

```python
from deepfastmlu.extra.plot_helpers import plot_history_curves

# Training the model
history = model.fit(X_train, y_train_onehot, validation_data=(X_val, y_val_onehot), epochs=25, batch_size=32)

# Visualize the training history
plot_history_curves(history, show_min_max_plot=True, user_metric='accuracy')
```
Example result:<br>
<img src="https://github.com/fabprezja/deep-fast-machine-learning-utils/assets/87379098/bb7f0143-0fd9-428d-a312-f2e17f24d409" alt="confs2" width="500">

<a name="plot-generator-confusion-matrix"></a>
#### Plot Generator Confusion Matrix

Utilize the plot_confusion_matrix function to effortlessly generate a confusion matrix for your model's predictions. Designed specifically for Keras image generators, it autonomously identifies class names, offering a straightforward way to gauge classification performance.

```python
from deepfastmlu.extra.plot_helpers import plot_confusion_matrix

# Create the confusion matrix for validation data
# model: A trained Keras model.
# val_generator: A Keras ImageDataGenerator used for validation.
# "Validation Data": Name of the generator, used in the plot title.
# "binary": Type of target labels ('binary' or 'categorical').
plot_confusion_matrix(model, val_generator, "Validation Data", "binary")
```
