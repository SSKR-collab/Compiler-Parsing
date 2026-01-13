# Time Series Classification with MiniRocket and Quantized Optimization
## Comprehensive Codebase Documentation

---

## Executive Summary

This project implements **time series classification** using **MiniRocket** (a fast deterministic feature extraction method) combined with **various optimization strategies** including standard Adam, Quantized Adam, and Dynamic Tree Quantization. The codebase compares different approaches on UCR time series datasets with the goal of achieving high classification accuracy while exploring optimization techniques for neural networks.

### What is Being Achieved:
- **Fast feature extraction** for time series data using MiniRocket transforms
- **Comparison of training strategies**: standard neural networks vs. MiniRocket features with quantized optimizers
- **Memory-efficient training** through quantization-aware training (QAT)
- **Support for multiple datasets** (UCR time series, MNIST, CIFAR-10)
- **Systematic evaluation** across different random seeds and hyperparameters

---

## Project Architecture

```
python_simulation/
├── requirements.txt                    # Python dependencies
├── parameters/
│   └── test.yaml                      # Configuration file for training
├── rocket/
│   ├── minirocket.py                  # Core MiniRocket implementation
│   ├── minirocket_dv.py               # MiniRocket for dilation variables
│   ├── minirocket_multivariate.py     # Multivariate extension
│   ├── minirocket_multivariate_variable.py
│   └── minirocket_variable.py         # Variable-length sequences support
├── datasets/
│   └── data.py                        # Data loading and preprocessing
├── training/
│   ├── loss.py                        # Loss and accuracy functions
│   └── quantized_adam.py              # Quantized Adam optimizer with dynamic tree quantization
├── optimizer/
│   └── cocob.py                       # COCOB optimizer (learning rate-free)
├── jax_training.py                    # JAX-based training loop
├── trainer.py                         # PyTorch-based training loop
├── softmax.py                         # MiniRocket with PyTorch implementation
├── evaluate.py                        # Evaluation and plotting utilities
├── compare_rocket_nn.py               # Comparison between ROCKET and NN results
└── test_AIfES.py                      # (Empty, for testing)
```

---

## Detailed Function Documentation

### 1. **core/jax_training.py**

#### Key Functions:

##### `init_weight(dim_in, dim_out, key) -> jax.Array`
- **Purpose**: Initialize neural network weights using uniform distribution
- **Parameters**:
  - `dim_in`: Input dimension
  - `dim_out`: Output dimension  
  - `key`: JAX random key
- **Returns**: Weight matrix of shape `(dim_out, dim_in)`
- **Details**: Uses uniform distribution with bounds `[-stdv, stdv]` where `stdv = 1/sqrt(dim_out)`

##### `init_bias(dim_out, key) -> jax.Array`
- **Purpose**: Initialize neural network biases
- **Parameters**:
  - `dim_out`: Output dimension
  - `key`: JAX random key
- **Returns**: Bias vector of shape `(dim_out,)`
- **Details**: Same uniform initialization as weights

##### `class FCLayer(eqx.Module)`
- **Purpose**: Fully connected layer module for JAX/Equinox
- **Attributes**:
  - `weight`: Weight matrix
  - `bias`: Bias vector
  - `activation`: Activation function callable
  - `activation_function_name`: Name of activation ('linear', 'relu', 'tanh')
- **Methods**:
  - `__init__(input_dim, output_dim, rng_key, activation_function='linear')`: Initialize layer
  - `__call__(x)`: Forward pass

##### `optim_step(model, loss, x, y, opt_state, optim)`
- **Purpose**: Perform a single optimization step (decorated with `@eqx.filter_jit` for JIT compilation)
- **Parameters**:
  - `model`: Neural network model
  - `loss`: Loss function
  - `x`: Input features
  - `y`: Target labels
  - `opt_state`: Optimizer state
  - `optim`: Optimizer instance
- **Returns**: Updated model, optimizer state, and loss value

##### `get_logger_name(dataset_name, seed, use_rocket, eval_dataset, quantize_adam, use_dynamic_tree_quantization, learning_rate, sample_dataset_iid=False) -> str`
- **Purpose**: Generate standardized filename for saving results
- **Returns**: Filename string with format `{dataset}_{seed}_{config}.p`
- **Details**: Encodes all training parameters in filename for easy result identification

##### `get_dataloader(ds, batch_size, num_workers, pin_memory=True) -> DataLoader`
- **Purpose**: Create PyTorch DataLoader with standard settings
- **Returns**: DataLoader object with shuffle=False

##### `class Trainer`
- **Purpose**: Main training orchestrator for JAX-based experiments
- **Key Methods**:
  - `__init__(params, seed)`: Initialize trainer with hyperparameters
  - `run()`: Main training loop iterating through epochs
  - `eval(batches)`: Evaluate model on validation/test sets
  - `print(text)`: Conditional printing based on config
  - `next_key()`: Generate new JAX random key

#### Trainer.run() Workflow:
1. For each epoch:
   - Load training batches
   - Transform features using RocketParameters if `use_rocket=True`
   - Perform optimization step
   - Evaluate on validation set
   - Save accuracies to pickle file
   - Log metrics

---

### 2. **core/trainer.py** (PyTorch Version)

#### Similar Structure to JAX version:

##### `get_logger_name(dataset_name, use_cocob, use_rocket, learning_rate)`
- **Purpose**: Generate filenames for PyTorch experiments
- **Returns**: Filename with format `{dataset}_{optimizer_config}.p`

##### `class Trainer` (PyTorch)
- **Purpose**: PyTorch-based training for comparison
- **Key Differences from JAX version**:
  - Uses `torch.nn` for model definition
  - Supports both ROCKET feature extraction and full neural networks
  - Optionally uses COCOB optimizer (deprecated)
  - Includes learning rate scheduler for Adam optimizer
- **Architecture Options**:
  - **If `use_rocket=True`**: Single linear layer on 10k features
  - **If `use_rocket=False`**: Multi-layer network with Tanh activation

##### `run_batch(param_array)` 
- **Purpose**: Parallel execution of multiple training runs
- **Uses**: Python multiprocessing with process pool
- **Returns**: Results from all parallel jobs

---

### 3. **rocket/minirocket.py**

This implements the MiniRocket algorithm from the paper: *"MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification"* by Dempster, Schmidt, & Webb (2012.08791)

#### Key Functions:

##### `_fit_biases(X, dilations, num_features_per_dilation, quantiles) -> ndarray`
- **Purpose**: Compute bias terms for convolutional kernels
- **Parameters**:
  - `X`: Training data of shape `(num_examples, input_length)`
  - `dilations`: Array of dilation factors
  - `num_features_per_dilation`: Number of features for each dilation
  - `quantiles`: Quantile values for bias computation
- **Algorithm**:
  1. Generate all 84 combinations of 3 indices from 9 kernel positions
  2. For each kernel/dilation:
     - Randomly select training example
     - Compute convolutions with different dilation rates
     - Extract bias as quantile of convolution output
- **Optimization**: Uses Numba JIT compilation for speed

##### `_fit_dilations(input_length, num_features, max_dilations_per_kernel) -> tuple`
- **Purpose**: Determine optimal dilation factors for kernels
- **Parameters**:
  - `input_length`: Length of time series
  - `num_features`: Desired number of features
  - `max_dilations_per_kernel`: Max dilations per kernel
- **Returns**: `(dilations, num_features_per_dilation)`
- **Algorithm**: Uses log-spaced distribution of dilations

##### `_quantiles(n) -> ndarray`
- **Purpose**: Generate low-discrepancy quantile sequence
- **Uses**: Golden ratio-based sequence: `(i * phi) % 1`
- **Returns**: Array of n quantiles in range [0, 1]

##### `fit(X, num_features=10_000, max_dilations_per_kernel=32) -> tuple`
- **Purpose**: Fit MiniRocket parameters from training data
- **Returns**: `(dilations, num_features_per_dilation, biases)`
- **Public API entry point**

##### `_PPV(a, b) -> float`
- **Purpose**: Proportion of Positive Values kernel (vectorized)
- **Logic**: Returns 1 if `a > b`, else 0
- **Vectorized**: Using Numba vectorize decorator

##### `transform(X, parameters) -> ndarray`
- **Purpose**: Extract MiniRocket features from time series data
- **Parameters**:
  - `X`: Input data of shape `(num_examples, input_length)`
  - `parameters`: Tuple of (dilations, num_features_per_dilation, biases)
- **Returns**: Transformed features of shape `(num_examples, num_features)`
- **Details**: Computes convolutions for all kernel/dilation combinations and applies PPV

---

### 4. **datasets/data.py**

#### Key Functions:

##### `load_ucr_dataset(name, test=False, num_trajectories=0) -> tuple`
- **Purpose**: Load UCR time series dataset
- **Parameters**:
  - `name`: Dataset name (e.g., "ElectricDevices")
  - `test`: If True, load test set; else load training set
  - `num_trajectories`: Limit number of samples (0 = all)
- **Preprocessing**:
  1. Read CSV file from `~/datasets/{name}/{name}_TRAIN.tsv`
  2. Interpolate missing values
  3. Shuffle data
  4. Convert class labels to 0-indexed integers
- **Returns**: `(X, y)` tuples of float32 and int64 arrays

##### `quantize_8_bit(data, offset, scaling) -> ndarray`
- **Purpose**: Quantize continuous data to 8-bit integer range
- **Formula**: `clip((data - offset) / scaling * 127, -127, 127)`
- **Used for**: Memory-efficient feature representation

##### `class MnistDataloader`
- **Purpose**: Load MNIST dataset from binary files
- **Methods**:
  - `read_images_labels(images_filepath, labels_filepath)`: Parse MNIST binary format
  - `load_data()`: Load training and test sets (limited to 1000 samples)

##### `class ClassificationDataset`
- **Purpose**: Main dataset wrapper supporting multiple data sources
- **Constructor Parameters**:
  - `params`: Configuration dictionary
  - `seed`: Random seed for reproducibility
  - `sample_dataset_iid`: If True, sample data IID; else reshuffle for non-IID
- **Supported Datasets**:
  - "Mnist": MNIST handwritten digits
  - "cifar": CIFAR-10 dataset
  - Any UCR time series dataset name
- **Key Attributes**:
  - `train_ds`: Training dataset
  - `eval_ds`: Evaluation/validation dataset
  - `test_ds`: Test dataset
  - `num_features`: Dimension of features after ROCKET transform
  - `num_classes`: Number of classes
  - `length_timeseries`: Original time series length
- **Processing Steps**:
  1. Load raw data based on dataset type
  2. Normalize data (mean=0, std=1)
  3. Apply 8-bit quantization if using ROCKET
  4. Apply MiniRocket feature extraction if enabled
  5. Split into train/eval/test sets
  6. Create PartDataset wrappers

##### `class PartDataset(Dataset)`
- **Purpose**: PyTorch Dataset wrapper
- **Methods**:
  - `__len__()`: Return number of samples
  - `__getitem__(idx)`: Return dict with 'input' and 'target' keys

---

### 5. **training/loss.py**

#### Key Functions:

##### `cross_entropy(y, pred_y) -> float`
- **Purpose**: Compute cross-entropy loss per sample
- **Parameters**:
  - `y`: Target class index
  - `pred_y`: Logits from model
- **Computation**: `log_softmax(pred_y)[y]`

##### `accuracy(y, y_pred) -> float`
- **Purpose**: Compute per-sample accuracy as percentage
- **Parameters**:
  - `y`: True label
  - `y_pred`: Predicted logits
- **Logic**: `(argmax(y_pred) == y) * 100`

##### `loss_batch(loss, y1, y2) -> float`
- **Purpose**: Batch-level loss computation
- **Uses**: `jax.vmap` to vectorize loss function across batch
- **Decorator**: `@eqx.filter_jit` for JIT compilation

##### `acc_batch(y1, y2) -> float`
- **Purpose**: Batch-level accuracy computation
- **Uses**: Vectorized accuracy function

##### `loss_func(model, loss, x, y) -> float`
- **Purpose**: Full forward pass computing loss
- **Steps**:
  1. Vectorize model forward pass: `jax.vmap(model)(x)`
  2. Compute batch loss
- **Returns**: Average loss over batch

##### `loss_acc_func(model, loss, x, y) -> tuple`
- **Purpose**: Compute both loss and accuracy in single pass
- **Returns**: `(loss, accuracy)` tuple

##### `loss_acc_grad_func(model, loss, x, y) -> tuple`
- **Purpose**: Compute loss, accuracy, and gradients
- **Returns**: `(loss, accuracy, gradients)` tuple

---

### 6. **training/quantized_adam.py**

#### Quantization Functions:

##### `quantization_round(x) -> float`
- **Purpose**: Custom differentiable rounding function
- **Uses**: JAX custom_jvp for gradient definition
- **VJP**: Gradient passes through unchanged (straight-through estimator)

##### `param_quantize(x, scaling, bits=8) -> int`
- **Purpose**: Quantize parameters to fixed-point representation
- **Algorithm**:
  1. Scale by parameter scaling factor
  2. Clip to `[-127, 127]` for 8-bit
  3. Round to nearest integer
- **Returns**: Quantized integer values

##### `param_dequantize(x, scaling, bits=8) -> float`
- **Purpose**: Reconstruct floating-point from quantized form
- **Formula**: `x / 127 * scaling`

##### `param_calculate_scaling(x) -> float`
- **Purpose**: Compute per-parameter scaling factor
- **Returns**: `max(|x|)`

##### `quantize_graph(graph, bits=8) -> tuple`
- **Purpose**: Quantize all parameters in model
- **Returns**: `(quantized_params, scaling_factors)`

##### `dequantize_graph(graph, scalings, bits=8) -> tree`
- **Purpose**: Reconstruct full-precision model from quantized form

##### `generate_dynamic_tree_quantization_fn() -> tuple`
- **Purpose**: Create dynamic tree quantization (logarithmic quantization)
- **Returns**: `(quantize_graph, dequantize_graph)` functions
- **Key Idea**: Uses hierarchical quantization with power-of-2 scaling
- **Quantization Scheme**:
  1. 256 discrete quantization values per tile
  2. Per-tile scaling factors (power of 10)
  3. Mantissa and exponent representation
  4. Supports both positive and negative values

##### `class QuantizedAdam`
- **Purpose**: Adam optimizer with quantization support
- **Constructor Parameters**:
  - `learning_rate`: Learning rate for Adam
  - `use_dynamic_tree_quantization`: If True, use dynamic tree; else standard quantization
- **Methods** (inherited from Optax):
  - `init(params)`: Initialize optimizer state
  - `update(grads, state, params)`: Compute parameter updates

---

### 7. **optimizer/cocob.py**

#### COCOB Optimizer Implementation:

##### `class COCOB_Backprop_old(Optimizer)`
- **Purpose**: Coin Betting Optimizer for learning-rate-free training
- **Reference**: *"Training Deep Networks without Learning Rates Through Coin Betting"* (NIPS 2017)
- **Key Features**:
  - No learning rate hyperparameter
  - Automatically adapts step sizes
- **Constructor Parameters**:
  - `params`: Model parameters
  - `weight_decay`: L2 regularization coefficient
  - `alpha`: Scaling factor (default: 100)
- **Internal State**:
  - `W1`: Initial weights
  - `L`: Running max of gradient magnitudes
  - `G`: Running sum of gradient magnitudes
  - `Reward`: Accumulated reward from good decisions
  - `Theta`: Cumulative gradients
  - `Theta`: Parameter updates
- **Status**: Currently DEPRECATED (see trainer.py line 113)

---

### 8. **evaluate.py**

#### Evaluation and Comparison Functions:

##### `plot_data(file, color, label, use_jax=True) -> None`
- **Purpose**: Load and plot accuracy curves
- **Inputs**:
  - `file`: Filename to load
  - `color`: Line color for matplotlib
  - `label`: Whether to show label
  - `use_jax`: If True, look in JAX results; else PyTorch results

##### `Main Execution Block`:
- Loads results from pickle files
- Plots training curves for different configurations
- Compares:
  - Quantized vs. Non-quantized Adam
  - ROCKET vs. Neural Network features
  - Different learning rates

---

### 9. **compare_rocket_nn.py**

#### Detailed Comparison Functions:

##### `load_data(dataset_name, seed, use_rocket, eval_dataset, quantize_adam, use_dynamic_tree_quantization, sample_dataset_iid, learning_rate) -> ndarray`
- **Purpose**: Load saved accuracy curves from disk
- **Returns**: Array of accuracy values across epochs
- **Handles**: Different directory paths for different configurations

##### `get_final_accuracy(dataset_name, seed, use_rocket, quantize_adam, use_dynamic_tree_quantization, sample_dataset_iid, learning_rate) -> tuple`
- **Purpose**: Extract final test accuracy using validation-best performance
- **Algorithm**:
  1. Load validation accuracies
  2. Load test accuracies
  3. Find epoch with best validation accuracy
  4. Return test accuracy at that epoch
- **Returns**: `(success, final_accuracy)` tuple

##### `plot_comparison_entire_dataset() -> None`
- **Purpose**: Compare ROCKET vs. Neural Network across all datasets
- **Comparison**: Non-IID ROCKET vs. IID Neural Network
- **Outputs**:
  - Scatter plot of accuracies
  - CSV with comparison results
  - Histogram of accuracy differences

##### `plot_comparison_quantized_adam() -> None`
- **Purpose**: Compare different quantization strategies
- **Comparisons**:
  1. Quantized Adam vs. Quantized Adam + Dynamic Tree
  2. Standard Adam vs. Quantized Adam + Dynamic Tree
- **Outputs**: Scatter plots and improvements metrics

---

### 10. **softmax.py**

#### PyTorch-based MiniRocket Training:

##### `train(path, num_classes, training_size, **kwargs) -> tuple`
- **Purpose**: Train MiniRocket + linear classifier on CSV data
- **Parameters**:
  - `path`: Path to training data CSV
  - `num_classes`: Number of output classes
  - `training_size`: Number of training samples
  - `**kwargs`: Override default hyperparameters
- **Default Hyperparameters**:
  - `num_features`: 10,000
  - `minibatch_size`: 256
  - `lr`: 1e-4
  - `max_epochs`: 50
  - `validation_size`: 2^11 = 2048
- **Returns**: `(parameters, best_model, f_mean, f_std)`
- **Training Features**:
  - Caching of transformed features to disk to save computation
  - Learning rate scheduling (ReduceLROnPlateau)
  - Early stopping with patience

##### `predict(path, parameters, model, f_mean, f_std, **kwargs) -> ndarray`
- **Purpose**: Make predictions on test data
- **Returns**: Predicted class labels

---

## Configuration File (parameters/test.yaml)

```yaml
show_print: True                          # Enable logging
saving_path: ../../jax_results           # Where to save results

# Model Options
use_rocket: True                         # Use MiniRocket features
num_neurons: 17                          # Hidden neurons (if not using ROCKET)
num_layers: 10                           # Hidden layers (if not using ROCKET)

# Dataset Options
dataset_name: FaceAll                    # Dataset to use
train_size: 0.8                          # Train/eval split ratio
dataloader_num_workers: 1                # Number of loader threads

# Optimizer Options
batch_size: 128                          # Training batch size
batch_size_testing: 4096                 # Eval/test batch size
learning_rate: 0.001                     # Learning rate
max_epochs: 1000                         # Maximum epochs
patience_lr: 200000                      # LR scheduler patience
patience: 100000                         # Early stopping patience
use_cocob: False                         # Use COCOB optimizer (deprecated)
quantize_adam: True                      # Use quantized Adam
use_dynamictree_quantization: True      # Use dynamic tree quantization
sample_dataset_iid: True                 # Sample data IID vs non-IID
```

---

## Dependencies (requirements.txt)

### Core Libraries:
- **jax/equinox**: JAX-based neural network framework
- **torch**: PyTorch for alternative implementations
- **numba**: JIT compilation for MiniRocket
- **numpy, scipy**: Numerical computing
- **pandas**: Data manipulation
- **matplotlib**: Visualization

### Key Versions:
- equinox==0.11.7
- optax==0.2.3 (JAX optimization)
- torch (via implicit dependency)
- numba==0.59.0
- numpy==1.26.4

---

## Workflow and Execution Flow

### Training Pipeline:

```
1. Load Configuration (test.yaml)
   ↓
2. Load Dataset
   ├─ Read raw UCR/MNIST/CIFAR data
   ├─ Normalize and preprocess
   └─ Apply 8-bit quantization if using ROCKET
   ↓
3. Extract Features (if use_rocket=True)
   ├─ Fit MiniRocket parameters on training data
   ├─ Transform all data splits
   └─ Get 10,000-dimensional features
   ↓
4. Initialize Model
   ├─ Create linear classifier (if ROCKET) OR
   └─ Create multi-layer network (if NN)
   ↓
5. Training Loop (per epoch)
   ├─ Forward pass through batches
   ├─ Compute cross-entropy loss
   ├─ Backward pass (compute gradients)
   ├─ Update parameters with Adam/QuantizedAdam
   ├─ Evaluate on validation set
   ├─ Save accuracies
   └─ Check early stopping
   ↓
6. Save Final Results
   └─ Pickle arrays of test/eval accuracies
```

---

## Key Algorithms and Concepts

### MiniRocket Transform:
1. **Input**: Time series of variable length
2. **Kernels**: 84 fixed 9-point kernels (all 3-combinations from 9 positions)
3. **Dilations**: Log-spaced dilation factors (1, 2, 4, 8, ...)
4. **Convolutions**: Apply kernels at each dilation to input
5. **PPV Features**: Proportion of Positive Values (1 if conv > bias, 0 else)
6. **Output**: Binary features indicating whether convolution exceeded threshold

### Quantization-Aware Training (QAT):
1. **Symmetric Quantization**: Map float values to 8-bit integers
2. **Per-parameter scaling**: Each parameter has own scaling factor
3. **Straight-through estimator**: Gradients pass through quantization unchanged
4. **Dynamic Tree (DT)**: Hierarchical quantization with power-of-2 scaling

### Data Distribution Modes:
- **IID**: Randomly shuffle all training data
- **Non-IID**: Distribute samples across "devices" (simulating federated learning)

---

## Codebase Demerits and Enhancement Opportunities

### 🔴 **Critical Issues**

#### 1. **Deprecated Code and Dead Ends**
- **Issue**: COCOB optimizer is marked "Deprecated" and functionality is incomplete
- **Impact**: Confuses developers; clutters codebase; creates maintenance burden
- **Enhancement**:
  - Remove COCOB completely if not needed, OR
  - Complete implementation with proper testing
  - Decision: Remove (COCOB has limited practical benefit over Adam)

#### 2. **Hardcoded Paths and Device Selection**
- **Issue**: Paths like `~/datasets/{name}` are hardcoded; device selection uses string checks
- **Current Code** (trainer.py):
  ```python
  device = "cuda" if torch.cuda.is_available() else "cpu"
  ```
- **Problems**:
  - Not portable across systems
  - Assumes datasets exist in home directory
  - No configuration for device preference
- **Enhancement**:
  ```python
  # Make configurable via parameters
  def get_device(device_config: str):
      if device_config == "auto":
          return "cuda" if torch.cuda.is_available() else "cpu"
      elif device_config == "cpu":
          return "cpu"
      elif "cuda" in device_config:
          return device_config
      else:
          raise ValueError(f"Unknown device: {device_config}")
  ```

#### 3. **Magic Numbers Throughout Code**
- **Examples**:
  - 84 kernels (hardcoded in MiniRocket)
  - 9-point kernel size (implicit in algorithm)
  - 2^11 validation size (softmax.py)
  - 256-value quantization table
- **Enhancement**: Define as class constants:
  ```python
  class MiniRocketConfig:
      NUM_KERNELS = 84
      KERNEL_SIZE = 9
      QUANTILE_DIVISIONS = 84
  ```

#### 4. **No Input Validation or Error Handling**
- **Issue**: Functions assume valid inputs; minimal error checking
- **Example** (jax_training.py):
  ```python
  self.__key, subkey = jax.random.split(self.__key)
  return subkey
  # No check if __key is initialized
  ```
- **Enhancement**: Add assertions and typed inputs
  ```python
  def next_key(self) -> jax.random.PRNGKey:
      if self.__key is None:
          raise RuntimeError("Key not initialized")
      self.__key, subkey = jax.random.split(self.__key)
      return subkey
  ```

---

### 🟡 **Major Design Issues**

#### 5. **Dual Implementation (PyTorch and JAX) Without Abstraction**
- **Issue**: Two separate Trainer classes with nearly identical logic
- **Current**: `trainer.py` (PyTorch) vs `jax_training.py` (JAX)
- **Problems**:
  - Code duplication (~80% identical)
  - Bug fixes must be made twice
  - Inconsistent parameter handling
  - Maintenance nightmare
- **Enhancement**: Create abstract base class:
  ```python
  class TrainerBase(ABC):
      @abstractmethod
      def run_epoch(self): pass
      
      @abstractmethod
      def eval(self, dataloader): pass
      
      def run(self):
          # Common training loop
          for epoch in range(self.max_epochs):
              self.run_epoch()
              self.eval(self.eval_dl)
  
  class PyTorchTrainer(TrainerBase):
      # PyTorch-specific implementation
  
  class JAXTrainer(TrainerBase):
      # JAX-specific implementation
  ```

#### 6. **Inconsistent Logging and Monitoring**
- **Issue**: Results saved to pickle files; no structured logging
- **Problems**:
  - Difficult to compare runs
  - No metadata about hyperparameters in results
  - No intermediate logging during training
- **Enhancement**: Use standard logging framework
  ```python
  import logging
  import json
  
  logger = logging.getLogger(__name__)
  
  # Log training progress
  logger.info(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")
  
  # Save metadata with results
  metadata = {
      'dataset': params['dataset_name'],
      'timestamp': datetime.now().isoformat(),
      'hyperparameters': params,
      'accuracies': accuracies,
  }
  with open(f"{save_path}/metadata.json", 'w') as f:
      json.dump(metadata, f)
  ```

#### 7. **Poor Separation of Concerns**
- **Issue**: Dataset loading, feature extraction, and training mixed together
- **Example** (data.py): `ClassificationDataset` does too much:
  - Loads raw data
  - Applies normalization
  - Applies quantization
  - Applies feature transformation
  - Splits into train/eval/test
  - Creates PyTorch datasets
- **Enhancement**: Break into pipeline stages:
  ```python
  class DataLoader:
      def load(self) -> np.ndarray
  
  class DataPreprocessor:
      def normalize(self) -> np.ndarray
      def quantize(self) -> np.ndarray
  
  class FeatureExtractor:
      def transform(self) -> np.ndarray
  
  class DataSplitter:
      def split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
  
  # Compose them
  pipeline = Pipeline([
      DataLoader(dataset_name),
      DataPreprocessor(),
      FeatureExtractor(),
      DataSplitter(),
  ])
  ```

#### 8. **Missing Type Hints**
- **Issue**: No type annotations in any file
- **Problems**:
  - IDE autocomplete doesn't work
  - Silent type errors
  - Harder to understand function signatures
- **Enhancement**: Add comprehensive type hints
  ```python
  from typing import Tuple, Optional, Dict, Any
  import numpy as np
  from jax import Array
  
  def load_data(
      dataset_name: str,
      seed: int,
      test: bool = False
  ) -> Tuple[np.ndarray, np.ndarray]:
      """Load UCR time series dataset.
      
      Args:
          dataset_name: Name of dataset (e.g., 'ElectricDevices')
          seed: Random seed for reproducibility
          test: If True, load test set; else training set
          
      Returns:
          Tuple of (features, labels) as numpy arrays
      """
      ...
  ```

#### 9. **Hardcoded Configuration Values**
- **Issue**: Important hyperparameters hardcoded in functions
- **Examples**:
  - `max_dilations_per_kernel = 32` in MiniRocket
  - `10_000` features hardcoded
  - Quantization bits hardcoded to 8
- **Enhancement**: Pass as parameters or use configuration:
  ```python
  @dataclass
  class MiniRocketConfig:
      num_features: int = 10_000
      max_dilations_per_kernel: int = 32
      quantization_bits: int = 8
      kernel_size: int = 9
      num_kernels: int = 84
  
  def fit(X: np.ndarray, config: MiniRocketConfig) -> Parameters:
      ...
  ```

#### 10. **No Testing Infrastructure**
- **Issue**: No unit tests, integration tests, or validation scripts
- **Problems**:
  - Bugs go undetected
  - Refactoring is risky
  - No CI/CD pipeline possible
  - Reproducibility not guaranteed
- **Enhancement**: Add pytest framework
  ```python
  # tests/test_minirocket.py
  import pytest
  from rocket.minirocket import fit, transform
  
  def test_fit_transform_shapes():
      X = np.random.randn(100, 500).astype(np.float32)
      params = fit(X)
      X_transformed = transform(X, params)
      assert X_transformed.shape == (100, 10000)
  
  def test_deterministic_output():
      X = np.random.randn(50, 300).astype(np.float32)
      X_t1 = transform(X, params)
      X_t2 = transform(X, params)
      np.testing.assert_array_equal(X_t1, X_t2)
  ```

---

### 🟠 **Performance Issues**

#### 11. **Inefficient Data Loading**
- **Issue**: Data loaded from disk on every epoch
- **Current** (softmax.py):
  ```python
  file = pd.read_csv(path, ..., chunksize=chunk_size, engine="c")
  for chunk in file:
      # Process chunk
  ```
- **Problem**: Disk I/O is bottleneck, especially with network filesystems
- **Enhancement**: Pre-load into memory with memory mapping
  ```python
  # For large datasets, use memory mapping
  data = np.load("data.npy", mmap_mode='r')
  
  # For small datasets, load once
  data = np.load("data.npy")
  ```

#### 12. **No Batch Processing Optimization**
- **Issue**: Batches processed sequentially; no support for distributed training
- **Problem**: Can't scale to larger models or datasets
- **Enhancement**: Add distributed training support
  ```python
  # Use PyTorch DistributedDataParallel
  model = DistributedDataParallel(model)
  
  # Or use JAX pmap for multi-GPU
  @jax.pmap
  def step_batch(model, batch, optimizer):
      ...
  ```

#### 13. **Unnecessary Data Conversions**
- **Issue**: Data converted between numpy, torch, jax multiple times
- **Example** (data.py → jax_training.py):
  ```python
  # data.py loads as numpy
  X_train = np.array(...)
  
  # jax_training.py converts
  X_transform = jnp.array(batch["input"])
  ```
- **Enhancement**: Lazy conversion using unified array type
  ```python
  import pyarrow as pa
  
  # Store in Arrow format, convert on-demand
  X_train = pa.array(...)
  ```

---

### 🟣 **API and Usability Issues**

#### 14. **Inconsistent Result Saving**
- **Issue**: Different saving mechanisms in different places
- **Problems**:
  - softmax.py uses pickle with protocol
  - jax_training.py uses pickle differently
  - No metadata saved with results
  - Filenames are cryptic
- **Enhancement**: Unified results manager
  ```python
  class ResultsManager:
      def save_run(self,
                   dataset: str,
                   seed: int,
                   config: Dict,
                   metrics: Dict) -> None:
          """Save training run with full metadata."""
          run_id = f"{dataset}_{seed}_{hash(config)}"
          run_dir = self.results_path / run_id
          run_dir.mkdir(exist_ok=True)
          
          with open(run_dir / "config.json", 'w') as f:
              json.dump(config, f)
          with open(run_dir / "metrics.json", 'w') as f:
              json.dump(metrics, f)
  ```

#### 15. **Unclear Function Purposes**
- **Issue**: Many functions lack docstrings; purposes unclear
- **Examples**:
  - What does `quantize_8_bit` actually do? (quantizes to [-127, 127])
  - What's the relationship between `use_rocket` and feature dimensionality?
  - Why does IID sampling require rebatching?
- **Enhancement**: Add comprehensive docstrings with examples
  ```python
  def quantize_8_bit(data: np.ndarray,
                     offset: float,
                     scaling: float) -> np.ndarray:
      """Quantize continuous values to 8-bit signed integers.
      
      This function scales data relative to an offset, divides by a scaling
      factor, and clips to the range [-127, 127] for 8-bit representation.
      
      Args:
          data: Input array of float values
          offset: Mean/center point for quantization
          scaling: Scale factor (typically 99.9 percentile)
          
      Returns:
          Array of quantized int8 values
          
      Example:
          >>> x = np.array([1.0, 2.5, -0.5])
          >>> quantize_8_bit(x, offset=1.0, scaling=0.5)
          array([  0, 120, -75], dtype=int8)
      """
      return np.clip((data - offset) / scaling * 127, -127, 127)
  ```

---

### 🔵 **Minor Issues**

#### 16. **Inconsistent Naming Conventions**
- **Examples**:
  - `X_train_transform` vs `X_training`
  - `sample_dataset_iid` vs `use_rocket` (inconsistent boolean naming)
  - `use_dynamictree_quantization` vs `use_dynamic_tree_quantization` (typo)
- **Enhancement**: Establish naming style guide
  ```
  # Boolean variables use "use_X" or "is_X" prefix
  use_rocket = True
  is_training = False
  
  # Private variables use "__" prefix
  self.__model = None
  
  # Constants use UPPER_CASE
  MAX_EPOCHS = 1000
  BATCH_SIZE = 128
  ```

#### 17. **Unused Imports and Variables**
- **Examples** (compare_rocket_nn.py):
  - `from pathlib import Path` - imported but only used once
  - `boundary` variable computed but not fully used
  - Multiple unused comparison plots
- **Enhancement**: Run linting tools
  ```bash
  pylint *.py  # or
  flake8 *.py --select=F401,F841
  ```

#### 18. **No Configuration Validation**
- **Issue**: Invalid hyperparameters silently accepted
- **Enhancement**: Validate config on load
  ```python
  def validate_config(params: Dict) -> None:
      """Validate hyperparameters."""
      assert 0 < params['learning_rate'] < 1, "Learning rate out of range"
      assert params['batch_size'] > 0, "Batch size must be positive"
      assert 0 < params['train_size'] < 1, "train_size must be in (0, 1)"
      if params['use_rocket']:
          assert params['num_features'] > 0
      else:
          assert params['num_layers'] > 0
          assert params['num_neurons'] > 0
  ```

#### 19. **Hardcoded Dataset Splits**
- **Issue**: Train/eval/test split is hardcoded (train=80%, test=200 samples)
- **Enhancement**: Make configurable
  ```yaml
  dataset_splits:
      train_ratio: 0.8
      eval_ratio: 0.1
      test_ratio: 0.1
      # OR
      test_samples: 200  # Override with explicit count
  ```

#### 20. **No Experiment Tracking**
- **Issue**: No way to compare experiments systematically
- **Enhancement**: Add experiment tracking
  ```python
  import wandb  # or mlflow
  
  wandb.init(project="time-series-classification")
  wandb.config.update(params)
  
  for epoch in range(max_epochs):
      wandb.log({
          'loss': loss,
          'accuracy': acc,
          'epoch': epoch,
      })
  ```

---

## Summary of Enhancement Priorities

| Priority | Issue | Effort | Impact | Quick Fix? |
|----------|-------|--------|--------|-----------|
| 🔴 Critical | Deprecated COCOB | 1h | Medium | Remove code |
| 🔴 Critical | Hardcoded paths | 2h | High | Config file |
| 🔴 Critical | No input validation | 3h | High | Add assertions |
| 🟡 Major | Code duplication (PyTorch/JAX) | 4h | Very High | Refactor with ABC |
| 🟡 Major | Inconsistent logging | 3h | High | Use logging module |
| 🟡 Major | Poor separation of concerns | 6h | Very High | Create pipeline stages |
| 🟠 Performance | Inefficient data loading | 2h | Medium | Memory mapping |
| 🟣 Usability | Missing type hints | 5h | High | Add typing |
| 🔵 Minor | Inconsistent naming | 3h | Medium | Establish guide + linting |
| 🔵 Minor | No unit tests | 8h | Very High | Add pytest suite |

---

## Recommendations

1. **Immediate Actions** (This week):
   - Remove deprecated COCOB code
   - Move hardcoded paths to config
   - Add type hints to critical functions

2. **Short-term** (This month):
   - Refactor PyTorch/JAX duplication
   - Add comprehensive logging
   - Create unit test suite
   - Add input validation

3. **Long-term** (This quarter):
   - Implement distributed training support
   - Add experiment tracking (wandb/mlflow)
   - Create visualization dashboard
   - Benchmark performance improvements

---

## Conclusion

This codebase implements a sophisticated time series classification pipeline combining MiniRocket feature extraction with various optimization strategies. While the core algorithms are sound, the implementation suffers from typical research code issues: duplication, hardcoded values, poor error handling, and lack of proper testing.

The **highest-impact improvements** would be:
1. **Refactor PyTorch/JAX duplication** (fixes maintenance burden)
2. **Add unit tests** (enables safe refactoring)
3. **Implement proper logging** (enables systematic comparison)
4. **Add type hints** (improves developer experience)

These changes would transform the codebase from research-grade to production-ready while maintaining scientific rigor.
