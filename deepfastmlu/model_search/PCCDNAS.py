from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2


class PCCDNAS:
    """
    Principal Component Cascade Dense Neural Architecture Search (PCCDNAS).

    PCCDNAS provides an automated method for designing neural networks. Using PCA
    (Principal Component Analysis), it systematically sets the number of neurons
    in each layer of the network. After applying PCA to the initial data, the
    neuron count for the first layer is determined based on the principal components
    (PCs) for a given variance threshold. Subsequently, the cascade mechanism ensures
    that the activations from each trained layer undergo PCA again. This process,
    in turn, determines the neuron count for the subsequent layers using the same
    principal component variance threshold criteria.
    """

    def __init__(self):
        """
        Initializes an empty Sequential model, an empty list for layer neurons,
        and a scaler for data normalization.
        """
        self._model = Sequential()
        self._num_neurons = []
        self._scaler = None

    def data_init(self, X_train, y_train, validation=None, normalize=True, unit=False):
        """
        Initialize data for searching the model.

        Args:
            X_train (ndarray): Training data.
            y_train (ndarray): Training labels/targets.
            validation (float, tuple, optional): Validation data. Can be a percentage (float) or a tuple (X_val, y_val).
            normalize (bool, optional): Whether to normalize the data or not. Default is True.
            unit (bool, optional): If True, standard deviation is used for normalization. Default is False.
        """
        self._X_train = X_train
        self._y_train = y_train
        self._validation = validation
        self._normalize = normalize
        self._unit = unit

        self._process_validation_data()
        if self._normalize:
            self._normalize_data()

    def initialize_model_search(self, **kwargs):
        """
        Initialize the model's hyperparameters based on provided keyword arguments.

        Note:
            All parameters are passed as keyword arguments.

        Args:
            epochs (int, optional): Number of training epochs. Default is 15.
            layers (int, optional): Number of layers in the model. Default is 2.
            activation (str, optional): Activation function for the layers. Default is 'elu'.
            pca_variance (float or list of float, optional): Desired explained variance for PCA. Default is 0.95.
            loss (str, optional): Loss function for the model. Default is 'categorical_crossentropy'.
            optimizer (str, optional): Optimizer for the model. Default is 'adam'.
            metrics (list, optional): List of metrics to be evaluated. Default is ['accuracy'].
            output_neurons (int, optional): Number of neurons in the output layer. Default is 1.
            out_activation (str, optional): Activation function for the output layer. Default is 'sigmoid'.
            stop_criteria (str, optional): Criteria for early stopping. Default is 'val_loss'.
            es_mode (str, optional): Mode for early stopping. Default is 'max'.
            dropout (float, optional): Dropout rate for dropout layers (includes). Default is None.
            regularize (tuple (str, float), optional): Regularization type and value. Default is None.
            batch_size (int, optional): Batch size for training. Default is 32.
            kernel_initializer (str, optional): Kernel initializer for the dense layers. Default is 'he_normal'.
            batch_norm (bool, optional): Whether to include batch normalization layers. Default is True.
            es_patience (int, optional): Number of epochs with no improvement for early stopping. Default is 5.
            verbose (int, optional): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Default is 1.
            custom_callback (callback, optional): Custom callback for training. Default is None.
            learn_rate (float, optional): Learning rate for the optimizer. Default is 0.0001.
        """

        # Default values
        defaults = {
            'epochs': 15,
            'layers': 2,
            'activation': 'elu',
            'pca_variance': 0.95,
            'loss': 'categorical_crossentropy',
            'optimizer': 'adam',
            'metrics': ['accuracy'],
            'output_neurons': 1,
            'out_activation': 'sigmoid',
            'stop_criteria': 'val_loss',
            'es_mode': 'max',
            'dropout': None,
            'regularize': None,
            'batch_size': 32,
            'kernel_initializer': 'he_normal',
            'batch_norm': True,
            'es_patience': 5,
            'verbose': 1,
            'custom_callback': None,
            'learn_rate': 0.0001
        }

        for key, default_value in defaults.items():
            setattr(self, f"_{key}", kwargs.get(key, default_value))

        self._process_pca_variance()

    def _process_pca_variance(self):
        """
        Processes PCA variance to ensure it's valid. Either a single number
        or a list of length equal to the number of layers is expected.
        """
        if isinstance(self._pca_variance, (int, float)):
            self._variance_list = [self._pca_variance] * self._layers
        elif isinstance(self._pca_variance, list) and len(self._pca_variance) == self._layers:
            self._variance_list = self._pca_variance
        else:
            raise ValueError(
                'PCA variance must be a single number or a list of length equal to the number of layers.')

    def _process_validation_data(self):
        """
        Processes validation data based on its type (float or tuple).
        """
        if isinstance(self._validation, float):
            self._validation_split = self._validation
            self._validation_data = None
        elif isinstance(self._validation, tuple) and len(self._validation) == 2:
            self._validation_data = self._validation
            self._validation_split = None
        else:
            raise ValueError("Validation must be either float or a tuple in form (X_val, y_val)")

    def _normalize_data(self):
        """
        Normalize the training data using the Standard Scaler.
        If validation data is present, it is also transformed (not fitted).
        """
        self._scaler = StandardScaler(with_std=self._unit)
        self._X_train = self._scaler.fit_transform(self._X_train)
        if self._validation_data:
            self._validation_data = (
                self._scaler.transform(self._validation_data[0]), self._validation_data[1]
            )

    def _create_callback(self):
        """
        Creates an early stopping callback for the Keras model based on the
        specified criteria.

        Returns:
            EarlyStopping: Configured early stopping callback.
        """
        return EarlyStopping(
            monitor=self._stop_criteria, mode=self._es_mode, verbose=self._verbose,
            patience=self._es_patience, restore_best_weights=True
        )

    def _perform_pca(self, data, variance):
        """
        Performs PCA on the provided data to determine the number of components
        that explain the given variance.

        Args:
            data (ndarray): Input data for PCA.
            variance (float): Desired explained variance for PCA.

        Returns:
            int: Number of PCA components.
        """
        pca = PCA(variance)
        pca.fit(data)
        return pca.n_components_

    def _build_layer(self, layer_idx):
        """
        Builds an individual layer of the neural network based on the PCA result.

        Args:
            layer_idx (int): Index of the layer to be built.

        Returns:
            int: Number of PCA components for the current layer unit count.
        """
        regularizer = self._get_regularizer()

        if layer_idx == 0:
            n_pca_components = self._perform_pca(self._X_train, self._variance_list[layer_idx])
        else:
            self._append_and_train_previous_layer(regularizer)
            layer_output_model = Model(inputs=self._model.input, outputs=self._model.layers[-1].output)
            X_train_transformed = layer_output_model.predict(self._X_train)
            n_pca_components = self._perform_pca(X_train_transformed, self._variance_list[layer_idx])

        self._num_neurons.append(n_pca_components)
        return n_pca_components

    def _get_regularizer(self):
        """
        Returns the appropriate regularizer (L1 or L2) based on the provided type.

        Returns:
            Regularizer: Configured regularizer.
        """
        if self._regularize:
            return l1(self._regularize[1]) if self._regularize[0].lower() == 'l1' else l2(self._regularize[1])

    def _append_and_train_previous_layer(self, regularizer):
        """
        Appends necessary layers to the model, trains it, and pops the last layer.
        This method prepares the model for the next layer of neurons.

        Args:
            regularizer (Regularizer): Regularizer to be applied to the layer.
        """
        self._append_batch_norm_if_needed()
        self._append_dense_dropout_compile(regularizer)
        self._train_model()
        self._model.pop()

    def _append_batch_norm_if_needed(self):
        """
        Adds a batch normalization layer to the model if specified in the hyperparameters.
        """
        if self._batch_norm:
            self._model.add(BatchNormalization())

    def _append_dense_dropout_compile(self, regularizer):
        """
        Adds dense and dropout layers to the model and then compiles it.

        Args:
            regularizer (Regularizer): Regularizer to be applied to the layer.
        """
        self._model.add(Dense(
            self._output_neurons, activation=self._out_activation,
            kernel_initializer=self._kernel_initializer, kernel_regularizer=regularizer
        ))
        if self._dropout:
            self._model.add(Dropout(self._dropout))
        self._model.compile(loss=self._loss, optimizer=Adam(learning_rate=self._learn_rate), metrics=self._metrics)

    def _train_model(self):
        """
        Trains the neural network model on the provided data and hyperparameters.
        """
        history = self._model.fit(
            self._X_train, self._y_train, epochs=self._epochs, verbose=self._verbose,
            validation_split=self._validation_split, validation_data=self._validation_data,
            batch_size=self._batch_size, callbacks=[self._create_callback()]
        )
        self._print_layer_info(history, len(self._num_neurons))

    def _print_layer_info(self, history, layer_idx):
        """
        Prints the information about the trained layer, including the number of neurons
        and the best validation metric.

        Args:
            history (History): Training history from Keras model training.
            layer_idx (int or str): Index or descriptor of the layer.
        """
        best_val_stop = min(history.history[self._stop_criteria]) if self._es_mode == 'min' else max(
            history.history[self._stop_criteria])
        print(f'Layer {layer_idx}: {self._num_neurons[-1]} neurons, best {self._stop_criteria}: {best_val_stop}')

    def _reset_model(self):
        """
        Resets the neural network model and the list of neurons.
        """
        self._model = Sequential()
        self._num_neurons = []

    def build(self):
        """
        Builds and trains the complete neural network model based on the PCA specification
        and the specified hyperparameters.

        Returns:
            tuple: A tuple containing the trained Keras model and a list of the number of neurons
            for each layer.
        """
        self._reset_model()
        regularizer = self._get_regularizer()

        for layer_idx in range(self._layers):
            n_pca_components = self._build_layer(layer_idx)
            self._append_batch_norm_if_needed()
            self._model.add(Dense(
                n_pca_components, input_dim=self._X_train.shape[1], use_bias=False,
                activation=self._activation, kernel_initializer=self._kernel_initializer,
                kernel_regularizer=regularizer
            ))
            if self._dropout:
                self._model.add(Dropout(self._dropout))

        self._model.add(Dense(
            self._output_neurons, activation=self._out_activation,
            kernel_regularizer=regularizer, use_bias=False
        ))
        self._model.compile(loss=self._loss, optimizer=Adam(learning_rate=self._learn_rate), metrics=self._metrics)

        callbacks = [self._create_callback()]
        if self._custom_callback:
            callbacks.append(self._custom_callback)

        history = self._model.fit(
            self._X_train, self._y_train, epochs=self._epochs, verbose=self._verbose,
            validation_split=self._validation_split, validation_data=self._validation_data,
            batch_size=self._batch_size, callbacks=callbacks
        )
        self._print_layer_info(history, 'Final')
        return self._model, self._num_neurons