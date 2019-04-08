from keras import backend as K
from keras.engine.topology import Layer

class VariationalDropout(Layer):
    """
    Module implementing Variational Dropout [https://arxiv.org/abs/1506.02557],
    generalization of Gaussian dropout, where the dropout rates are learned, often leading to better models.
    # Arguments
        rate : max dropout rate
        noise_shape : shape of noise to be introduced
        seed : seed of random generator
    """
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(VariationalDropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def call(self, inputs, training=None):
        """This is where the layer's logic lives.
        # Arguments
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.
        # Returns
            A tensor or list/tuple of tensors.
        """
        if 0. < self.rate < 1.:
            symbolic_shape = K.shape(inputs)
            noise_shape = [shape if shape is not None and shape > 0 else symbolic_shape[axis]
                           for axis, shape in enumerate(self.noise_shape)]
            noise_shape = tuple(noise_shape)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape, seed=self.seed)

            return K.in_train_phase(dropped_inputs, inputs, training=training)

        return inputs

    def get_config(self):
        """Returns the config of the layer.
        A layer config is a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.
        The config of a layer does not include connectivity
        information, nor the layer class name. These are handled
        by `Network` (one layer of abstraction above).
        # Returns
            Python dictionary.
        """
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape,
                  'seed': self.seed}
        base_config = super(VariationalDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
