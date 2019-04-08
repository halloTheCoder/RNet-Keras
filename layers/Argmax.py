import numpy as np
from keras import backend as K
from keras.engine import Layer, InputSpec

class Argmax(Layer):
    """
    Module responsible for taking Argmax of output tensors
    # Arguments
        axis : axis along with Argmax is taken
    """
    def __init__(self, axis=-1, **kwargs):
        super(Argmax, self).__init__(**kwargs)
        
        #supports_masking: Boolean indicator of whether the layer 
        # supports masking, typically for unused timesteps in a sequence.
        self.supports_masking = True  
        
        self.axis = axis

    def call(self, inputs, mask=None):
        """This is where the layer's logic lives.
        # Arguments
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.
        # Returns
            A tensor or list/tuple of tensors.
        """
        return K.argmax(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.
        # Arguments
            input_shape: Shape tuple (tuple of integers)
                or list of shape tuples (one per output tensor of the layer).
                Shape tuples can include None for free dimensions,
                instead of an integer.
        # Returns
            An output shape tuple i.e. input_shape dim reduced along axis.
        """
        input_shape = list(input_shape)
        del input_shape[self.axis]
        return tuple(input_shape)

    def compute_mask(self, x, mask):
        """Computes an output mask tensor.
        # Arguments
            inputs: Tensor or list of tensors.
            mask: Tensor or list of tensors.
        # Returns
            None or a tensor (or list of tensors,
            one per output tensor of the layer).
        """
        return None

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
        config = {'axis': self.axis}
        base_config = super(Argmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
