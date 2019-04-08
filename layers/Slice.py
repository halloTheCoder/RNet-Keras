import numpy as np
from keras import backend as K
from keras.engine import Layer, InputSpec

class Slice(Layer):
    """
    Module responsible for slicing and returning the input tensor at given indices.
    It also supports Masking.
    # Arguments
        indices : indices of the input tensor that needs to be returned
        axis : axis along which indices are considered
    """
    def __init__(self, indices, axis=1, **kwargs):
        self.supports_masking = True
        self.axis = axis
        
        if isinstance(indices, slice):
            self.indices = (indices.start, indices.stop, indices.step)
        else:
            self.indices = indices

        self.slices = [ slice(None) ] * self.axis

        if isinstance(self.indices, int):
            self.slices.append(self.indices)
        elif isinstance(self.indices, (list, tuple)):
            self.slices.append(slice(*self.indices))
        else:
            raise TypeError("indices must be int or slice")
        
        super(Slice, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        """This is where the layer's logic lives.
        # Arguments
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.
        # Returns
            A tensor or list/tuple of tensors.
        """
        return inputs[self.slices]

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
        for i, slice in enumerate(self.slices):
            if i == self.axis:
                continue
            start = slice.start or 0
            stop = slice.stop or input_shape[i]
            step = slice.step or 1
            input_shape[i] = None if stop is None else (stop - start) // step
        del input_shape[self.axis]

        return tuple(input_shape)

    def compute_mask(self, x, mask=None):
        """Computes an output mask tensor.
        # Arguments
            inputs: Tensor or list of tensors.
            mask: Tensor or list of tensors.
        # Returns
            None or a tensor (or list of tensors,
            one per output tensor of the layer).
        """
        if mask is None:
            return mask
        if self.axis == 1:
            return mask[self.slices]
        else:
            return mask

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
        config = {'axis': self.axis,
                  'indices': self.indices}
        base_config = super(Slice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
