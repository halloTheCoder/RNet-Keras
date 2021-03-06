from keras import backend as K
from keras.layers import Layer
from keras.layers.wrappers import TimeDistributed

from .helpers import softmax

class QuestionPooling(Layer):
    """
    Class implementing simple attention-outputs of input tensors.
    # Arguments
        **kwargs: Additional keyword arguments
    """
    def __init__(self, **kwargs):
        super(QuestionPooling, self).__init__(**kwargs)
        self.supports_masking = True

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
        assert(isinstance(input_shape, list) and len(input_shape) == 5)

        input_shape = input_shape[0]
        B, Q, H = input_shape
        
        return (B, H)

    def build(self, input_shape):
        """Creates the layer weights.
        Must be implemented on all layers that have weights.
        # Arguments
            input_shape: Keras tensor (future input to layer)
                or list/tuple of Keras tensors to reference
                for weight shape computations.
        """
        assert(isinstance(input_shape, list) and len(input_shape) == 5)
        input_shape = input_shape[0]
        
        B, Q, H_ = input_shape
        H = H_ // 2

    def call(self, inputs, mask=None):
        """This is where the layer's logic lives.
        # Arguments
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.
        # Returns
            A tensor or list/tuple of tensors.
        """
        assert(isinstance(inputs, list) and len(inputs) == 5)
        uQ, WQ_u, WQ_v, v, VQ_r = inputs
        uQ_mask = mask[0] if mask is not None else None

        ones = K.ones_like(K.sum(uQ, axis=1, keepdims=True)) # (B, 1, 2H)
        s_hat = K.dot(uQ, WQ_u)
        s_hat += K.dot(ones, K.dot(WQ_v, VQ_r))
        s_hat = K.tanh(s_hat)
        s = K.dot(s_hat, v)
        s = K.batch_flatten(s)

        a = softmax(s, mask=uQ_mask, axis=1)

        rQ = K.batch_dot(uQ, a, axes=[1, 1])

        return rQ

    def compute_mask(self, input, mask=None):
        """Computes an output mask tensor.
        # Arguments
            inputs: Tensor or list of tensors.
            mask: Tensor or list of tensors.
        # Returns
            None or a tensor (or list of tensors,
            one per output tensor of the layer).
        """
        return None
