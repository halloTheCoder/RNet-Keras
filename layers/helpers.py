from keras import backend as K

#Implementation of custom softmax function which supports masking
#(More about in doc)
def softmax(x, axis, mask=None):
    if mask is None:
        mask = K.constant(True)
    mask = K.cast(mask, K.floatx())
    if K.ndim(x) is K.ndim(mask) + 1:
        mask = K.expand_dims(mask)

    m = K.max(x, axis=axis, keepdims=True)
    e = K.exp(x - m) * mask
    s = K.sum(e, axis=axis, keepdims=True)
    s += K.cast(K.cast(s < K.epsilon(), K.floatx()) * K.epsilon(), K.floatx())
    return e / s
    