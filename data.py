import random
import itertools
import pickle

import numpy as np

import keras
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

import keras.backend as K

def load_dataset(filename):
    """
    Utility function to load data from pickled file
    # Arguments
        filename : datafile to be unpickled
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def padded_batch_input(inputs, indices=None, dtype=K.floatx(), maxlen=None):
    """
    Utility function to pad [??kya pad] inputs to same length
    # Arguments
        inputs : .
        indices : denotes indices of inputs tensor which are need to be padded
        dtype : data type of padded tensors
        maxlen : length to which tensors are to be padded
    """
    if indices is None:
        indices = np.arange(len(inputs))
    
    batch_input = [inputs[i] for i in indices]
    return pad_sequences(batch_input, maxlen, dtype, padding='post')

def categorical_batch_target(target, classes, indices=None, dtype=K.floatx()):
    """
    Utility function to pad [??kya pad] inputs to same length
    # Arguments
        target : 
        classes : 
        indices : denotes indices of inputs tensor which are need to be padded
        dtype : data type of padded tensors
    """
    if indices is None:
        indices = np.arange(len(target))
        
    batch_target = [min(target[i], classes-1) for i in indices]
    return to_categorical(batch_target, classes).astype(dtype)

def lengthGroup(length):
    if length < 150:
        return 0
    if length < 240:
        return 1
    if length < 380:
        return 2
    if length < 520:
        return 3
    if length < 660:
        return 4
    return 5

class BatchGen:
    """
    Class to load data in batches
    # Arguments
        inputs : 
        targets : 
        batch_size : the batch size 
        shuffle : Bool, whether to shuffle inputs and targets tensor
        dtype : data type of tensors
        flatten_target : Bool, whether to flatten the targets tensor
        sort_by_length :  Bool, whether to sort the inputs and targets tensor by length of inputs tensor
        group : to group the inputs tensor and targets tensor on the basis of inputs tensor length 
    """
    def __init__(self, 
                 inputs, #what is dim of inputs
                 targets=None, 
                 batch_size=None, 
                 shuffle=True,
                 balance=False, 
                 dtype=K.floatx(),
                 flatten_targets=False,
                 sort_by_length=False,
                 group=False,
                 maxlen=None
                ):
        assert len(set([len(i) for i in inputs])) == 1
        assert(not shuffle or not sort_by_length)  #Both cannot be true
        
        self.inputs = inputs
        self.nb_samples = len(inputs[0])  ##??inputs.shape, target.shape
        
        self.batch_size = batch_size if batch_size else self.nb_samples
        
        self.dtype = dtype
        self.shuffle = shuffle
        self.balance = balance
        self.targets = targets
        self.flatten_targets = flatten_targets
        
        if isinstance(maxlen, (list, tuple)):
            self.maxlen = maxlen
        else:
            self.maxlen = [maxlen] * len(inputs)  ##done to use later for zip
            
        self.sort_by_length = None
        if sort_by_length:
            self.sort_by_length = np.argsort([-len(p) for p in inputs[0]])  ##sorting by lengths of input
        
        self.generator = self._generator()
        self._steps = -(-self.nb_samples // self.batch_size)  ##round up rather than down
        
        self.groups = None
        if group:                                         ##grouping inputs by their length
            indices = np.arange(self.nb_samples)
            ff = lambda i : lengthGroup(len(inputs[0][i]))
            indices = np.argsort([ff(i) for i in indices])
            self.groups = itertools.groupby(indices, ff)
            self.groups = {k: np.array(list(v)) for k, v in self.groups}
    
    def _generator(self):
        while True:
            if self.shuffle:
                permutation = np.random.permutation(self.nb_samples)
            elif self.sort_by_length is not None:
                permutation = self.sort_by_length
            elif self.groups is not None:
                for k, v in self.groups.items():
                    np.random.shuffle(v)
                
                tmp = np.concatenate(list(self.groups.values()))
                batches = np.array_split(tmp, self._steps)
                
                reminader = []
                if len(batches[-1]) < self._steps:
                    remainder = batches[-1:]
                    batches = batches[:-1]
                
                random.shuffle(batches)
                batches += remainder
                permutation = np.concatenate(batches)
            
            else:
                permutation = np.arange(self.nb_samples)
            
            i = 0
            longest = 767
            
            while i < self.nb_samples:
                if self.sort_by_length is not None:
                    bs = self.batch_size*longest//self.inputs[0][permutation[i]].shape[0]
                else:
                    bs = self.batch_size
                    
                indices = permutation[i:i+bs]
                i = i+bs
                
                batch_X = [padded_batch_input(x, indices, self.dtype, maxlen)
                           for x, maxlen in zip(self.inputs, self.maxlen)]
                P = batch_X[0].shape[1]
                
                if not self.targets:
                    yield batch_X
                    continue
                
                batch_Y = [categorical_batch_target(target, P, indices, self.dtype)
                           for target in self.targets]
                
                if self.flatten_targets:
                    batch_Y = np.concatenate(batch_Y, axis=-1)
                    
                if not self.balance:
                    yield (batch_X, batch_Y)
                    continue
                
                batch_W = np.array([bs/self.batch_size for x in batch_Y[0]]).astype(self.dtype)
                yield (batch_X, batch_Y, batch_W)
        
    def __iter__(self):
        return self.generator
    
    def next(self):
        return self.generator.next()
    
    def __next__(self):
        return self.generator.__next__()
    
    def steps(self):
        if self.sort_by_length is None: 
            return self._steps

        print("Steps was called")
        if self.shuffle:
            permutation = np.random.permutation(self.nb_samples)
        elif self.sort_by_length is not None:
            permutation = self.sort_by_length
        else:
            permutation = np.arange(self.nb_samples)

        i = 0
        longest = 767

        self._steps = 0
        while i < self.nb_samples:
            bs = self.batch_size * 767 // self.inputs[0][permutation[i]].shape[0]
            i = i + bs
            self._steps += 1

        return self._steps


batch_gen = BatchGen      ##for backward compatibility
