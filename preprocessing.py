import os
import json
import pickle
import argparse

from unidecode import unidecode
from tqdm import tqdm

import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import get_file

from nltk.tokenize import word_tokenize, TreebankWordTokenizer

def get_glove_file_path():
    """
    Utility function to get GLOVE vector file path after downloading it.
    """
    SERVER = 'http://nlp.stanford.edu/data/'
    VERSION = 'glove.840B.300d'

    origin = '{server}{version}.zip'.format(server=SERVER, version=VERSION)
    cache_dir = os.path.join(os.path.abspath(os.path.dirname('__file__')), 'data')

    fname = '/tmp/glove.zip'
    get_file(fname,
             origin=origin,
             cache_dir=cache_dir,
             cache_subdir='',
             extract=True)

    # Remove unnecessary .zip file and keep only extracted .txt version
    os.remove(fname)
    return os.path.join(cache_dir, VERSION) + '.txt'

def word_tokenizer():
    """
    Utility function that returns the function for tokenizing sentence 
    into words and char_offsets of tokens in that sentence.
    # Arguments
        filename : datafile to be unpickled.
    """
    def tokenize_context(context):
        tokens = tokenizer.tokenize(context)
        char_offsets = list(tokenizer.span_tokenize(context))
        return tokens, char_offsets

    return tokenize_context

def word2vec(word2vec_path):
    """
    Loads 
    """
    from gensim.models import KeyedVectors
    from gensim.scripts.glove2word2vec import glove2word2vec
    # Download word2vec data if it's not present yet
    if not os.path.exists(word2vec_path):
        glove_file_path = get_glove_file_path()
        print('\nConverting Glove to word2vec... ', end='')
        glove2word2vec(glove_file_path, word2vec_path)  # Convert glove to word2vec
        os.remove(glove_file_path)                      # Remove glove file and keep only word2vec
        print('Done')

    print('Reading word2vec data... ', end='')
    model = KeyedVectors.load_word2vec_format(word2vec_path)
    print('Done')

    def get_word_vector(word):
        try:
            return model[word]
        except KeyError:
            return np.zeros(model.vector_size)

    return get_word_vector

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word2vec_path', type=str,
                        default='data/word2vec_from_glove_300.vec',
                        help='Word2Vec vectors file path')
    parser.add_argument('--outfile', type=str, nargs='+', default='data/tmp.pkl',
                        help='Desired path to output pickle')
    parser.add_argument('--include_str', action='store_true',
                        help='Include strings')
    parser.add_argument('data', type=str, nargs='+', help='Data json')
    args = parser.parse_args()

    tokenizer = TreebankWordTokenizer()
    tokenizer_func = word_tokenizer()
    
    print('Loading word2vec ...')
    word_vector = word2vec(args.word2vec_path)

    def parse_sample(context, question, answer_start, answer_end, **kwargs):
        inputs = []
        targets = []

        tokens, char_offsets = tokenizer_func(context)
        try:
            answer_start = [s <= answer_start < e
                            for s, e in char_offsets].index(True)
            targets.append(answer_start)
            answer_end   = [s <= answer_end < e
                            for s, e in char_offsets].index(True)
            targets.append(answer_end)
        except ValueError:
            return None

        tokens = [unidecode(token) for token in tokens]

        context_vecs = [word_vector(token) for token in tokens]
        context_vecs = np.vstack(context_vecs).astype(np.float32)
        inputs.append(context_vecs)

        if args.include_str:
            context_str = [np.fromstring(token, dtype=np.uint8).astype(np.int32)
                        for token in tokens]
            context_str = pad_sequences(context_str, maxlen=25)
            inputs.append(context_str)

        tokens, char_offsets = tokenizer_func(question)
        tokens = [unidecode(token) for token in tokens]

        question_vecs = [word_vector(token) for token in tokens]
        question_vecs = np.vstack(question_vecs).astype(np.float32)
        inputs.append(question_vecs)

        if args.include_str:
            question_str = [np.fromstring(token, dtype=np.uint8).astype(np.int32)
                            for token in tokens]
            question_str = pad_sequences(question_str, maxlen=25)
            inputs.append(question_str)

        return [inputs, targets]

    if not isinstance(args.data, (list, tuple)):
        args.data = list(args.data)
        args.outfile = list(args.outfile)

    assert len(args.data) == len(args.outfile)

    for i in range(len(args.data)):
        if not args.outfile[i].endswith('.pkl'):
            args.outfile += '.pkl'

        # print(args.data[i], args.outfile[i])
        # continue

        print(f'Reading SQuAD data {args.data[i]} ... ', end='')
        with open(args.data[i]) as fd:
            samples = json.load(fd)
        print('Done!')

        print('Parsing samples... ', end='')
        samples = [parse_sample(**sample) for sample in tqdm(samples)]
        samples = [sample for sample in samples if sample is not None]
        print('Done!')

        # Transpose
        def transpose(x):
            return list(map(list, zip(*x)))

        data = [transpose(input) for input in transpose(samples)]

        print('Writing to file {}... '.format(args.outfile[i]), end='')
        with open(args.outfile[i], 'wb') as fd:
            pickle.dump(data, fd, protocol=pickle.HIGHEST_PROTOCOL)
        print('Done!')
