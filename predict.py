import numpy as np
import argparse
import json
import os

from tqdm import tqdm

from keras import backend as K
from keras.models import Model, load_model

from data import BatchGen, load_dataset
from preprocessing import word_tokenizer

from layers.QuestionAttnGRU import QuestionAttnGRU
from layers.SelfAttnGRU import SelfAttnGRU
from layers.PointerGRU import PointerGRU
from layers.QuestionPooling import QuestionPooling
from layers.VariationalDropout import VariationalDropout
from layers.Slice import Slice
from layers.SharedWeight import SharedWeightLayer
from layers.Argmax import Argmax

from model import RNet

np.random.seed(10)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=70, type=int, help='Batch size')
parser.add_argument('--dev_data', default='data/dev_data.pkl', type=str,
                     help='Validation Set')
parser.add_argument('model', type=str, help='Model to run')
parser.add_argument('prediction', type=str, default='pred.json',
                     help='Outfile to save predictions')
args = parser.parse_args()

print('Preparing model...', end='')
model = load_model(args.model, locals())

inputs = model.inputs
outputs = [ Argmax() (output) for output in model.outputs ]

predicting_model = Model(inputs, outputs)
print('Done!')

print('Loading data...', end='')
dev_data = load_dataset(args.dev_data)
char_level_embeddings = len(dev_data[0]) is 4
maxlen = [300, 300, 30, 30] if char_level_embeddings else [300, 30]
#Loading dev data using BatchGen class
dev_data_gen = BatchGen(*dev_data, batch_size=args.batch_size, shuffle=False, group=False, maxlen=maxlen)

with open('data/dev_parsed.json') as f:
    samples = json.load(f)
print('Done!')

print('Running predicting model...', end='')
predictions = predicting_model.predict_generator(generator=dev_data_gen,
                                                 steps=dev_data_gen.steps(),
                                                 verbose=1)
print('Done!')

tokenize = word_tokenizer()

print('Preparing prediction file...', end='')
contexts = [sample['context'] for sample in samples]

answers = {}
for sample, context, start, end in tqdm(zip(samples, contexts, *predictions)):
    id = sample['id']
    context_tokens, _ = tokenize(context)
    answer = ' '.join(context_tokens[start : end+1])
    answers[id] = answer
print('Done!')

print('Writing predictions to file {}...'.format(args.prediction), end='')
with open(args.prediction, 'w') as f:
    json.dump(answers, f)
print('Done!')
