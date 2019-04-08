# R-NET implementation in Keras

This repository is an attempt to reproduce the results presented in the [technical report by Microsoft Research Asia](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf). The report describes a complex neural network called [R-NET](https://www.microsoft.com/en-us/research/publication/mrc/) designed for question answering.

The best performance I got so far was 
- EM=57.52% and F1=67.42% on the dev set. 

## Requirements
`
keras==2.0.6
tensorflow==1.13.1
gensim==3.7.1
nltk==3.4
`

## Instructions (make sure you are running Keras version 2.0.6)

1. We need to parse and split the data
```sh
python parse_data.py data/train-v1.1.json --train_ratio 0.9 --outfile data/train_parsed.json --outfile_valid data/valid_parsed.json
python parse_data.py data/dev-v1.1.json --outfile data/dev_parsed.json
```

2. Preprocess the data
```sh
python preprocessing.py data/train_parsed.json --outfile data/train_data_str.pkl --include_str
python preprocessing.py data/valid_parsed.json --outfile data/valid_data_str.pkl --include_str
python preprocessing.py data/dev_parsed.json --outfile data/dev_data_str.pkl --include_str
```

3. Train the model
```sh
python train.py --hdim 45 --batch_size 50 --nb_epochs 50 --optimizer adadelta --lr 1 --dropout 0.2 --char_level_embeddings --train_data data/train_data_str.pkl --valid_data data/valid_data_str.pkl
```

4. Predict on dev/test set samples
```sh
python predict.py --batch_size 100 --dev_data data/dev_data_str.pkl models/31-t3.05458271443-v3.27696280528.model prediction.json
```

Our best model can be downloaded from Release v0.1: https://github.com/YerevaNN/R-NET-in-Keras/releases/download/v0.1/31-t3.05458271443-v3.27696280528.model
