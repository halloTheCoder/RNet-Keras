# R-NET implementation in Keras

This repository is an attempt to reproduce the results presented in the [technical report by Microsoft Research Asia](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf). The report describes a complex neural network called **R-NET**.

Till 2017, R-NET was the best single model(i.e. comparision on stand-alone models, without any ensemble) on the Stanford QA database: SQuAD. <br /> 
SQuAD dataset uses two performance metrics, **Exact-Match(EM)** and **F1-score(F1)**. Human performance is estimated to be EM = 82.3% and F1 = 91.2% on the test/dev set.

R-NET (March 2017) has one additional BiGRU between the self-matching attention layer and the pointer network and reaches EM=72.3% and F1=80.7% on the test/dev test. <br />
R-Net at present (on [SQUAD-explorer](https://rajpurkar.github.io/SQuAD-explorer/)) reaches EM=82.136% and F1=88.126%, which means R-NET development continued after March 2017. Also, ensembling the models helped it to reach higher scores.

The best performance I got so far was 
- **EM = 38.55** and **F1 = 47.87%**

Reason for such low metrics
- Chances to be further improved as hyperparameter tuning was not carried out.
- Trained only for 19 epoch due to huge training time, about 3 hrs on Nvidia K40c.
- Further technical reasons can be found in blog.

**I have attached a [PDF document](https://github.com/halloTheCoder/RNet-Keras/tree/master/doc.pdf) explaining the model architecture and the current limitations**

## Requirements
[requirements.txt](https://github.com/halloTheCoder/RNet-Keras/tree/master/requirements.txt)

## Instructions (make sure you are running Keras version 2.0.6)

1. We need to parse and split the data
```sh
python parse_data.py data/train-v1.1.json --train_ratio 0.9 --outfile data/train_parsed.json --outfile_valid data/valid_parsed.json
python parse_data.py data/dev-v1.1.json --outfile data/dev_parsed.json
```

2. Preprocess the data
```sh
python preprocessing.py data/train_parsed.json data/valid_parsed.json data/dev_parsed.json \
--outfile data/train_data_str.pkl data/valid_data_str.pkl data/dev_data_str.pkl --include_str
```

3. Train the model
```sh
python train.py --hdim 45 --batch_size 50 --nb_epochs 50 --optimizer adadelta --lr 1 --dropout 0.2 --char_level_embeddings --train_data data/train_data_str.pkl --valid_data data/valid_data_str.pkl
```

4. Predict on dev/test set samples
```sh
python predict.py --batch_size 100 --dev_data data/dev_data_str.pkl models/31-t3.05458271443-v3.27696280528.model prediction.json
```

5. Evaluate on dev/test set samples
```sh
python evaluate.py --data/dev-v1.1.json --predfile prediction.json
```

Best model can be downloaded from : 
