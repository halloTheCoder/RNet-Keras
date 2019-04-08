import os
import sys
import re
import string
import json
from collections import Counter
import argparse

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate(dataset, predictions):
    f1 = 0
    exact_match = 0
    total = 0
    for article in dataset:
        total += 1
        if article['id'] not in predictions:
            message = 'Unanswered question ' + article['id'] + ' will receive score 0.'
            print(message, file=sys.stderr)
            continue
        ground_truths = list(map(lambda x: x['answer'], article['answers']))
        prediction = predictions[article['id']]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, 
                        help='Path to the dataset file')
    parser.add_argument('--predfile', type=str,
                        help='Desired path to output pickle')

    args = parser.parse_args()
    
    if not args.data.endswith('.json'):
        args.data += '.json'
    
    if not args.predfile.endswith('.json'):
        args.predfile += '.json'

    print('Reading SQuAD dev data... ', end='')
    with open(args.data) as fd:
        data = json.load(fd)
    print('Done!')
    
    data = data['data']
    
    print('Parsing dev data file... ', end='')
    samples = []
    for topic in data:
        cqas = [{'context' :      paragraph['context'],
                 'id' :           qa['id'],
                 'question' :     qa['question'],
                 'answers' : [{
                     'answer' :       ans['text'],
                     'answer_start' : ans['answer_start'],
                     'answer_end' :   ans['answer_start'] + len(ans['text']) - 1
                 } for ans in qa['answers']],
                 'topic' :        topic['title']
                } 
                for paragraph in topic['paragraphs'] 
                for qa in paragraph['qas']]
        
        samples += cqas
    print('Done!')
    
    print('Loading prediction file... ', end='')
    with open(args.predfile) as f:
        preds = json.load(f)
    print('Done')
    
    print(json.dumps(evaluate(samples, preds)))
