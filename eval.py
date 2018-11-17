import json
import pickle as pk

from nltk.translate.bleu_score import sentence_bleu

from generate import predict

from util import map_item


path_label_ind = 'feat/nn/label_ind.pkl'
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

slots = list(label_inds.keys())
slots.remove('N')
slots.remove('O')


def test(name, sents):
    label_mat = list()
    pred_mat = list()
    for text, quaples in sents.items():
        words = text.split()
        tags = list()
        labels = list()
        for quaple in quaples:
            tags.append(quaple['pos'])
            labels.append(quaple['label'])
        label_mat.append(labels)
        if name == 'crf':
            pairs = predict(words, tags)
        else:
            pairs = predict(words, name)
        preds = [pred for word, pred in pairs]
        pred_mat.append(preds)
    f1 = flat_f1_score(label_mat, pred_mat, average='weighted', labels=slots)
    print('\n%s %s %.2f' % (name, ' f1:', f1))
    print('%s %s %.2f' % (name, 'acc:', flat_accuracy_score(label_mat, pred_mat)))


if __name__ == '__main__':
    path = 'data/test.json'
    with open(path, 'r') as f:
        sents = json.load(f)
    test('crf', sents)
    test('dnn', sents)
    test('rnn', sents)
    test('rnn_bi', sents)
    test('rnn_bi_crf', sents)