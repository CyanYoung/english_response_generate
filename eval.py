from nltk.translate.bleu_score import sentence_bleu

from build import load_feat

from generate import predict


path_feats = dict()
path_feats['sent1'] = 'feat/sent1_train.pkl'
path_feats['sent2'] = 'feat/sent2_train.pkl'
path_feats['label'] = 'feat/label_train.pkl'
sent1s, sent2s, labels = load_feat(path_feats)


def test(name, sent1s, labels):
    preds = list()
    for sent1 in sent1s:
        preds.append(predict(sent1, name, 'search'))
    print('\n%s %s %.2f\n' % (name, 'bleu:', sentence_bleu(labels, preds)))


if __name__ == '__main__':
    test('s2s', sent1s, labels)
    test('att', sent1s, labels)
