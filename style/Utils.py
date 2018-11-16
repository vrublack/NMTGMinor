def saliency(phrase, dist_this, dist_other, lambda_smooth=1):
    return (dist_this[phrase] + lambda_smooth) / (dist_other[phrase] + lambda_smooth)


from nltk import ngrams, FreqDist, word_tokenize

with open('/Users/valentin/PycharmProjects/text_style_transfer/model/data/valid.0') as f:
    lines_0 = f.read().split('\n')
dist_0 = FreqDist()
for line in lines_0:
    tokens = word_tokenize(line)
    dist_0.update(ngrams(tokens, 1))
    dist_0.update(ngrams(tokens, 2))
    dist_0.update(ngrams(tokens, 3))
    dist_0.update(ngrams(tokens, 4))

with open('/Users/valentin/PycharmProjects/text_style_transfer/model/data/valid.1') as f:
    lines_1 = f.read().split('\n')
dist_1 = FreqDist()
for line in lines_1:
    tokens = word_tokenize(line)
    dist_1.update(ngrams(tokens, 1))
    dist_1.update(ngrams(tokens, 2))
    dist_1.update(ngrams(tokens, 3))
    dist_1.update(ngrams(tokens, 4))


def removeSalient(sentence, dist_this, dist_other, gamma_thresh=15):
    x = word_tokenize(sentence)
    for n_gr in range(4, 0, -1):
        for start in range(len(x) - n_gr + 1):
            phrase = x[start:start + n_gr]
            if saliency(tuple(phrase), dist_this, dist_other) > gamma_thresh:
                for i in range(start, start + n_gr):
                    x[i] = '<del>'

    return ' '.join(x)


with open('/Users/valentin/PycharmProjects/text_style_transfer/model/data/valid-del.0', 'w') as f:
    for line in lines_0:
        f.write(removeSalient(line, dist_0, dist_1) + '\n')

with open('/Users/valentin/PycharmProjects/text_style_transfer/model/data/valid-del.1', 'w') as f:
    for line in lines_1:
        f.write(removeSalient(line, dist_1, dist_0) + '\n')
