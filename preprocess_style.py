import onmt
import onmt.Markdown
import argparse
import torch


def loadImageLibs():
    "Conditional import of torch image libs."
    global Image, transforms
    from PIL import Image
    from torchvision import transforms


parser = argparse.ArgumentParser(description='preprocess.py')
onmt.Markdown.add_md_help_argument(parser)

# **Preprocess Options**

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-style1_type', default="text",
                    help="Type of the style1 input. Options are [text|img].")
parser.add_argument('-sort_type', default="ascending",
                    help="Type of sorting. Options are [ascending|descending].")
parser.add_argument('-style1_img_dir', default=".",
                    help="Location of style1 images")


parser.add_argument('-train_style1', required=True,
                    help="Path to the training style1 data")
parser.add_argument('-train_style2', required=True,
                    help="Path to the training style2 data")
parser.add_argument('-valid_style1', required=True,
                    help="Path to the validation style1 data")
parser.add_argument('-valid_style2', required=True,
                    help="Path to the validation style2 data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-style1_vocab_size', type=int, default=50000,
                    help="Size of the style1 vocabulary")
parser.add_argument('-style2_vocab_size', type=int, default=50000,
                    help="Size of the style2 vocabulary")
parser.add_argument('-style1_vocab',
                    help="Path to an existing style1 vocabulary")
parser.add_argument('-style2_vocab',
                    help="Path to an existing style2 vocabulary")

parser.add_argument('-style1_seq_length', type=int, default=64,
                    help="Maximum style1 sequence length")
parser.add_argument('-style1_seq_length_trunc', type=int, default=0,
                    help="Truncate style1 sequence length.")
parser.add_argument('-style2_seq_length', type=int, default=66,
                    help="Maximum style2 sequence length to keep.")
parser.add_argument('-style2_seq_length_trunc', type=int, default=0,
                    help="Truncate style2 sequence length.")

parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
                    
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')
parser.add_argument('-join_vocab', action='store_true', help='Using one dictionary for both style1 and style2')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)

def makeJoinVocabulary(filenames, size):
    
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                      lower=opt.lower)
    
    for filename in filenames:
        print("Reading file %s ... " % filename)
        with open(filename) as f:
            for sent in f.readlines():
                for word in sent.split():
                    vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def makeVocabulary(filename, size):
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                      lower=opt.lower)

    with open(filename) as f:
        for sent in f.readlines():
            for word in sent.split():
                vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize, join=False):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        
        # If a dictionary is still missing, generate it.
        if join:
            
            print('Building ' + 'shared' + ' vocabulary...')
            genWordVocab = makeJoinVocabulary(dataFile, vocabSize)
        else:
            print('Building ' + name + ' vocabulary...')
            genWordVocab = makeVocabulary(dataFile, vocabSize)

        vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(style1File, style2File, style1Dicts, style2Dicts, max_style1_length=64, max_style2_length=64):
    style1, style2 = [], []
    sizesStyle1 = []
    sizesStyle2 = []

    print('Processing %s & %s ...' % (style1File, style2File))
    style1F = open(style1File)
    style2F = open(style2File)

    count, ignored = 0, 0
    for l in style1F.readlines():
        l = l.strip()

        # style1 and/or style2 are empty
        if l == "":
            print('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue

        words = l.split()

        if len(words) <= max_style1_length:

            # Check truncation condition.
            if opt.style1_seq_length_trunc != 0:
                words = words[:opt.style1_seq_length_trunc]

            style1 += [style1Dicts.convertToIdx(words,
                                                onmt.Constants.UNK_WORD)]

            sizesStyle1 += [len(words)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)


    count, ignored = 0, 0
    for l in style2F.readlines():
        l = l.strip()

        # style1 and/or style2 are empty
        if l == "":
            print('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue

        words = l.split()

        if len(words) <= max_style2_length:

            # Check truncation condition.
            if opt.style2_seq_length_trunc != 0:
                words = words[:opt.style2_seq_length_trunc]

            style2 += [style2Dicts.convertToIdx(words,
                                                onmt.Constants.UNK_WORD)]

            sizesStyle2 += [len(words)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    style1F.close()
    style2F.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(style1))
        style1 = [style1[idx] for idx in perm]
        style2 = [style2[idx] for idx in perm]
        sizesStyle1 = [sizesStyle1[idx] for idx in perm]
        sizesStyle2 = [sizesStyle2[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm1 = torch.sort(torch.Tensor(sizesStyle1), descending=(opt.sort_type == 'descending'))
    _, perm2 = torch.sort(torch.Tensor(sizesStyle2), descending=(opt.sort_type == 'descending'))
    style1 = [style1[idx] for idx in perm1]
    style2 = [style2[idx] for idx in perm2]

    print(('Prepared %d / %d sentences ' +
          '(%d ignored due to length == 0 or style1 len > %d or style2 len > %d)') %
          (len(style1), len(style2), ignored, max_style1_length, max_style2_length))

    return style1, style2


def main():

    dicts = {}
    
    if opt.join_vocab:
        dicts['style1'] = initVocabulary('style1', [opt.train_style1, opt.train_style2], opt.style1_vocab,
                                      opt.style1_vocab_size, join=True)
        dicts['style2'] = dicts['style1']
    else:
        dicts['style1'] = initVocabulary('style1', opt.train_style1, opt.style1_vocab,
                                      opt.style1_vocab_size)

        dicts['style2'] = initVocabulary('style2', opt.train_style2, opt.style2_vocab,
                                      opt.style2_vocab_size)
                                      
    print('Preparing training ...')
    train = {}
    train['style1'], train['style2'] = makeData(opt.train_style1, opt.train_style2,
                                          dicts['style1'], dicts['style2'],
                                          max_style1_length=opt.style1_seq_length,
                                          max_style2_length=opt.style2_seq_length)

    print('Preparing validation ...')
    valid = {}
    valid['style1'], valid['style2'] = makeData(opt.valid_style1, opt.valid_style2,
                                          dicts['style1'], dicts['style2'], 
                                          max_style1_length=max(256,opt.style1_seq_length), 
                                          max_style2_length=max(256,opt.style2_seq_length))

    if opt.style1_vocab is None:
        saveVocabulary('style1', dicts['style1'], opt.save_data + '.style1.dict')
    if opt.style2_vocab is None:
        saveVocabulary('style2', dicts['style2'], opt.save_data + '.style2.dict')

    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'type':  opt.style1_type,
                 'train': train,
                 'valid': valid}
    torch.save(save_data, opt.save_data + '.train.pt')


if __name__ == "__main__":
    main()
