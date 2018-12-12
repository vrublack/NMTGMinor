import onmt
import onmt.Markdown
import argparse
import torch

from onmt.data_utils.IndexedDataset import IndexedDatasetBuilder


def loadImageLibs():
    "Conditional import of torch image libs."
    global Image, transforms
    from PIL import Image
    from torchvision import transforms


parser = argparse.ArgumentParser(description='preprocess.py')
onmt.Markdown.add_md_help_argument(parser)

# **Preprocess Options**

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-src_type', default="text",
                    help="Type of the source input. Options are [text|img].")
parser.add_argument('-sort_type', default="ascending",
                    help="Type of sorting. Options are [ascending|descending].")
parser.add_argument('-input_type', default="word",
                    help="Input type: word/char")
parser.add_argument('-format', default="bin",
                    help="Save data format: binary or raw. Binary should be used to load faster")


parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                    help="Path to the validation target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")

parser.add_argument('-src_seq_length', type=int, default=64,
                    help="Maximum source sequence length")
parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                    help="Truncate source sequence length.")
parser.add_argument('-tgt_seq_length', type=int, default=66,
                    help="Maximum target sequence length to keep.")
parser.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                    help="Truncate target sequence length.")

parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
                    
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')
parser.add_argument('-remove_duplicate', action='store_true', help='remove duplicated sequences')
parser.add_argument('-sort_by_target', action='store_true', help='lowercase data')
parser.add_argument('-join_vocab', action='store_true', help='Using one dictionary for both source and target')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)

def makeJoinVocabulary(filenames, size, input_type="word"):
    
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                      lower=opt.lower)
    
    for filename in filenames:
        print("Reading file %s ... " % filename)
        with open(filename) as f:
            for sent in f.readlines():
                
                if input_type == "word":
                    for word in sent.split():
                        vocab.add(word)
                elif input_type == "char":
                    sent = sent.strip()
                    for char in sent:
                        vocab.add(char)
                else:
                    raise NotImplementedError("Input type not implemented")
                

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def makeVocabulary(filename, size, input_type='word'):
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                      lower=opt.lower)

    with open(filename) as f:
        for sent in f.readlines():
            if input_type == "word":
                for word in sent.split():
                    vocab.add(word)
            elif input_type == "char":
                sent = sent.strip()
                for char in sent:
                    vocab.add(char)
            else:
                raise NotImplementedError("Input type not implemented")

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize, join=False, input_type='word'):

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
            genWordVocab = makeJoinVocabulary(dataFile, vocabSize, input_type=input_type)
        else:
            print('Building ' + name + ' vocabulary...')
            genWordVocab = makeVocabulary(dataFile, vocabSize, input_type=input_type)

        vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, tgtDicts, max_src_length=64, max_tgt_length=64, sort_by_target=False, input_type='word', remove_duplicate=False):
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0
    n_duplicate = 0
    src_sizes = []
    tgt_sizes = []

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile)
    tgtF = open(tgtFile)

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: src and tgt do not have the same # of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        all_data = []

        # source and/or target are empty
        if remove_duplicate:
            if sline == tline:
                n_duplicate += 1
                # ~ print('WARNING: ignoring a duplicated pair ('+str(count+1)+')')
                continue
            
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            continue
        
        if input_type == 'word':
            srcWords = sline.split()
            tgtWords = tline.split()
        elif input_type == 'char':
            srcWords = list(sline)
            tgtWords = list(tline)

        if len(srcWords) <= max_src_length \
           and len(tgtWords) <= max_tgt_length - 2:

            # Check truncation condition.
            if opt.src_seq_length_trunc != 0:
                srcWords = srcWords[:opt.src_seq_length_trunc]
            if opt.tgt_seq_length_trunc != 0:
                tgtWords = tgtWords[:opt.tgt_seq_length_trunc]
            
            
            # For src text, we use BOS for possible reconstruction
            
            src_sent = srcDicts.convertToIdx(srcWords,
                                              onmt.Constants.UNK_WORD)
            src += [src_sent]                                 

            tgt_sent = tgtDicts.convertToIdx(tgtWords,
                                          onmt.Constants.UNK_WORD,
                                          onmt.Constants.BOS_WORD,
                                          onmt.Constants.EOS_WORD)
            tgt += [tgt_sent]
            # if sort_by_target:
                # sizes += [len(tgtWords)]
            # else:
            #     sizes += [len(srcWords)]
            src_sizes += [len(src_sent)]
            tgt_sizes += [len(tgt_sent)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)
    
    if remove_duplicate:
        print(' ... %d sentences removed for duplication' % n_duplicate)
    srcF.close()
    tgtF.close()


    print('... sorting sentences by size')
    z = zip(src, tgt, src_sizes, tgt_sizes)

    sorted_by_src_z = sorted(z, key=lambda x: x[2])
    sorted_by_tgt_z = sorted(sorted_by_src_z, key=lambda x: x[3])

    src = [z_[0] for z_ in sorted_by_tgt_z]
    tgt = [z_[1] for z_ in sorted_by_tgt_z]


    print(('Prepared %d sentences ' +
          '(%d ignored due to error or src len > %d or tgt len > %d)') %
          (len(src), ignored, max_src_length, max_tgt_length))

    return src, tgt


def main():

    dicts = {}
    
    if opt.join_vocab:
        dicts['src'] = initVocabulary('source', [opt.train_src, opt.train_tgt], opt.src_vocab,
                                      opt.tgt_vocab_size, join=True, input_type=opt.input_type)
        dicts['tgt'] = dicts['src']
    else:
        dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,
                                      opt.src_vocab_size, input_type=opt.input_type)

        dicts['tgt'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
                                      opt.tgt_vocab_size, input_type=opt.input_type)
    train = {}
    valid = {}
    
    print('Preparing training ...')
        
    train['src'], train['tgt'] = makeData(opt.train_src, opt.train_tgt,
                                          dicts['src'], dicts['tgt'],
                                          max_src_length=opt.src_seq_length,
                                          max_tgt_length=opt.tgt_seq_length, 
                                          sort_by_target=opt.sort_by_target,
                                          input_type=opt.input_type,
                                          remove_duplicate=opt.remove_duplicate)

    print('Preparing validation ...')
   
    valid['src'], valid['tgt'] = makeData(opt.valid_src, opt.valid_tgt,
                                          dicts['src'], dicts['tgt'], 
                                          max_src_length=max(1024,opt.src_seq_length), 
                                          max_tgt_length=max(1024,opt.tgt_seq_length),
                                          input_type=opt.input_type)
    
    if opt.format == 'raw':                                  
        

        if opt.src_vocab is None:
            saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
        if opt.tgt_vocab is None:
            saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')

        print('Saving data to \'' + opt.save_data + '.train.pt\'...')
        save_data = {'dicts': dicts,
                     'type':  opt.src_type,
                     'train': train,
                     'valid': valid}
        torch.save(save_data, opt.save_data + '.train.pt')
        print("Done")
    elif opt.format in ['bin', 'binary']:
        # save the dictionary first
        torch.save(dicts, opt.save_data + '.dict.pt')
        
        train_bin = dict()
        
        # binarize the data now
        for set in ['src', 'tgt']:
            train_bin[set] = IndexedDatasetBuilder(opt.save_data + ".train.%s.bin" % set )
            
            for tensor_sent in train[set]:
                train_bin[set].add_item(tensor_sent)
                
            train_bin[set].finalize(opt.save_data + ".train.%s.idx" % set)
            
        valid_bin = dict()
        for set in ['src', 'tgt']:
            valid_bin[set] = IndexedDatasetBuilder(opt.save_data + ".valid.%s.bin" % set )
            
            for tensor_sent in valid[set]:
                valid_bin[set].add_item(tensor_sent)
                
            valid_bin[set].finalize(opt.save_data + ".valid.%s.idx" % set)
        
        print("Done")
    else: raise NotImplementedError


if __name__ == "__main__":
    main()
