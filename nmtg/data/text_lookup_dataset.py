import logging
import os

import numpy as np

from nmtg.data import data_utils
from nmtg.data.dictionary import Dictionary
from nmtg.data.dataset import Dataset, TextLineDataset


logger = logging.getLogger(__name__)


class TextLookupDataset(Dataset):
    def __init__(self, text_dataset: TextLineDataset, dictionary: Dictionary, words=True,
                 lower=False, bos=True, eos=True, align_right=False, trunc_len=0, lang=None, domain_label=False):
        """
        A dataset which contains indices derived by splitting
        text lines and looking up indices in a dictionary
        :param text_dataset: The source text
        :param dictionary: The lookup
        :param words: Whether to split characters or words
        :param bos: Whether to include a beginning-of-sequence token
        :param eos: Whether to include an end-of-sequence token
        :param align_right: Whether to align the padded batches to the right
        :param domain_label: Whether to treat last sequence element as the domain label for the sequence that shouldn't be in the sequence because the discriminator has to classify it
        """
        self.source = text_dataset
        self.dictionary = dictionary
        self.words = words
        self.lower = lower
        self.bos = bos
        self.eos = eos
        self.align_right = align_right
        self.trunc_len = trunc_len
        self.lang = lang
        self.domain_label = domain_label

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        line = self.source[index]
        has_domain_label = False
        if self.domain_label:
            dom_tag_start_str = '<dom'
            dom_tag_end_str = '>'
            start_pos = line.find(dom_tag_start_str)
            if start_pos != -1:  # on target side there are no labels
                has_domain_label = True
                end_pos = line.find(dom_tag_end_str, start_pos + 1)
                domain_index = int(line[start_pos + len(dom_tag_start_str):end_pos]) - 1
                line = line[:start_pos]
        if self.lower:
            line = line.lower()
        if self.words:
            line = line.split()
        if self.trunc_len > 0:
            line = line[:self.trunc_len]
        if len(line) == 0:
            logger.warning('Zero-length input at {}'.format(index))
        if self.lang is None:
            seq = self.dictionary.to_indices(line, bos=self.bos, eos=self.eos)
        else:
            # noinspection PyArgumentList
            seq = self.dictionary.to_indices(line, bos=self.bos, eos=self.eos, lang=self.lang)

        if has_domain_label:
            return seq, domain_index
        else:
            return seq

    def collate_samples(self, samples):
        indices, lengths = data_utils.collate_sequences(samples,
                                                        self.dictionary.pad(),
                                                        self.align_right)
        return {'indices': indices, 'lengths': lengths, 'size': lengths.sum().item()}

    @classmethod
    def load(cls, filename, dictionary, data_dir, load_into_memory=False, words=True, **kwargs):
        base_name = os.path.basename(filename)

        if load_into_memory:
            text_data = TextLineDataset.load_into_memory(filename)
            if words:
                lengths = np.array([len(sample.split()) for sample in text_data])
            else:
                lengths = np.array([len(sample) for sample in text_data])
        else:
            offsets_filename = os.path.join(data_dir, base_name + '.idx.npy')
            text_data = TextLineDataset.load_indexed(filename, offsets_filename)
            lengths_filename = os.path.join(data_dir, base_name + '.len.npy')
            lengths = np.load(lengths_filename)

        return cls(text_data, dictionary, words, **kwargs), lengths
