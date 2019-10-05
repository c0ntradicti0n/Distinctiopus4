import logging
from typing import Dict, List

from overrides import overrides
import torch

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from pywsd import disambiguate

import numpy as np

logger = logging.getLogger(__name__)

import pywsd

import pickle

@TokenIndexer.register("synset_indexer")
class SynsetIndexer(TokenIndexer[str]):

    def __init__(self, namespace: str = "synset_indexer", token_min_padding_length: int = 0, key_path: str= "") -> None:
        super().__init__(token_min_padding_length)
        self._namespace = namespace
        with open(key_path, 'rb') as handle:
            self.tok2id, self.id2tok = pickle.load(handle)



    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary, index_name: str
    ) -> Dict[str, List[int]]:
        synsets = disambiguate(" ".join([t.text.lower() for t in tokens]), algorithm=pywsd.similarity.max_similarity, similarity_option='wup',
                     keepLemmas=True)
        return {index_name: [self.tok2id[str(s[2])] if str(s[2]) in self.tok2id else 0  for s in synsets] }

    @overrides
    def get_padding_lengths(self, token: List[int]) -> Dict[str, int]:
        # pylint: disable=unused-argument
        return {}


    @overrides
    def as_padded_tensor(
        self,
        tokens: Dict[str, List[int]],
        desired_num_tokens: Dict[str, int],
        padding_lengths: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        return {
            key: torch.IntTensor(pad_sequence_to_length(val, desired_num_tokens[key]))
            for key, val in tokens.items()
        }
