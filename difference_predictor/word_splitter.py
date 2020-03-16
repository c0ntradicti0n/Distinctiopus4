import re
from typing import List, Optional

from allennlp.data.tokenizers.word_splitter import WordSplitter, _remove_spaces
from overrides import overrides
import spacy
from spacy.tokens import Doc
import ftfy

from pytorch_pretrained_bert.tokenization import BasicTokenizer as BertTokenizer

from allennlp.common import Registrable
from allennlp.common.util import get_spacy_model
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.openai_transformer_byte_pair_indexer import text_standardize


@WordSplitter.register('customspacy')
class CustomSpacyWordSplitter(WordSplitter):
    """
    A ``WordSplitter`` that uses spaCy's tokenizer.  It's fast and reasonable - this is the
    recommended ``WordSplitter``. By default it will return allennlp Tokens,
    which are small, efficient NamedTuples (and are serializable). If you want
    to keep the original spaCy tokens, pass keep_spacy_tokens=True.
    """
    def __init__(self,
                 language: str = 'en_core_web_sm',
                 pos_tags: bool = False,
                 parse: bool = False,
                 ner: bool = False,
                 keep_spacy_tokens: bool = False) -> None:
        self.spacy = get_spacy_model(language, pos_tags, parse, ner)

        self._keep_spacy_tokens = keep_spacy_tokens

    def _sanitize(self, tokens: List[spacy.tokens.Token]) -> List[Token]:
        """
        Converts spaCy tokens to allennlp tokens. Is a no-op if
        keep_spacy_tokens is True
        """
        if self._keep_spacy_tokens:
            return tokens
        else:
            return [Token(token.text,
                          token.idx,
                          token.lemma_,
                          token.pos_,
                          token.tag_,
                          token.dep_,
                          token.ent_type_) for token in tokens]

    @overrides
    def batch_split_words(self, sentences: List[str]) -> List[List[Token]]:
        return [self._sanitize(_remove_spaces(tokens))
                for tokens in self.spacy.pipe(sentences, n_threads=-1)]

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        # This works because our Token class matches spacy's.
        return self._sanitize(_remove_spaces(self.spacy(sentence)))

    def split_pretokenized_words(self, sentence: List[str]) -> List[Token]:
        # This works because our Token class matches spacy's.
        doc = self.spacy.tokenizer.tokens_from_list(sentence)
        return self._sanitize(_remove_spaces(doc))

