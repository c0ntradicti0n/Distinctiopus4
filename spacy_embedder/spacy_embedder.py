import pprint
import sys

import overrides.overrides as overrides
import torch
from allennlp.data import Vocabulary
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from nltk import flatten
from numpy import float32
import numpy as np
from sklearn import preprocessing
from spacy.attrs import *
from sklearn.preprocessing import Normalizer
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

import spacy
nlp = spacy.load("en_core_sci_sm")

def prevent_sentence_boundary_detection(doc):
    for token in doc:
        # This will entirely disable spaCy's sentence detection
        token.is_sent_start = False
    return doc

nlp.add_pipe(prevent_sentence_boundary_detection, name='prevent-sbd', before='parser')

@TokenEmbedder.register("spacy_embedder")
class SpacyEmbedder (TokenEmbedder):
    """
    Represents a sequence of tokens as a relation based embeddings.


    Each sequence gets a vector of length vocabulary size, where the i'th entry in the vector
    corresponds to number of times the i'th token in the vocabulary appears in the sequence.
    By default, we ignore padding tokens.
    Parameters
    ----------
    vocab: ``Vocabulary``
    projection_dim : ``int``, optional (default = ``None``)
        if specified, will project the resulting bag of positions representation
        to specified dimension.
    ignore_oov : ``bool``, optional (default = ``False``)
        If true, we ignore the OOV token.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 tags: str = [],
                 normalize=True):
        super(SpacyEmbedder, self).__init__()
        self.tag_functions = [eval("lambda t: t.%s"% tag) for tag in tags]
        # they said 'don't use 'eval' TODO security
        self.vocab = vocab
        self.normalize = normalize
        self.labelbinarizer = preprocessing.LabelBinarizer()
        print (set(type(s) for s in dir(spacy.symbols)))
        self.spacy_labels = [eval(".".join(["spacy.symbols", item])) for item in dir(spacy.symbols) if item and not item.startswith("__")]
        self.spacy_labels = [l for l in self.spacy_labels if isinstance(l, int)]
        self.labelbinarizer.fit(self.spacy_labels) # hack to cython properties...

    @overrides
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, timesteps, sequence_length)`` of word ids
            representing the current batch.
        Returns
        -------
        The distance position representations for the input sequence, shape
        ``(batch_size, vocab_size)``
        """
        input_array = inputs.cpu().detach().numpy()
        (batch_size, sequence_length) = input_array.shape
        strings = [[self.vocab.get_index_to_token_vocabulary()[i] for i in inp] for inp in input_array]
        sentences = [[token for token in sentence if token not in [DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN]] for sentence in strings]
        docs = [nlp(" ".join(sentence)) for sentence in sentences]
        #print ("first sentence", docs[0])
        #print (docs, 'DOCs')
        tagged = [
            [
                SpacyEmbedder.pad_or_truncate([f(t) for t in doc ],
                                                                            target_len=sequence_length)
             for f in self.tag_functions]
                for doc in docs ]
        tagged = np.array(tagged).astype(np.int32)
        #print ("first tags fro spacy", tagged[0])
        #print (tagged.shape)
        #print (tagged.shape)
        #print (set(type(t) for t in flatten(tagged.tolist())))
        tagged = [[self.labelbinarizer.transform(tagged_kind) for tagged_kind in tagged_doc] for tagged_doc in tagged]
        tagged = np.array(tagged, dtype=float32)
        tagged = tagged.reshape(batch_size, sequence_length, self.get_output_dim())
        #print ("first token", tagged[0,0,:])
        #print (tagged.shape)
        #print (set(type(t) for t in flatten(tagged.tolist())))
        tensor = torch.from_numpy(tagged)
        #np.set_printoptions(threshold=sys.maxsize)
        #pprint.pprint (tagged)
        return tensor

    def pad_or_truncate(some_list, target_len):
        return some_list[:target_len] + [0] * (target_len - len(some_list))

    @overrides
    def get_output_dim(self) -> int:
        return len(self.spacy_labels) * len(self.tag_functions)

