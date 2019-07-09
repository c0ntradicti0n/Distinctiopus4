import tempfile

from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.models import load_archive

import xnym_embeddings.xnym_embeddings
import attention_please_tagger.attention_please_tagger
import spacy_embedder.spacy_embedder
params = Params.from_file('experiment_configs/difference_stacked_birectional_lstm3_without_elmo.config')
serialization_dir = tempfile.mkdtemp()
model = train_model(params, serialization_dir)

load_archive