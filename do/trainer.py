import tempfile

from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.models import load_archive

import xnym_embeddings.xnym_embeddings
import attentivecrftagger.attentivecrftagger
import spacy_embedder.spacy_embedder
params = Params.from_file('experiment_configs/elmo_lstm3_feedforward4_crf_straight.config')
serialization_dir = tempfile.mkdtemp()
model = train_model(params, serialization_dir)

load_archive