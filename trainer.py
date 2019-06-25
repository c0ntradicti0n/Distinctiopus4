import tempfile

from allennlp.commands.train import train_model
from allennlp.common import Params

import xnym_embeddings.xnym_embeddings
import attention_please_tagger.attention_please_tagger
params = Params.from_file('difference.config')
serialization_dir = tempfile.mkdtemp()
model = train_model(params, serialization_dir)