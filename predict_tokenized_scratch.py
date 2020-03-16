from pprint import pprint

from allennlp.predictors import Predictor
from difference_predictor.difference_predictor import MoreVersatileTaggerPredictor

from attentivecrftagger.attentivecrftagger import AttentiveCrfTagger


default_predictor = Predictor.from_path("../CorpusCook/server/models/model_first.tar.gz")
tokens = ["I", "want", "to", "have", "an", "In./././/stance"] # This would have been confused by the spacy tokenizer
predictor = MoreVersatileTaggerPredictor(default_predictor._model, dataset_reader=default_predictor._dataset_reader)

pprint (predictor.predict_tokens(tokens))

