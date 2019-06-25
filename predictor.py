

# predict and write to test set
import xnym_embeddings.xnym_embeddings
from allennlp.predictors import Predictor
predictor = Predictor.from_path("0.699F1 Model with only subject contrast and real noise.tar.gz")

import jsonlines

with jsonlines.open('./corpus_data/sentences.jsonl') as reader:
    for obj in reader:
        sentence = obj['sentence']
        results = predictor.predict(sentence=sentence, )
        for word, tag in zip(results["words"], results["tags"]):
            print(f"{word}\t{tag}")
