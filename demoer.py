from allennlp.service.server_simple import main
import sys
import xnym_embeddings.xnym_embeddings
import attention_please_tagger.attention_please_tagger
import difference_predictor.difference_predictor


if __name__ == "__main__":
    these_args = sys.argv[1:]
    main(these_args)

