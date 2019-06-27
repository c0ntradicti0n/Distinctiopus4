rm -r ./output
allennlp train --include-package xnym_embeddings.xnym_embeddings --include-package attention_please_tagger.attention_please_tagger $1 -s ./output/
