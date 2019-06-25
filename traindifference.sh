rm -r ./output
allennlp train --include-package xnym_embeddings.xnym_embeddings --include-package attention_please_tagger.attention_please_tagger ./difference.config -s ./output/
