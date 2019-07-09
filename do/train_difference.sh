rm -r ./output/$1
allennlp train --include-package xnym_embeddings.xnym_embeddings \
               --include-package spacy_embedder.spacy_embedder \
               --include-package attention_please_tagger.attention_please_tagger \
               $1 -s ./output/$1/

