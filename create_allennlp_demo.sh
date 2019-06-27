model=$1
python -m allennlp.service.server_simple \
    --include-package attention_please_tagger.attention_please_tagger \
    --include-package xnym_embeddings.xnym_embeddings \
    --include-package difference_predictor.difference_predictor \
    --archive-path $model \
    --predictor difference-tagger \
    --title "AllenNLP Tutorial" \
    --field-name sentence \
    --port 8234