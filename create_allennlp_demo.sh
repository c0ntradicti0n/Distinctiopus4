python -m allennlp.service.server_simple \
    --include-package xnym_embeddings.xnym_embeddings \
    --include-package difference_predictor.difference_predictor \
    --archive-path '0.76F1 Model cooked corpus with aspects and repetition of noise.tar.gz' \
    --predictor difference-tagger \
    --title "AllenNLP Tutorial" \
    --field-name sentence \
    --port 8234