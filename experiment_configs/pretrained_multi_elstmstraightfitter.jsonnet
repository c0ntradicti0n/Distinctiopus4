// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “End-to-end Neural Coreference Resolution.” EMNLP (2017).
//   + BERT

local elmo_model_name = "elmo";

{
  "dataset_reader": {
    "type": "conll2003",
    "tag_label": "ner",
    "coding_scheme": "BIOUL",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      },
      "elmo": {
        "type": "elmo_characters"
     }
    }
  },
  "train_data_path": "./manual_corpus/train.conll3",
  "validation_data_path": "./manual_corpus/test.conll3",
  "model": {
    "type": "attentive_crf_tagger",
    "label_encoding": "BIOUL",
    "constrain_crf_decoding": true,
    "calculate_span_f1": true,
    "dropout": 0.5,
    "include_start_end_transitions": false,
    "feedforward": {
         "input_dim": 200,
         "num_layers": 7,
         "hidden_dims": [200,200,200,200,200,200,200],
         "activations": "relu"
    },
    "text_field_embedder": {
      "token_embedders": {
              "tokens": {
            "type": "embedding",
            "embedding_dim": 50,
            "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.50d.txt.gz",
            "trainable": true
        },
          "elmo":{
                "type": "elmo_token_embedder",
            "options_file": "models/elmo_2x1024_128_2048cnn_1xhighway_options.json",
            "weight_file": "models/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5",
                "do_layer_norm": false,
                "dropout": 0.0
            },
            "token_characters": {
                "type": "character_encoding",
                "embedding": {
                    "embedding_dim": 9
                    },
                "encoder": {
                "type": "cnn",
                "embedding_dim": 9,
                "num_filters": 53,
                "ngram_filter_sizes": [3],
                "conv_layer_activation": "relu"
                }
        }
      }
    },
    "encoder": {
      "type": "stacked_bidirectional_lstm",
      "input_size":  359,
      "hidden_size": 100,
      "num_layers": 3,
    },
    "regularizer": [
      [
        "scalar_parameters",
        {
          "type": "l2",
          "alpha": 0.1
        }
      ]
    ]
  },
  "iterator": {
        "type": "basic",
        "batch_size": 64
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr":    0.003
    },
    "histogram_interval":10,
    "shuffle": true,
    "validation_metric": "+f1-measure-overall",
    "num_serialized_models_to_keep": 3,
    "num_epochs": 100,
    "grad_norm": 5.0,
    "patience": 100,
    "cuda_device": -1
  }
}