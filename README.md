# Multi-accent seq2seq frontend

This is a repo for Multi-accent (EDI, GAM, RPX) Sequence-to-Sequence (Seq2Seq) Text-to-Speech (TTS) linguistic frontend.

## Train

Example 1: train a model for GAM
```console
$python train.py --config_path config/gam/gam_r1_nostop_nopre_h384_lr5e-5.json --output_path ../root
```

Example 2: train a model for multiple accents (EDI, GAM, RPX)
```console
$python train.py --config_path config/multi/multi_r1_nostop_nopre_h512_lr5e-5.json --output_path ../root
```

## Inference

See `inference.py` for a full list of arguments. *Note currently beam search is only compatible with* `--batch_size 1`.

Example 1: inference with greedy decoding for lang_id=0
```console
$python inference.py --config_fpath ../root/config.json --src_vocab_fpath ../root/vocab/src.vocab --tgt_vocab_fpath ../root/vocab/tgt.vocab --restore_fpath ../root/step_x_epoch_y.pth.tar --text_fpath fpath/to/text/file/for/inference  --output_dir dir/for/output --lang 0 --batch_size 1 --decoding_method greedy
```

Example 2: inference with beam search decoding for lang_id=1, beam_size=5, n_best=5 candidates are kept.
```console
$python inference.py --config_fpath ../root/config.json --src_vocab_fpath ../root/vocab/src.vocab --tgt_vocab_fpath ../root/vocab/tgt.vocab --restore_fpath ../root/step_x_epoch_y.pth.tar --text_fpath fpath/to/text/file/for/inference  --output_dir dir/for/output --lang 1 --batch_size 1 --decoding_method beam_search --beam_size 5 --n_best 5
```

## Citation

Paper: Learning Pronunciation from Other Accents via Pronunciation Knowledge Transfer
Authors: Siqi Sun, Korin Richmond

Bibtex to be included