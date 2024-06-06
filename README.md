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

Example for inference. See `inference.py` for a full list of arguments.
```console
$python inference.py --config_path ../root/config.json --src_vocab_fpath ../root/vocab/src.vocab --tgt_vocab_fpath ../root/vocab/tgt.vocab --restore_fpath ../root/step_x_.pth.tar --text_fpath 'fpath/to/text/file/for/inference'  --output_dir 'dir/to/output' --batch_size 1
```

## Citation

Paper: Learning Pronunciation from Other Accents via Pronunciation Knowledge Transfer

To be included