def add_default_param(c, param_key, param_value):
    if not hasattr(c, param_key):
        c[param_key] = param_value


def set_default_params(c):
    add_default_param(c, 'enc_embedding_dim', 512)
    add_default_param(c, 'dec_embedding_dim', 512)
    add_default_param(c, 'lang_embedding_dim', 32)
    add_default_param(c, 'enc_hidden_dim', 512)
    add_default_param(c, 'dec_hidden_dim', 512)
    add_default_param(c, 'post_hidden_dim', 512)
    add_default_param(c, 'has_prenet', True)
    add_default_param(c, 'dropout', 0.3)
    add_default_param(c, 'r', 5)
    add_default_param(c, 'attn_type', 'gmm')
    add_default_param(c, 'num_mixtures', 5)
    add_default_param(c, 'gmm_version', 'v2')
    add_default_param(c, 'has_stopnet', True)
    add_default_param(c, 'separate_stopnet', True)
    add_default_param(c, 'has_postnet', False)

    add_default_param(c, 'learning_rate', 0.001)
    add_default_param(c, 'clip_thresh', 1)
    add_default_param(c, 'check_grad_norm', False)
    add_default_param(c, 'batch_size', 64)
    add_default_param(c, 'batch_group_size', 50)
    add_default_param(c, 'over_sampling', None)
    add_default_param(c, 'min_seq_len', 0)
    add_default_param(c, 'max_seq_len', 500)
    add_default_param(c, 'load_data_workers', 4)
    add_default_param(c, 'print_step', 100)
    add_default_param(c, 'save_step', 10000)
    add_default_param(c, 'max_decoder_steps', 500)

    add_default_param(c, 'scheduled_sampling', {})
    add_default_param(c, 'verbose', True)

    add_default_param(c, 'src_vocab', 'vocab/src.vocab')
    add_default_param(c, 'tgt_vocab', 'vocab/tgt.vocab')
    add_default_param(c, 'word_vocab', 'vocab/src.word.vocab')

    return c
