from tacotron import Tacotron


def setup_model(c, num_chars, num_phones):
    model = Tacotron(
        num_chars=num_chars,
        num_phones=num_phones,
        num_langs=len(c.data),
        enc_embedding_dim=c.enc_embedding_dim,
        dec_embedding_dim=c.dec_embedding_dim,
        lang_embedding_dim=c.lang_embedding_dim,
        enc_hidden_dim=c.enc_hidden_dim,
        dec_hidden_dim=c.dec_hidden_dim,
        post_hidden_dim=c.post_hidden_dim,
        has_prenet=c.has_prenet,
        dropout=c.dropout,
        r=c.r,
        separate_stopnet=c.separate_stopnet,
        attn_type=c.attn_type,
        num_mixtures=c.num_mixtures,
        gmm_version=c.gmm_version,
        has_postnet=c.has_postnet,
        has_stopnet=c.has_stopnet)
    return model
