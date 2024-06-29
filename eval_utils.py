import re
import nltk
import pandas as pd
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import random
import statistics as st


def compare_word_prons(ref_pron, pred_pron):
    """
    Get the detailed comparison between the reference word pronunciation and the prediction word pronunciation

    Args:
        ref_pron (str): reference word pronunciation
        pred_pron (str): prediction word pronunciation

    Returns:
        _ (dict): a dict of comparison results

    """
    pron_match = ref_pron == pred_pron

    # get phoneme edit distance and check whether the phoneme sequences match
    ref_phones = re.sub(r'[0-9\-]', '', ref_pron).split()
    pred_phones = re.sub(r'[0-9\-]', '', pred_pron).split()
    edit_dist = nltk.edit_distance(ref_phones, pred_phones)
    num_phones = len(ref_phones)
    phone_match = ref_phones == pred_phones
    ref_phone_pat = ' '.join(ref_phones)
    pred_phone_pat = ' '.join(pred_phones)

    # check whether the stress patterns match
    ref_stresses = re.sub(r'[^0-9\s]', '', ref_pron).split()
    pred_stresses = re.sub(r'[^0-9\s]', '', pred_pron).split()
    stress_match = ref_stresses == pred_stresses
    ref_stress_pat = ''.join(ref_stresses)
    pred_stress_pat = ''.join(pred_stresses)

    # check whether the syllable numbers match
    ref_syls = re.sub(r'[^\-\s]', '', ref_pron).split()
    pred_syls = re.sub(r'[^\-\s]', '', pred_pron).split()
    sylnum_match = ref_syls == pred_syls

    # if edit distance is 0, then compute the edit distance for pronunciation without lexical stress
    ref_phones_syls = re.sub(r'[0-9]', '', ref_pron).split()
    pred_phones_syls = re.sub(r'[0-9]', '', pred_pron).split()
    phones_syls_edit_dist = nltk.edit_distance(ref_phones_syls, pred_phones_syls) if edit_dist == 0 else None
    num_phones_syls = len(ref_phones_syls)

    return {'pron_match': pron_match,
            'edit_dist': edit_dist,
            'num_phones': num_phones,
            'phone_match': phone_match,
            'stress_match': stress_match,
            'sylnum_match': sylnum_match,
            'phones_syls_edit_dist': phones_syls_edit_dist,
            'num_phones_syls': num_phones_syls,
            'ref_stress_pat': ref_stress_pat,
            'pred_stress_pat': pred_stress_pat,
            'ref_phone_pat': ref_phone_pat,
            'pred_phone_pat': pred_phone_pat}


def get_alignment(ref, pred, src, row):
    """
    Get the alignment between the list of reference word pronunciations and the list of prediction
    word pronunciations. If the alignment indices do not match, then return None

    Args:
        ref (list): list of reference word pronunciations
        pred (list): list of prediction word pronunciations
        src (list): list of word strings
        row (int): row number

    Returns:
        alignment (list): if the alignment indices match, return a list of dict of aligned word pronunciations corresponding
            to that row. Otherwise, return None.

    """
    align_index = nltk.edit_distance_align(ref, pred)
    align_index = align_index[1:]  # remove [0, 0]

    assert len(ref) == len(pred) == len(src)
    for i, j in align_index:
        if i != j:
            return None

    alignment = [{'row': row,
                  'col': i,
                  'src': src[i - 1],
                  'ref': ref[i - 1],
                  'pred': pred[j - 1],
                  **compare_word_prons(ref[i - 1], pred[j - 1])} for i, j in align_index]
    return alignment


def get_token_error_rates(ref_fn, pred_fn):
    """
    Get the Token Error Rate (TER)

    Args:
        ref_fn (str): reference filename
        pred_fn (str): prediction filename

    Returns:
        TER (float): token error rate

    """
    total_edit_dist = 0
    total_tokens = 0

    with open(ref_fn) as ref_file:
        with open(pred_fn) as pred_file:
            refs = ref_file.readlines()
            preds = pred_file.readlines()

            assert len(preds) == len(refs)
            for ref, pred in zip(refs, preds):
                ref = ref.rstrip().split()
                pred = pred.rstrip().split()

                total_edit_dist += nltk.edit_distance(ref, pred)
                total_tokens += len(ref)

    TER = total_edit_dist / total_tokens
    return TER


def get_syl_error_rates(ref_fn, pred_fn):
    """
    Get the Syllable Error Rate (SER)

    Args:
        ref_fn (str): reference filename
        pred_fn (str): prediction filename

    Returns:
        SER (float): syllable error rate

    """
    total_edit_dist = 0
    total_syls = 0

    with open(ref_fn) as ref_file:
        with open(pred_fn) as pred_file:
            refs = ref_file.readlines()
            preds = pred_file.readlines()

            assert len(preds) == len(refs)
            for ref, pred in zip(refs, preds):
                ref = ref.rstrip().replace('_B', '+').replace('+', '-').split('-')[:-1]  # remove final empty string
                pred = pred.rstrip().replace('_B', '+').replace('+', '-').split('-')[:-1]  # remove final empty string

                # remove the leading and trailing blank spaces
                ref = list(map(str.strip, ref))
                pred = list(map(str.strip, pred))

                total_edit_dist += nltk.edit_distance(ref, pred)
                total_syls += len(ref)

    SER = total_edit_dist / total_syls
    return SER


def get_word_trans_error_rates(ref_fn, pred_fn, src_fn):
    """
    Get the Word Transcription Error Rate (WTER) and etc.

    Args:
        ref_fn (str): reference filename
        pred_fn (str): prediction filename
        src_fn (str): source filename

    Returns:
        WTER (float): word transcription error rate
        BER (float): prosodic boundary error rate
        total_align_error (int): total number of alignment errors
        lens_diff (int): total length differences for misaligned sentences
        total_sentences (int): total number of sentences
        total_words (int): total number of words
        total_word_bnds (int): total number of word boundaries
        flat_alignment (list): a flat list of dict of aligned word pronunciations

    """
    total_sentences = 0
    # word transcription error rate
    total_edit_dist = 0
    total_words = 0
    # prosodic boundary error rate
    total_bnd_edit_dist = 0
    total_word_bnds = 0

    flat_alignment = []
    total_align_error = 0
    lens_diff = 0

    with open(ref_fn) as ref_file:
        with open(pred_fn) as pred_file:
            with open(src_fn) as src_file:
                refs = ref_file.readlines()
                preds = pred_file.readlines()
                srcs = src_file.readlines()

                assert len(refs) == len(preds) == len(srcs)
                for row, (ref, pred, src) in enumerate(zip(refs, preds, srcs)):
                    ref_prons = ref.rstrip().replace('_B', '+').split('+')[:-1]  # remove final empty string
                    pred_prons = pred.rstrip().replace('_B', '+').split('+')[:-1]  # remove final empty string

                    # remove the leading and trailing blank spaces
                    ref_prons = list(map(str.strip, ref_prons))
                    pred_prons = list(map(str.strip, pred_prons))

                    words = src.rstrip().split()
                    ref_bnds = re.sub(r'[^+_B\s]', '', ref.rstrip()).split()
                    pred_bnds = re.sub(r'[^+_B\s]', '', pred.rstrip()).split()

                    # ensure reference pronunciation is of the same length as the words
                    if len(ref_prons) != len(words):
                        # print("Length not matching between ref and src \n ref: {} \n src: {}".format(ref_prons, words))
                        continue

                    total_sentences += 1
                    total_edit_dist += nltk.edit_distance(ref_prons, pred_prons)
                    total_words += len(ref_prons)

                    total_bnd_edit_dist += nltk.edit_distance(ref_bnds, pred_bnds)
                    total_word_bnds += len(ref_bnds)

                    # ensure reference pronunciation is of the same length as the predicted pronunciation
                    if len(ref_prons) != len(pred_prons):
                        total_align_error += 1
                        lens_diff += abs(len(ref_prons) - len(pred_prons))
                        continue

                    # ensure alignment is stepwise
                    alignment = get_alignment(ref_prons, pred_prons, words, row + 1)
                    if alignment is None:
                        total_align_error += 1
                        continue

                    flat_alignment += alignment

    WTER = total_edit_dist / total_words
    BER = total_bnd_edit_dist / total_word_bnds
    return WTER, BER, total_align_error, lens_diff, total_sentences, total_words, total_word_bnds, flat_alignment


def get_word_level_results(flat_alignment):
    """
    Get the overall word level results from all the aligned word pronunciations

    Args:
        flat_alignment (list): a flat list of dict of aligned word pronunciations

    Returns:
        PER (float): overall phoneme error rate
        word_acc (float): overall word accuracy
        stress_acc (float): overall stress accuracy
        syl_acc (float): overall syllable number accuracy
        PSER (float): overall phoneme-syllable error rate
        ref_stress_counter (Counter): a counter for reference stress patterns
        pred_stress_counter (Counter): a counter for predicted stress patterns

    """
    num_words = len(flat_alignment) + 1e-10
    total_phone_edit_dist = sum([result['edit_dist'] for result in flat_alignment])
    total_phones = sum([result['num_phones'] for result in flat_alignment])
    word_acc = sum([result['phone_match'] for result in flat_alignment]) / num_words
    stress_acc = sum([result['stress_match'] for result in flat_alignment]) / num_words
    syl_acc = sum([result['sylnum_match'] for result in flat_alignment]) / num_words
    total_phone_syl_edit_dist = sum([result['phones_syls_edit_dist'] for result in flat_alignment if result['phones_syls_edit_dist'] is not None])
    total_phones_syls = sum([result['num_phones_syls'] for result in flat_alignment if result['phones_syls_edit_dist'] is not None])

    # stress pattern counters
    ref_stress_counter = Counter([result['ref_stress_pat'] for result in flat_alignment])
    pred_stress_counter = Counter([result['pred_stress_pat'] for result in flat_alignment])

    PER = total_phone_edit_dist / (total_phones + 1e-10)
    PSER = total_phone_syl_edit_dist / (total_phones_syls + 1e-10)
    return PER, word_acc, stress_acc, syl_acc, PSER, ref_stress_counter, pred_stress_counter, total_phones


def counters2dfs(ref_stress_counter, pred_stress_counter):
    """
    Convert the stress pattern counters to a dataframe

    Args:
        ref_stress_counter (Counter): reference stress pattern counter
        pred_stress_counter (Counter): prediction stress pattern counter

    Returns:
        df (pd.DataFrame): a dataframe with two indices: pattern and is_zeros

    """
    ref_df = pd.DataFrame.from_dict(ref_stress_counter, orient='index', columns=['ref'])
    pred_df = pd.DataFrame.from_dict(pred_stress_counter, orient='index', columns=['pred'])
    df = ref_df.merge(pred_df, how='outer', left_index=True, right_index=True).fillna(0).astype(int)
    df = df.rename_axis('pattern')
    df['is_zeros'] = ~df.index.str.contains('1|2|3')  # indicate whether the pattern is all zeros
    df = df.set_index('is_zeros', append=True)  # set this column as index
    df = df.sort_values(by=['is_zeros', 'ref', 'pattern'], ascending=[False, False, True])
    return df


def get_all_error_rates(ref_fn, pred_fn, src_fn, plot=False):
    """
    Get all the Error Rates and information.

    Args:
        ref_fn (str): reference filename
        pred_fn (str): prediction filename
        src_fn (str): source filename

    Returns:
        _ (dict): a dictionary containing all the error rates and so on

    """
    TER = get_token_error_rates(ref_fn, pred_fn)
    SER = get_syl_error_rates(ref_fn, pred_fn)
    WTER, BER, total_align_error, lens_diff, total_sentences, total_words, total_word_bnds, word_align = get_word_trans_error_rates(ref_fn, pred_fn, src_fn)
    PER, word_acc, stress_acc, syl_acc, PSER, ref_stress_counter, pred_stress_counter, total_phones = get_word_level_results(word_align)

    print("Sentence level results")
    print("Total sentences:", total_sentences)
    print("Total words:", total_words)
    print("Total word boundaries:", total_word_bnds)

    print("Token Error Rate:", TER)
    print("Syllable Error Rate:", SER)

    print("Word Transcription Error Rate:", WTER)
    print("Prosodic Boundary Error Rate:", BER)
    print("Total align errors:", total_align_error)
    print("Sum of length difference:", lens_diff)

    print("\nWord level results")
    print("Total number of phones:", total_phones)
    print("Phoneme Error Rate (word-level):", PER)
    print("Word Accuracy (phoneme-only, word-level):", word_acc)
    print("Stress Accuracy (word-level):", stress_acc)
    print("Syllabification Accuracy (word-level):", syl_acc)
    print("Phoneme-Syllabification Error Rate (word-level):", PSER)

    # plot the dataframe
    if plot:
        df = counters2dfs(ref_stress_counter, pred_stress_counter)
        ax = df.plot(kind='bar', figsize=(20, 14), logy=True, fontsize='xx-large')
        ax.legend(fontsize='xx-large')
        ax.set_title(label='Stress pattern counts', fontsize='xx-large')
        ax.set_xlabel(xlabel='Stress pattern', fontsize='xx-large')
        ax.set_ylabel(ylabel='Counts (log)', fontsize='xx-large')

    return {'len': total_sentences,
            'TER': TER,
            'SER': SER,
            'WTER': WTER,
            'BER': BER,
            'align_error': total_align_error,
            'lens_diff': lens_diff,
            'PER': PER,
            'word_acc': word_acc,
            'stress_acc': stress_acc,
            'syl_acc': syl_acc,
            'PSER': PSER,
            'align': word_align}


def get_lexicon(lex_fpath):
    lex = defaultdict(list)
    with open(lex_fpath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pat = re.compile(r'^\("(.+)" (\(?[\w\s\|\-\$]*\)?) (\(.+\))$')
            m = pat.match(line)
            if m:
                word, pos, pron = m.group(1), m.group(2), m.group(3)
                lex[word].append((pos, pron))
    return lex


def get_vocab(fpath):
    vocab = {}
    with open(fpath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            word = line[0]
            freq = line[1]
            vocab[word] = freq
    return vocab


def analyze_ckpt(ckpt, lex, vocab, tgt_dir='', match_sort=False, verbose=True, remove_dup=False, src_fn='', stress_plot=False):
    """
    Analyze the checkpoint according to the four categories

    Args:
        ckpt (dict): a checkpoint dictionary containing the word alignments
        lex (dict): a dictionary of lexicon mapping from the word to a list of (pos, pronunciation)
        vocab (dict): a dictionary of vocabulary mapping from the word to the frequency
        tgt_dir (str): target dir name to save the analysis results
        match_sort (bool): whether sort the target files
        verbose (bool): whether be verbose when writing to the target files
        remove_dup (bool): whether remove the duplicated words when writing to the target files
        src_fn(str): whether create source files containing the sentence (for human evaluation)
        stress_plot (bool): whether plot the stress patterns for the four categories

    Returns:
        _ (dict): a dictionary containing the info about the iliv and ilov analysis results

    """
    if tgt_dir:
        os.makedirs(tgt_dir, exist_ok=True)

        iliv_bn = 'in_lex_in_vocab'
        ilov_bn = 'in_lex_out_vocab'
        oliv_bn = 'out_lex_in_vocab'
        olov_bn = 'out_lex_out_vocab'

        filiv = open(os.path.join(tgt_dir, iliv_bn + '.txt'), 'w')
        filov = open(os.path.join(tgt_dir, ilov_bn + '.txt'), 'w')
        foliv = open(os.path.join(tgt_dir, oliv_bn + '.txt'), 'w')
        folov = open(os.path.join(tgt_dir, olov_bn + '.txt'), 'w')

        if src_fn:
            foliv_src = open(os.path.join(tgt_dir, oliv_bn + '.src.txt'), 'w')
            folov_src = open(os.path.join(tgt_dir, olov_bn + '.src.txt'), 'w')
            with open(src_fn, 'r') as fs:
                src_lines = fs.readlines()

        if remove_dup:
            iliv_dict, ilov_dict, oliv_dict, olov_dict = {}, {}, {}, {}

    word_align = ckpt['align']
    align_error = ckpt['align_error']
    lens_diff = ckpt['lens_diff']

    total_iliv_match = 0
    total_iliv_miss = 0
    total_ilov_match = 0
    total_ilov_miss = 0
    total_oliv_match = 0
    total_oliv_miss = 0
    total_olov_match = 0
    total_olov_miss = 0

    align_iliv = []
    align_ilov = []
    align_oliv = []
    align_olov = []

    for alignment in word_align:
        row, col, src = alignment['row'], alignment['col'], alignment['src']
        ref, pred, match = alignment['ref'], alignment['pred'], alignment['pron_match']
        ref_phone, pred_phone, phone_match = alignment['ref_phone_pat'], alignment['pred_phone_pat'], alignment['phone_match']
        # ref_stress, pred_stress, stress_match = alignment['ref_stress_pat'], alignment['pred_stress_pat'], alignment['stress_match']

        src_in_lex = src in lex or src.lower() in lex or src.split("'")[0] in lex or src.split("'")[0].lower() in lex
        src_in_vocab = src in vocab or src.lower() in vocab
        src_freq = vocab[src] if src_in_vocab else 0

        total_iliv_match += 1 if src_in_lex and src_in_vocab and match else 0
        total_iliv_miss += 1 if src_in_lex and src_in_vocab and not match else 0
        total_ilov_match += 1 if src_in_lex and not src_in_vocab and match else 0
        total_ilov_miss += 1 if src_in_lex and not src_in_vocab and not match else 0
        total_oliv_match += 1 if not src_in_lex and src_in_vocab and match else 0
        total_oliv_miss += 1 if not src_in_lex and src_in_vocab and not match else 0
        total_olov_match += 1 if not src_in_lex and not src_in_vocab and match else 0
        total_olov_miss += 1 if not src_in_lex and not src_in_vocab and not match else 0

        if verbose:
            write_list = [src, str(match), str(phone_match), str(src_freq), str(row), str(col), ref, pred, ref_phone, pred_phone]
            # write_list = [src, str(match), str(stress_match), str(src_freq), str(row), str(col), ref, pred, ref_stress, pred_stress]
        else:
            write_list = [str(row), str(col), src, pred]

        if src_in_lex and src_in_vocab:
            align_iliv.append(alignment)
            if tgt_dir:
                if remove_dup:
                    if src in iliv_dict:
                        continue
                    else:
                        iliv_dict[src] = pred
                filiv.write('|'.join(write_list) + '\n')
        elif src_in_lex and not src_in_vocab:
            align_ilov.append(alignment)
            if tgt_dir:
                if remove_dup:
                    if src in ilov_dict:
                        continue
                    else:
                        ilov_dict[src] = pred
                filov.write('|'.join(write_list) + '\n')
        elif not src_in_lex and src_in_vocab:
            align_oliv.append(alignment)
            if tgt_dir:
                if remove_dup:
                    if src in oliv_dict:
                        continue
                    else:
                        oliv_dict[src] = pred
                foliv.write('|'.join(write_list) + '\n')
                if src_fn:
                    foliv_src.write(src_lines[row - 1])
        elif not src_in_lex and not src_in_vocab:
            align_olov.append(alignment)
            if tgt_dir:
                if remove_dup:
                    if src in olov_dict:
                        continue
                    else:
                        olov_dict[src] = pred
                folov.write('|'.join(write_list) + '\n')
                if src_fn:
                    folov_src.write(src_lines[row - 1])

    PER_iliv, word_acc_iliv, stress_acc_iliv, syl_acc_iliv, PSER_iliv, ref_stress_counter_iliv, pred_stress_counter_iliv, total_phones_iliv = get_word_level_results(align_iliv)
    PER_ilov, word_acc_ilov, stress_acc_ilov, syl_acc_ilov, PSER_ilov, ref_stress_counter_ilov, pred_stress_counter_ilov, total_phones_ilov = get_word_level_results(align_ilov)
    PER_oliv, word_acc_oliv, stress_acc_oliv, syl_acc_oliv, PSER_oliv, ref_stress_counter_oliv, pred_stress_counter_oliv, total_phones_oliv = get_word_level_results(align_oliv)
    PER_olov, word_acc_olov, stress_acc_olov, syl_acc_olov, PSER_olov, ref_stress_counter_olov, pred_stress_counter_olov, total_phones_olov = get_word_level_results(align_olov)

    PER_il, word_acc_il, stress_acc_il, syl_acc_il, PSER_il, ref_stress_counter_il, pred_stress_counter_il, total_phones_il = get_word_level_results(align_iliv + align_ilov)
    PER_ol, word_acc_ol, stress_acc_ol, syl_acc_ol, PSER_ol, ref_stress_counter_ol, pred_stress_counter_ol, total_phones_ol = get_word_level_results(align_oliv + align_olov)

    # four tasks
    total_iliv = total_iliv_match + total_iliv_miss
    total_ilov = total_ilov_match + total_ilov_miss
    total_oliv = total_oliv_match + total_oliv_miss
    total_olov = total_olov_match + total_olov_miss

    # lexicon
    total_in_lex = total_iliv + total_ilov
    total_out_lex = total_oliv + total_olov

    # vocabulary
    total_in_vocab = total_iliv + total_oliv
    total_out_vocab = total_ilov + total_olov

    # lexicon and match
    total_in_lex_match = total_iliv_match + total_ilov_match
    total_out_lex_match = total_oliv_match + total_olov_match

    # total words
    total = total_in_lex + total_out_lex

    print("\nAnalyze the checkpoint")
    print("# of words: ", total)
    print("in-lex {}, prop {:.8f}".format(total_in_lex, total_in_lex / total))
    print("out-lex {}, prop {:.8f}".format(total_out_lex, total_out_lex / total))
    print("in-vocab {}, prop {:.8f}".format(total_in_vocab, total_in_vocab / total))
    print("out-vocab {}, prop {:.8f}\n".format(total_out_vocab, total_out_vocab / total))

    print("1) in-lex in-vocab {}, match {}({:.8f})".format(total_iliv, total_iliv_match, total_iliv_match / total_iliv))
    print("PER {:.8f}(total={}), word ACC {:.8f}, stress ACC {:.8f}, syl ACC {:.8f}, PSER {:.8f}\n".format(PER_iliv, total_phones_iliv, word_acc_iliv, stress_acc_iliv, syl_acc_iliv, PSER_iliv))

    print("2) in-lex out-vocab {}, match {}({:.8f})".format(total_ilov, total_ilov_match, total_ilov_match / total_ilov))
    print("PER {:.8f}(total={}), word ACC {:.8f}, stress ACC {:.8f}, syl ACC {:.8f}, PSER {:.8f}\n".format(PER_ilov, total_phones_ilov, word_acc_ilov, stress_acc_ilov, syl_acc_ilov, PSER_ilov))

    print("-) in-lex match {}({:.8f})".format(total_in_lex_match, total_in_lex_match / total_in_lex))
    print("PER {:.8f}(total={}), word ACC {:.8f}, stress ACC {:.8f}, syl ACC {:.8f}, PSER {:.8f}\n".format(PER_il, total_phones_il, word_acc_il, stress_acc_il, syl_acc_il, PSER_il))

    print("3) out-lex in-vocab {}, match {}({:.8f})".format(total_oliv, total_oliv_match, total_oliv_match / (total_oliv + 1e-10)))
    print("PER {:.8f}(total={}), word ACC {:.8f}, stress ACC {:.8f}, syl ACC {:.8f}, PSER {:.8f}\n".format(PER_oliv, total_phones_oliv, word_acc_oliv, stress_acc_oliv, syl_acc_oliv, PSER_oliv))

    print("4) out-lex out-vocab {}, match {}({:.8f})".format(total_olov, total_olov_match, total_olov_match / (total_olov + 1e-10)))
    print("PER {:.8f}(total={}), word ACC {:.8f}, stress ACC {:.8f}, syl ACC {:.8f}, PSER {:.8f}\n".format(PER_olov, total_phones_olov, word_acc_olov, stress_acc_olov, syl_acc_olov, PSER_olov))

    print("-) out-lex match {}({:.8f})".format(total_out_lex_match, total_out_lex_match / (total_out_lex + 1e-10)))
    print("PER {:.8f}(total={}), word ACC {:.8f}, stress ACC {:.8f}, syl ACC {:.8f}, PSER {:.8f}\n".format(PER_ol, total_phones_ol, word_acc_ol, stress_acc_ol, syl_acc_ol, PSER_ol))

    print("total number of align-error sentences {}". format(align_error))
    print("sum of length difference {}\n". format(lens_diff))

    if tgt_dir:
        filiv.close()
        filov.close()
        foliv.close()
        folov.close()

        if src_fn:
            foliv_src.close()
            folov_src.close()

    if match_sort:
        os.system("sort -t '|' -k2,2 -k3,3 -k1,1 " + os.path.join(tgt_dir, iliv_bn + '.txt') + " > " + os.path.join(tgt_dir, iliv_bn + '.sort'))
        os.system("sort -t '|' -k2,2 -k3,3 -k1,1 " + os.path.join(tgt_dir, ilov_bn + '.txt') + " > " + os.path.join(tgt_dir, ilov_bn + '.sort'))
        os.system("sort -t '|' -k2,2 -k3,3 -k1,1 " + os.path.join(tgt_dir, oliv_bn + '.txt') + " > " + os.path.join(tgt_dir, oliv_bn + '.sort'))
        os.system("sort -t '|' -k2,2 -k3,3 -k1,1 " + os.path.join(tgt_dir, olov_bn + '.txt') + " > " + os.path.join(tgt_dir, olov_bn + '.sort'))

    if stress_plot:
        # plot the stress pattern bar plots
        df_iliv = counters2dfs(ref_stress_counter_iliv, pred_stress_counter_iliv)
        df_ilov = counters2dfs(ref_stress_counter_ilov, pred_stress_counter_ilov)
        df_oliv = counters2dfs(ref_stress_counter_oliv, pred_stress_counter_oliv)
        df_olov = counters2dfs(ref_stress_counter_olov, pred_stress_counter_olov)

        fig, axes = plt.subplots(nrows=2, ncols=2)
        if not df_iliv.empty:
            df_iliv.plot(kind='bar', figsize=(20, 14), ax=axes[0, 0], logy=True, title='in-lex in-vocab')
        if not df_ilov.empty:
            df_ilov.plot(kind='bar', figsize=(20, 14), ax=axes[0, 1], logy=True, title='in-lex out-vocab')
        if not df_oliv.empty:
            df_oliv.plot(kind='bar', figsize=(20, 14), ax=axes[1, 0], logy=True, title='out-lex in-vocab')
        if not df_olov.empty:
            df_olov.plot(kind='bar', figsize=(20, 14), ax=axes[1, 1], logy=True, title='out-lex out-vocab')
        fig.tight_layout()

    return {'iliv_match': total_iliv_match,
            'iliv_acc': total_iliv_match / total_iliv,
            'ilov_match': total_ilov_match,
            'ilov_acc': total_ilov_match / total_ilov,
            'PER_ilov': PER_ilov,
            'word_acc_ilov': word_acc_ilov,
            'stress_acc_ilov': stress_acc_ilov,
            'syl_acc_ilov': syl_acc_ilov,
            'PSER_ilov': PSER_ilov}


def pos_counters2dfs(match_pos_counter, miss_pos_counter):
    """
    Convert the pos counters to a dataframe

    Args:
        match_pos_counter (Counter): match pos counter
        miss_pos_counter (Counter): miss pos counter

    Returns:
        df (pd.DataFrame): a dataframe with one index: pos

    """
    match_df = pd.DataFrame.from_dict(match_pos_counter, orient='index', columns=['match'])
    miss_df = pd.DataFrame.from_dict(miss_pos_counter, orient='index', columns=['miss'])
    df = match_df.merge(miss_df, how='outer', left_index=True, right_index=True).fillna(0).astype(int)
    df = df.rename_axis('POS')
    df = df.assign(total=df['match'] + df['miss']).sort_values('total', ascending=[False]).drop(columns='total')
    # df = df.sort_values(by=['miss'], ascending=[False])
    return df


def analyze_ilov_ckpt(ckpt, lex, vocab, ilov_vocab, tgt_dir='', match_sort=False, verbose=True, remove_dup=False, src_fn='', stress_plot=False, ilov_plot=False):
    """
    Analyze the checkpoint according to the four categories

    Args:
        ckpt (dict): a checkpoint dictionary containing the word alignments
        lex (dict): a dictionary of lexicon mapping from the word to a list of (pos, pronunciation)
        vocab (dict): a dictionary of vocabulary mapping from the word to the frequency
        ilov_vocab (dict): a dictionary of ilov vocabulary mapping from the word to the frequency
        tgt_dir (str): target dir name to save the analysis results
        match_sort (bool): whether sort the target files
        verbose (bool): whether be verbose when writing to the target files
        remove_dup (bool): whether remove the duplicated words when writing to the target files
        src_fn(str): whether create source files containing the sentence (for human evaluation)
        stress_plot (bool): whether plot the stress patterns for the four categories
        ilov_plot (bool): whether plot the acc barplot for the ilov words

    Returns:
        _ (dict): a dictionary containing the info about the iliv and ilov analysis results

    """
    if tgt_dir:
        os.makedirs(tgt_dir, exist_ok=True)

        iliv_bn = 'in_lex_in_vocab'
        ilov_bn = 'in_lex_out_vocab'
        oliv_bn = 'out_lex_in_vocab'
        olov_bn = 'out_lex_out_vocab'

        filiv = open(os.path.join(tgt_dir, iliv_bn + '.txt'), 'w')
        filov = open(os.path.join(tgt_dir, ilov_bn + '.txt'), 'w')
        foliv = open(os.path.join(tgt_dir, oliv_bn + '.txt'), 'w')
        folov = open(os.path.join(tgt_dir, olov_bn + '.txt'), 'w')

        if src_fn:
            foliv_src = open(os.path.join(tgt_dir, oliv_bn + '.src.txt'), 'w')
            folov_src = open(os.path.join(tgt_dir, olov_bn + '.src.txt'), 'w')
            with open(src_fn, 'r') as fs:
                src_lines = fs.readlines()

        if remove_dup:
            iliv_dict, ilov_dict, oliv_dict, olov_dict = {}, {}, {}, {}

    word_align = ckpt['align']
    align_error = ckpt['align_error']
    lens_diff = ckpt['lens_diff']

    total_iliv_match = 0
    total_iliv_miss = 0
    total_ilov_match = 0
    total_ilov_miss = 0
    total_oliv_match = 0
    total_oliv_miss = 0
    total_olov_match = 0
    total_olov_miss = 0

    align_iliv = []
    align_ilov = []
    align_oliv = []
    align_olov = []

    total_ilov_1_match, total_ilov_1_miss = 0, 0
    total_ilov_2_match, total_ilov_2_miss = 0, 0
    total_ilov_3_match, total_ilov_3_miss = 0, 0
    total_ilov_4_match, total_ilov_4_miss = 0, 0
    total_ilov_5_match, total_ilov_5_miss = 0, 0
    total_ilov_6_match, total_ilov_6_miss = 0, 0
    total_ilov_7p_match, total_ilov_7p_miss = 0, 0

    total_ilov_nnp_match, total_ilov_nnp_miss = 0, 0
    total_ilov_non_nnp_match, total_ilov_non_nnp_miss = 0, 0

    if ilov_plot:
        ilov_1_match_pos, ilov_1_miss_pos = [], []
        ilov_2_match_pos, ilov_2_miss_pos = [], []
        ilov_3_match_pos, ilov_3_miss_pos = [], []
        ilov_4_match_pos, ilov_4_miss_pos = [], []
        ilov_5_match_pos, ilov_5_miss_pos = [], []
        ilov_6_match_pos, ilov_6_miss_pos = [], []
        ilov_7p_match_pos, ilov_7p_miss_pos = [], []

    for alignment in word_align:
        row, col, src = alignment['row'], alignment['col'], alignment['src']
        ref, pred, match = alignment['ref'], alignment['pred'], alignment['pron_match']
        ref_phone, pred_phone, phone_match = alignment['ref_phone_pat'], alignment['pred_phone_pat'], alignment['phone_match']
        # ref_stress, pred_stress, stress_match = alignment['ref_stress_pat'], alignment['pred_stress_pat'], alignment['stress_match']

        src_in_lex = src in lex or src.lower() in lex or src.split("'")[0] in lex or src.split("'")[0].lower() in lex
        src_in_vocab = src in vocab or src.lower() in vocab
        src_freq = vocab[src] if src_in_vocab else 0

        total_iliv_match += 1 if src_in_lex and src_in_vocab and match else 0
        total_iliv_miss += 1 if src_in_lex and src_in_vocab and not match else 0
        total_ilov_match += 1 if src_in_lex and not src_in_vocab and match else 0
        total_ilov_miss += 1 if src_in_lex and not src_in_vocab and not match else 0
        total_oliv_match += 1 if not src_in_lex and src_in_vocab and match else 0
        total_oliv_miss += 1 if not src_in_lex and src_in_vocab and not match else 0
        total_olov_match += 1 if not src_in_lex and not src_in_vocab and match else 0
        total_olov_miss += 1 if not src_in_lex and not src_in_vocab and not match else 0

        if src_in_lex and not src_in_vocab:
            ilov_freq = int(ilov_vocab[src]) if src in ilov_vocab else 0
            total_ilov_1_match += 1 if ilov_freq == 1 and match else 0
            total_ilov_1_miss += 1 if ilov_freq == 1 and not match else 0
            total_ilov_2_match += 1 if ilov_freq == 2 and match else 0
            total_ilov_2_miss += 1 if ilov_freq == 2 and not match else 0
            total_ilov_3_match += 1 if ilov_freq == 3 and match else 0
            total_ilov_3_miss += 1 if ilov_freq == 3 and not match else 0
            total_ilov_4_match += 1 if ilov_freq == 4 and match else 0
            total_ilov_4_miss += 1 if ilov_freq == 4 and not match else 0
            total_ilov_5_match += 1 if ilov_freq == 5 and match else 0
            total_ilov_5_miss += 1 if ilov_freq == 5 and not match else 0
            total_ilov_6_match += 1 if ilov_freq == 6 and match else 0
            total_ilov_6_miss += 1 if ilov_freq == 6 and not match else 0
            total_ilov_7p_match += 1 if ilov_freq >= 7 and match else 0
            total_ilov_7p_miss += 1 if ilov_freq >= 7 and not match else 0

            lex_src = None
            if src in lex:
                lex_src = src
            elif src.lower() in lex:
                lex_src = src.lower()
            elif src.split("'")[0] in lex:
                lex_src = src.split("'")[0]
            elif src.split("'")[0].lower() in lex:
                lex_src = src.split("'")[0].lower()

            src_is_nnp = False
            for pos, pron in lex[lex_src]:
                if 'nnp' in pos:
                    src_is_nnp = True
            total_ilov_nnp_match += 1 if src_is_nnp and match else 0
            total_ilov_nnp_miss += 1 if src_is_nnp and not match else 0
            total_ilov_non_nnp_match += 1 if not src_is_nnp and match else 0
            total_ilov_non_nnp_miss += 1 if not src_is_nnp and not match else 0

            if ilov_plot:
                random_pos = random.choice(lex[lex_src])[0]
                if ilov_freq == 1:
                    if match:
                        ilov_1_match_pos.append(random_pos)
                    else:
                        ilov_1_miss_pos.append(random_pos)
                elif ilov_freq == 2:
                    if match:
                        ilov_2_match_pos.append(random_pos)
                    else:
                        ilov_2_miss_pos.append(random_pos)
                elif ilov_freq == 3:
                    if match:
                        ilov_3_match_pos.append(random_pos)
                    else:
                        ilov_3_miss_pos.append(random_pos)
                elif ilov_freq == 4:
                    if match:
                        ilov_4_match_pos.append(random_pos)
                    else:
                        ilov_4_miss_pos.append(random_pos)
                elif ilov_freq == 5:
                    if match:
                        ilov_5_match_pos.append(random_pos)
                    else:
                        ilov_5_miss_pos.append(random_pos)
                elif ilov_freq == 6:
                    if match:
                        ilov_6_match_pos.append(random_pos)
                    else:
                        ilov_6_miss_pos.append(random_pos)
                elif ilov_freq >= 7:
                    if match:
                        ilov_7p_match_pos.append(random_pos)
                    else:
                        ilov_7p_miss_pos.append(random_pos)

        if verbose:
            write_list = [src, str(match), str(phone_match), str(src_freq), str(row), str(col), ref, pred, ref_phone, pred_phone]
            # write_list = [src, str(match), str(stress_match), str(src_freq), str(row), str(col), ref, pred, ref_stress, pred_stress]
        else:
            write_list = [str(row), str(col), src, pred]

        if src_in_lex and src_in_vocab:
            align_iliv.append(alignment)
            if tgt_dir:
                if remove_dup:
                    if src in iliv_dict:
                        continue
                    else:
                        iliv_dict[src] = pred
                filiv.write('|'.join(write_list) + '\n')
        elif src_in_lex and not src_in_vocab:
            align_ilov.append(alignment)
            if tgt_dir:
                if remove_dup:
                    if src in ilov_dict:
                        continue
                    else:
                        ilov_dict[src] = pred
                filov.write('|'.join(write_list) + '\n')
        elif not src_in_lex and src_in_vocab:
            align_oliv.append(alignment)
            if tgt_dir:
                if remove_dup:
                    if src in oliv_dict:
                        continue
                    else:
                        oliv_dict[src] = pred
                foliv.write('|'.join(write_list) + '\n')
                if src_fn:
                    foliv_src.write(src_lines[row - 1])
        elif not src_in_lex and not src_in_vocab:
            align_olov.append(alignment)
            if tgt_dir:
                if remove_dup:
                    if src in olov_dict:
                        continue
                    else:
                        olov_dict[src] = pred
                folov.write('|'.join(write_list) + '\n')
                if src_fn:
                    folov_src.write(src_lines[row - 1])

    PER_iliv, word_acc_iliv, stress_acc_iliv, syl_acc_iliv, PSER_iliv, ref_stress_counter_iliv, pred_stress_counter_iliv, total_phones_iliv = get_word_level_results(align_iliv)
    PER_ilov, word_acc_ilov, stress_acc_ilov, syl_acc_ilov, PSER_ilov, ref_stress_counter_ilov, pred_stress_counter_ilov, total_phones_ilov = get_word_level_results(align_ilov)
    PER_oliv, word_acc_oliv, stress_acc_oliv, syl_acc_oliv, PSER_oliv, ref_stress_counter_oliv, pred_stress_counter_oliv, total_phones_oliv = get_word_level_results(align_oliv)
    PER_olov, word_acc_olov, stress_acc_olov, syl_acc_olov, PSER_olov, ref_stress_counter_olov, pred_stress_counter_olov, total_phones_olov = get_word_level_results(align_olov)

    PER_il, word_acc_il, stress_acc_il, syl_acc_il, PSER_il, ref_stress_counter_il, pred_stress_counter_il, total_phones_il = get_word_level_results(align_iliv + align_ilov)
    PER_ol, word_acc_ol, stress_acc_ol, syl_acc_ol, PSER_ol, ref_stress_counter_ol, pred_stress_counter_ol, total_phones_ol = get_word_level_results(align_oliv + align_olov)

    # four tasks
    total_iliv = total_iliv_match + total_iliv_miss
    total_ilov = total_ilov_match + total_ilov_miss
    total_oliv = total_oliv_match + total_oliv_miss
    total_olov = total_olov_match + total_olov_miss

    # lexicon
    total_in_lex = total_iliv + total_ilov
    total_out_lex = total_oliv + total_olov

    # vocabulary
    total_in_vocab = total_iliv + total_oliv
    total_out_vocab = total_ilov + total_olov

    # lexicon and match
    total_in_lex_match = total_iliv_match + total_ilov_match
    total_out_lex_match = total_oliv_match + total_olov_match

    # total words
    total = total_in_lex + total_out_lex

    print("\nAnalyze the checkpoint")
    print("# of words: ", total)
    print("in-lex {}, prop {:.8f}".format(total_in_lex, total_in_lex / total))
    print("out-lex {}, prop {:.8f}".format(total_out_lex, total_out_lex / total))
    print("in-vocab {}, prop {:.8f}".format(total_in_vocab, total_in_vocab / total))
    print("out-vocab {}, prop {:.8f}\n".format(total_out_vocab, total_out_vocab / total))

    print("1) in-lex in-vocab {}, match {}({:.8f})".format(total_iliv, total_iliv_match, total_iliv_match / total_iliv))
    print("PER {:.8f}(total={}), word ACC {:.8f}, stress ACC {:.8f}, syl ACC {:.8f}, PSER {:.8f}\n".format(PER_iliv, total_phones_iliv, word_acc_iliv, stress_acc_iliv, syl_acc_iliv, PSER_iliv))

    print("2) in-lex out-vocab {}, match {}({:.8f})".format(total_ilov, total_ilov_match, total_ilov_match / total_ilov))
    print("PER {:.8f}(total={}), word ACC {:.8f}, stress ACC {:.8f}, syl ACC {:.8f}, PSER {:.8f}\n".format(PER_ilov, total_phones_ilov, word_acc_ilov, stress_acc_ilov, syl_acc_ilov, PSER_ilov))

    print("-) in-lex match {}({:.8f})".format(total_in_lex_match, total_in_lex_match / total_in_lex))
    print("PER {:.8f}(total={}), word ACC {:.8f}, stress ACC {:.8f}, syl ACC {:.8f}, PSER {:.8f}\n".format(PER_il, total_phones_il, word_acc_il, stress_acc_il, syl_acc_il, PSER_il))

    print("3) out-lex in-vocab {}, match {}({:.8f})".format(total_oliv, total_oliv_match, total_oliv_match / (total_oliv + 1e-10)))
    print("PER {:.8f}(total={}), word ACC {:.8f}, stress ACC {:.8f}, syl ACC {:.8f}, PSER {:.8f}\n".format(PER_oliv, total_phones_oliv, word_acc_oliv, stress_acc_oliv, syl_acc_oliv, PSER_oliv))

    print("4) out-lex out-vocab {}, match {}({:.8f})".format(total_olov, total_olov_match, total_olov_match / (total_olov + 1e-10)))
    print("PER {:.8f}(total={}), word ACC {:.8f}, stress ACC {:.8f}, syl ACC {:.8f}, PSER {:.8f}\n".format(PER_olov, total_phones_olov, word_acc_olov, stress_acc_olov, syl_acc_olov, PSER_olov))

    print("-) out-lex match {}({:.8f})".format(total_out_lex_match, total_out_lex_match / (total_out_lex + 1e-10)))
    print("PER {:.8f}(total={}), word ACC {:.8f}, stress ACC {:.8f}, syl ACC {:.8f}, PSER {:.8f}\n".format(PER_ol, total_phones_ol, word_acc_ol, stress_acc_ol, syl_acc_ol, PSER_ol))

    print("total number of align-error sentences {}". format(align_error))
    print("sum of length difference {}\n". format(lens_diff))

    ilov_acc_dict = {}
    ilov_acc_dict['1'] = total_ilov_1_match / (total_ilov_1_match + total_ilov_1_miss + 1e-10)
    ilov_acc_dict['2'] = total_ilov_2_match / (total_ilov_2_match + total_ilov_2_miss + 1e-10)
    ilov_acc_dict['3'] = total_ilov_3_match / (total_ilov_3_match + total_ilov_3_miss + 1e-10)
    ilov_acc_dict['4'] = total_ilov_4_match / (total_ilov_4_match + total_ilov_4_miss + 1e-10)
    ilov_acc_dict['5'] = total_ilov_5_match / (total_ilov_5_match + total_ilov_5_miss + 1e-10)
    ilov_acc_dict['6'] = total_ilov_6_match / (total_ilov_6_match + total_ilov_6_miss + 1e-10)
    ilov_acc_dict['7+'] = total_ilov_7p_match / (total_ilov_7p_match + total_ilov_7p_miss + 1e-10)
    ilov_acc_dict['nnp'] = total_ilov_nnp_match / (total_ilov_nnp_match + total_ilov_nnp_miss + 1e-10)
    ilov_acc_dict['non-nnp'] = total_ilov_non_nnp_match / (total_ilov_non_nnp_match + total_ilov_non_nnp_miss + 1e-10)

    print("-) in-lex out-vocab #==1 {}, match {}({:.8f})".format(total_ilov_1_match + total_ilov_1_miss, total_ilov_1_match, ilov_acc_dict['1']))
    print("-) in-lex out-vocab #==2 {}, match {}({:.8f})".format(total_ilov_2_match + total_ilov_2_miss, total_ilov_2_match, ilov_acc_dict['2']))
    print("-) in-lex out-vocab #==3 {}, match {}({:.8f})".format(total_ilov_3_match + total_ilov_3_miss, total_ilov_3_match, ilov_acc_dict['3']))
    print("-) in-lex out-vocab #==4 {}, match {}({:.8f})".format(total_ilov_4_match + total_ilov_4_miss, total_ilov_4_match, ilov_acc_dict['4']))
    print("-) in-lex out-vocab #==5 {}, match {}({:.8f})".format(total_ilov_5_match + total_ilov_5_miss, total_ilov_5_match, ilov_acc_dict['5']))
    print("-) in-lex out-vocab #==6 {}, match {}({:.8f})".format(total_ilov_6_match + total_ilov_6_miss, total_ilov_6_match, ilov_acc_dict['6']))
    print("-) in-lex out-vocab #>=7 {}, match {}({:.8f})".format(total_ilov_7p_match + total_ilov_7p_miss, total_ilov_7p_match, ilov_acc_dict['7+']))

    print("-) in-lex out-vocab POS==NNP {}, match {}({:.8f})".format(total_ilov_nnp_match + total_ilov_nnp_miss, total_ilov_nnp_match, ilov_acc_dict['nnp']))
    print("-) in-lex out-vocab POS==Non-NNP {}, match {}({:.8f})".format(total_ilov_non_nnp_match + total_ilov_non_nnp_miss, total_ilov_non_nnp_match, ilov_acc_dict['non-nnp']))

    if tgt_dir:
        filiv.close()
        filov.close()
        foliv.close()
        folov.close()

        if src_fn:
            foliv_src.close()
            folov_src.close()

    if match_sort:
        os.system("sort -t '|' -k2,2 -k3,3 -k1,1 " + os.path.join(tgt_dir, iliv_bn + '.txt') + " > " + os.path.join(tgt_dir, iliv_bn + '.sort'))
        os.system("sort -t '|' -k2,2 -k3,3 -k1,1 " + os.path.join(tgt_dir, ilov_bn + '.txt') + " > " + os.path.join(tgt_dir, ilov_bn + '.sort'))
        os.system("sort -t '|' -k2,2 -k3,3 -k1,1 " + os.path.join(tgt_dir, oliv_bn + '.txt') + " > " + os.path.join(tgt_dir, oliv_bn + '.sort'))
        os.system("sort -t '|' -k2,2 -k3,3 -k1,1 " + os.path.join(tgt_dir, olov_bn + '.txt') + " > " + os.path.join(tgt_dir, olov_bn + '.sort'))

    if stress_plot:
        # plot the stress pattern bar plots
        df_iliv = counters2dfs(ref_stress_counter_iliv, pred_stress_counter_iliv)
        df_ilov = counters2dfs(ref_stress_counter_ilov, pred_stress_counter_ilov)
        df_oliv = counters2dfs(ref_stress_counter_oliv, pred_stress_counter_oliv)
        df_olov = counters2dfs(ref_stress_counter_olov, pred_stress_counter_olov)

        fig, axes = plt.subplots(nrows=2, ncols=2)
        if not df_iliv.empty:
            df_iliv.plot(kind='bar', figsize=(20, 14), ax=axes[0, 0], logy=True, title='in-lex in-vocab')
        if not df_ilov.empty:
            df_ilov.plot(kind='bar', figsize=(20, 14), ax=axes[0, 1], logy=True, title='in-lex out-vocab')
        if not df_oliv.empty:
            df_oliv.plot(kind='bar', figsize=(20, 14), ax=axes[1, 0], logy=True, title='out-lex in-vocab')
        if not df_olov.empty:
            df_olov.plot(kind='bar', figsize=(20, 14), ax=axes[1, 1], logy=True, title='out-lex out-vocab')
        fig.tight_layout()

    if ilov_plot:
        df_ilov_acc = pd.DataFrame.from_dict(ilov_acc_dict, orient='index', columns=['acc'])
        df_ilov_acc = df_ilov_acc.rename_axis('freq')
        df_ilov_acc = df_ilov_acc.sort_values(by=['freq'], ascending=[True])

        df_ilov_1_pos = pos_counters2dfs(Counter(ilov_1_match_pos), Counter(ilov_1_miss_pos))
        df_ilov_2_pos = pos_counters2dfs(Counter(ilov_2_match_pos), Counter(ilov_2_miss_pos))
        df_ilov_3_pos = pos_counters2dfs(Counter(ilov_3_match_pos), Counter(ilov_3_miss_pos))
        df_ilov_4_pos = pos_counters2dfs(Counter(ilov_4_match_pos), Counter(ilov_4_miss_pos))
        df_ilov_5_pos = pos_counters2dfs(Counter(ilov_5_match_pos), Counter(ilov_5_miss_pos))
        df_ilov_6_pos = pos_counters2dfs(Counter(ilov_6_match_pos), Counter(ilov_6_miss_pos))
        df_ilov_7p_pos = pos_counters2dfs(Counter(ilov_7p_match_pos), Counter(ilov_7p_miss_pos))

        fig, axes = plt.subplots(nrows=4, ncols=2)
        if not df_ilov_acc.empty:
            acc_max = df_ilov_acc.to_numpy().max()
            acc_min = df_ilov_acc.to_numpy().min()
            df_ilov_acc.plot(
                kind='barh', figsize=(20, 14), ax=axes[0, 0], title='ilov acc',
                xlim=(max(0.0, acc_min - 0.5 * (acc_max - acc_min)), acc_max + 0.5 * (acc_max - acc_min)),
                color='red')
        if not df_ilov_1_pos.empty:
            df_ilov_1_pos.plot(kind='barh', figsize=(20, 14), ax=axes[0, 1], title='ilov 1 pos', stacked=True)
        if not df_ilov_2_pos.empty:
            df_ilov_2_pos.plot(kind='barh', figsize=(20, 14), ax=axes[1, 0], title='ilov 2 pos', stacked=True)
        if not df_ilov_3_pos.empty:
            df_ilov_3_pos.plot(kind='barh', figsize=(20, 14), ax=axes[1, 1], title='ilov 3 pos', stacked=True)
        if not df_ilov_4_pos.empty:
            df_ilov_4_pos.plot(kind='barh', figsize=(20, 14), ax=axes[2, 0], title='ilov 4 pos', stacked=True)
        if not df_ilov_5_pos.empty:
            df_ilov_5_pos.plot(kind='barh', figsize=(20, 14), ax=axes[2, 1], title='ilov 5 pos', stacked=True)
        if not df_ilov_6_pos.empty:
            df_ilov_6_pos.plot(kind='barh', figsize=(20, 14), ax=axes[3, 0], title='ilov 6 pos', stacked=True)
        if not df_ilov_7p_pos.empty:
            df_ilov_7p_pos.plot(kind='barh', figsize=(20, 14), ax=axes[3, 1], title='ilov 7+ pos', stacked=True)
        fig.tight_layout()

    return {'iliv_match': total_iliv_match,
            'iliv_acc': total_iliv_match / total_iliv,
            'ilov_match': total_ilov_match,
            'ilov_acc': total_ilov_match / total_ilov,
            'ilov_1_acc': ilov_acc_dict['1'],
            'ilov_2_acc': ilov_acc_dict['2'],
            'ilov_3_acc': ilov_acc_dict['3'],
            'ilov_4_acc': ilov_acc_dict['4'],
            'ilov_5_acc': ilov_acc_dict['5'],
            'ilov_6_acc': ilov_acc_dict['6'],
            'ilov_7+_acc': ilov_acc_dict['7+'],
            'ilov_nnp_acc': ilov_acc_dict['nnp'],
            'ilov_non-nnp_acc': ilov_acc_dict['non-nnp']}


def get_mean_and_std(ckpts, key):
    value_list = [ckpt_anly[key] for ckpt_anly in ckpts.values()]
    mean = st.mean(value_list)
    std = st.stdev(value_list)
    print("key: {}\t # = {}\t mean = {:.8f}\t std = {:.8f}".format(key, len(value_list), mean, std))
    return mean, std
