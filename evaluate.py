import torch
from utils.lang_utils import gen_word2idx_vec_rep
from utils.model_utils import device
from metrics import *
from utils.lang_utils import make_std_mask, subsequent_mask


def _compute_ppl(probs, batch, loss_compute):
    trg = batch.trg_y
    probs = probs[:min(trg.shape[1], probs.shape[0]), :]
    trg = trg[:, :min(trg.shape[1], probs.shape[0])]
    loss = loss_compute.loss_from_probs(probs, trg, batch.ntokens)
    ppl = math.exp(loss)
    return ppl


def beam_decode_indices(model, batch, start_idx, start_token, i2w,
                        max_len, loss_compute=None, gamma=0.0,
                        beam_width=4, eos_idx=3, len_norm=0.0, topk=1, pad_idx=0):
    if batch.src.shape[0] != 1:
        raise NotImplementedError("For now only support batch size 1")
    model = model.to(device())
    ys = torch.ones(1, 1).fill_(start_idx).type(torch.LongTensor).to(device())
    ys = ys.repeat(batch.src_mask.shape[0], 1)
    src = torch.cat([batch.ctx, batch.src], dim=1)
    src_mask = torch.cat([batch.ctx_mask, batch.src_mask], dim=2)
    src = src.to(device())
    src_mask = src_mask.to(device())
    memory = model.encode(src, src_mask)
    curr_candidates = [
        (ys,
        0.0,
        [],
        [start_token])
    ]
    final_ans = []
    for i in range(max_len):
        next_candidates = []
        for tup in curr_candidates:
            tgt = tup[0]
            score = tup[1]
            prev_prob_list = [t for t in tup[2]]
            words = [t for t in tup[3]]
            decoder_out = model.tf_decode(memory, tgt, src_mask, make_std_mask(tgt, pad=pad_idx))
            prob, vt, it = model.predict(decoder_out[:,-1].unsqueeze(1), topk=beam_width)
            prev_prob_list.append(prob)
            for idx in range(vt.shape[2]): # 1 x beam_width
                div_penalty = 0.0
                if i > 0: div_penalty = gamma * (idx+1)
                val = vt[0,0,idx].item()
                wi = it[0,0,idx].item()
                word = i2w[wi]
                new_score = score + val - div_penalty
                new_tgt = torch.cat([tgt, torch.ones(1, 1).type_as(src.data).fill_(wi)], dim=1)
                new_words = [w for w in words]
                new_words.append(word)
                if wi == eos_idx or i == max_len - 1:
                    if len_norm > 0:
                        length_penalty = (len_norm + new_tgt.shape[1]) / (len_norm + 1)
                        new_score /= length_penalty ** len_norm
                    else:
                        new_score = new_score / new_tgt.shape[1] if new_tgt.shape[1] > 0 else new_score
                    ppl = 0
                    if loss_compute is not None and batch is not None:
                        probs = torch.cat(prev_prob_list, dim=1).type(torch.FloatTensor).to(device())
                        ppl = _compute_ppl(probs, batch, loss_compute)
                    final_ans.append((new_tgt, new_score, ppl, new_words))
                else:
                    next_candidates.append((new_tgt, new_score, prev_prob_list, new_words))
        next_candidates = sorted(next_candidates, key=lambda t: t[1], reverse=True)
        next_candidates = next_candidates[:min(len(next_candidates), beam_width)]
        curr_candidates = next_candidates
    final_ans = sorted(final_ans, key=lambda t: t[1], reverse=True)
    final_ans = final_ans[:min(len(final_ans), beam_width)]
    return final_ans[:topk]


def greedy_decode_indices(model, src, src_mask, max_len, start_symbol, batch=None, loss_compute=None):
    model = model.to(device())
    src = src.to(device())
    src_mask = src_mask.to(device())
    memory = model.encode(src, src_mask)
    out = model.regressive_decode(memory, src_mask, start_symbol, max_len)
    probs = model.generator(out).squeeze(0)
    _, wis = torch.max(probs, dim=1)
    ppl = 0
    if loss_compute is not None and batch is not None:
        # out = model.forward(src, batch.trg, src_mask, batch.trg_mask)
        # probs = model.generator(out).squeeze(0)
        ppl = _compute_ppl(probs, batch, loss_compute)
    return wis, ppl


def multi_turn_greedy_decode_indices(model, src, src_mask, max_rsp_len, start_symbol, trg=None, loss_compute=None, ntokens=None):
    # TODO: make it work for batch size more than 1
    model = model.to(device())
    src = src.to(device())
    src_mask = src_mask.to(device())
    probs = []
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    dlg_ctx = model.update_dialog_context(memory, src_mask)
    d_mask = model.dialog_context.get_flat_seq_mask()
    out = None
    for i in range(max_rsp_len - 1):
        subseq_mask = subsequent_mask(ys.size(1))
        out = model.decode(memory, src_mask, ys, subseq_mask, dlg_ctx, d_mask)
        prob = model.generator(out[:, -1]) #TODO: add ppl
        probs.append(prob)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    out_seq_mask = subsequent_mask(ys.size(1)).type(torch.ByteTensor)[:,-1,:].unsqueeze(1)
    model.update_dialog_context(out, out_seq_mask, save_only=True)
    probs = torch.cat(probs, dim=0)
    ppl = 0
    if loss_compute is not None and trg is not None and ntokens is not None:
        trg = trg[:, :-1]
        probs = probs[:min(trg.shape[1], probs.shape[0]), :]
        trg = trg[:, :min(trg.shape[1], probs.shape[0])]
        loss = loss_compute.loss_from_probs(probs, trg, ntokens)
        ppl = math.exp(loss)
    return ys, ppl


def multi_turn_truth_forward(model, src, src_mask, tgt, tgt_mask):
    out = model.forward(src, tgt, src_mask, tgt_mask)
    probs = model.generator(out)
    _, indices = torch.max(probs, dim=2)
    return indices


def multi_turn_truth_rsp_from_batch(model, batch, params, word_delim=" "):
    pred_indices = multi_turn_truth_forward(model, batch.src, batch.src_mask, batch.trg, batch.trg_mask)
    rsp_lst = []
    if pred_indices.shape[0] == 1:
        pred_indices = pred_indices.squeeze(0)
    for idx in pred_indices:
        idx = idx.cpu().item()
        if idx == params.eos_idx:
            break
        if idx != params.sos_idx and idx != params.pad_idx:
            rsp_lst.append(params.i2w[idx])
    return word_delim.join(rsp_lst)


def gen_response_from_batch(model, batch, rsp_len, params, word_delim=" ", loss_compute=None):
    pred_indices, ppl = greedy_decode_indices(model, batch.src, batch.src_mask, rsp_len, params.sos_idx,
                                              batch=batch, loss_compute=loss_compute)
    rsp_lst = []
    if pred_indices.shape[0] == 1:
        pred_indices = pred_indices.squeeze(0)
    for idx in pred_indices:
        idx = idx.cpu().item()
        if idx == params.eos_idx:
            break
        if idx != params.sos_idx and idx != params.pad_idx:
            rsp_lst.append(params.i2w[idx])
    return word_delim.join(rsp_lst), ppl


def gen_multi_turn_response_from_batch(model, batch, rsp_len, params, word_delim=" ", loss_compute=None):
    pred_indices, ppl = multi_turn_greedy_decode_indices(model, batch.src, batch.src_mask, rsp_len, params.sos_idx,
                                                         trg=batch.trg_y, loss_compute=loss_compute, ntokens=batch.ntokens)
    rsp_lst = []
    if pred_indices.shape[0] == 1:
        pred_indices = pred_indices.squeeze(0)
    for idx in pred_indices:
        idx = idx.cpu().item()
        if idx == params.eos_idx:
            break
        if idx != params.sos_idx and idx != params.pad_idx:
            rsp_lst.append(params.i2w[idx])
    return word_delim.join(rsp_lst), ppl


def post_evaluate_test_results_file(f_name="test_results.txt", col_delim="|", word_delim=" ", ignore_header=True):
    with open(f_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = lines[1:] if ignore_header else lines
    lsplit = [line.rstrip().strip().split(col_delim) for line in lines]
    gen_count = 0
    rv = {}
    corpus_preds = []
    corpus_truth = []
    for val_list in lsplit:
        if len(val_list) != 4: continue
        gen = val_list[1]
        truth = val_list[2]
        ppl = val_list[3]
        gen_count += 1
        truth_seg = [w for w in truth.split(word_delim) if len(w) > 0]
        gen_seg = [w for w in gen.split(word_delim) if len(w) > 0]
        corpus_preds.append(gen_seg)
        corpus_truth.append([truth_seg])
        multi_eval(gen_seg, truth_seg, result_dict=rv)
        if "ppl" not in rv: rv["ppl"] = 0
        rv["ppl"] += float(ppl)
    if gen_count == 0: return rv
    for k, v in rv.items():
        rv[k] = v / gen_count
    rv["corpus_bleu_4"] = bleu_c(corpus_preds, corpus_truth)
    return rv


def corpus_eval(pred_lists, truth_lists):
    """
    :param pred_lists: shape like [ [pred_word1, pred_word2...], [pred_word1, pred_word2...] ]
    :param truth_lists: shape like [ [[truth_word1, truth_word2...]], [[truth_word1, truth_word2...]] ]
    :param result_dict:
    """
    assert len(pred_lists) > 0, "pred_lists cannot be empty"
    assert len(pred_lists) == len(truth_lists), "One prediction must correspond to one truth"
    rv = {
        "bleu_1":0,
        "bleu_2":0,
        "bleu_3":0,
        "bleu_4":0,
        "rouge_1":0,
        "rouge_2":0,
        "rouge_L":0,
        "em":0,
    }
    for i, pred_seg in enumerate(pred_lists):
        truth_seg = truth_lists[i][0] # TODO: support multiple truth?
        rv["bleu_1"] += bleu_1(pred_seg, truth_seg)
        rv["bleu_2"] += bleu_2(pred_seg, truth_seg)
        rv["bleu_3"] += bleu_3(pred_seg, truth_seg)
        rv["bleu_4"] += bleu_4(pred_seg, truth_seg)
        rv["rouge_1"] += rouge_1(pred_seg, truth_seg)
        rv["rouge_2"] += rouge_2(pred_seg, truth_seg)
        rv["rouge_L"] += rouge_L(pred_seg, truth_seg)
        rv["em"] += em(pred_seg, truth_seg)
    for k, v in rv.items():
        rv[k] = v / len(pred_lists) # macro average
    rv["corpus_bleu_4"] = bleu_c(pred_lists, truth_lists)
    return rv


def multi_eval(gen_seg, truth_seg, result_dict={}):
    if "em" not in result_dict: result_dict["em"] = 0
    result_dict["em"] += em(gen_seg, truth_seg)
    if "bleu_1" not in result_dict: result_dict["bleu_1"] = 0
    result_dict["bleu_1"] += bleu_1(gen_seg, truth_seg)
    if "bleu_2" not in result_dict: result_dict["bleu_2"] = 0
    result_dict["bleu_2"] += bleu_2(gen_seg, truth_seg)
    if "bleu_3" not in result_dict: result_dict["bleu_3"] = 0
    result_dict["bleu_3"] += bleu_3(gen_seg, truth_seg)
    if "bleu_4" not in result_dict: result_dict["bleu_4"] = 0
    result_dict["bleu_4"] += bleu_4(gen_seg, truth_seg)
    if "rouge_1" not in result_dict: result_dict["rouge_1"] = 0
    result_dict["rouge_1"] += rouge_1(gen_seg, truth_seg)
    if "rouge_2" not in result_dict: result_dict["rouge_2"] = 0
    result_dict["rouge_2"] += rouge_2(gen_seg, truth_seg)
    if "rouge_L" not in result_dict: result_dict["rouge_L"] = 0
    result_dict["rouge_L"] += rouge_L(gen_seg, truth_seg)
    return result_dict


references = [[['<SOS>', 'Which', 'NFL', 'team', 'represented', 'the', 'AFC', 'at', 'Super', 'Bowl', '50', '?', '<EOS>']], [['<SOS>', 'Which', 'NFL', 'team', 'represented', 'the', 'NFC', 'at', 'Super', 'Bowl', '50', '?', '<EOS>']], [['<SOS>', 'Where', 'did', 'Super', 'Bowl', '50', 'take', 'place', '?', '<EOS>']], [['<SOS>', 'Where', 'did', 'Super', 'Bowl', '50', 'take', 'place', '?', '<EOS>']], [['<SOS>', 'Where', 'did', 'Super', 'Bowl', '50', 'take', 'place', '?', '<EOS>']], [['<SOS>', 'Which', 'NFL', 'team', 'won', 'Super', 'Bowl', '50', '?', '<EOS>']], [['<SOS>', 'What', 'color', 'was', 'used', 'to', 'emphasize', 'the', '50th', 'anniversary', 'of', 'the', 'Super', 'Bowl', '?', '<EOS>']], [['<SOS>', 'What', 'color', 'was', 'used', 'to', 'emphasize', 'the', '50th', 'anniversary', 'of', 'the', 'Super', 'Bowl', '?', '<EOS>']], [['<SOS>', 'What', 'was', 'the', 'theme', 'of', 'Super', 'Bowl', '50', '?', '<EOS>']], [['<SOS>', 'What', 'was', 'the', 'theme', 'of', 'Super', 'Bowl', '50', '?', '<EOS>']], [['<SOS>', 'What', 'was', 'the', 'theme', 'of', 'Super', 'Bowl', '50', '?', '<EOS>']], [['<SOS>', 'What', 'day', 'was', 'the', 'game', 'played', 'on', '?', '<EOS>']]]
candidates = [['defeated', '<OOV>', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], ['Super', '<OOV>', 'title', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], ['Francisco', '<OOV>', '<OOV>', '?', '<OOV>', '<OOV>', '<OOV>', '<OOV>', 'game', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'Levi', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'Stadium', '<OOV>', '<OOV>', '<OOV>', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], ['<OOV>', 'the', '<OOV>', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], ['<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], ['<OOV>', 'National', '<OOV>', '<OOV>', 'Denver', '<OOV>', 'Broncos', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', 'of', 'of', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', 'National', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>'], ['50th', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', '50th', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], ['gold', '<OOV>', 'was', '<OOV>', '?', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'prominently', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>'], ['the', '<OOV>', 'the', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'could', '<OOV>', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], ['suspending', '<OOV>', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], ['<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', '50th', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '?', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], [',', '2016', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', ',', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>']]
if __name__ == "__main__":
    assert len(candidates) == len(references)
    # rv = {}
    # for i, v in enumerate(references):
    #     gold_seg = v[0]
    #     pred_seg = candidates[i]
    #     multi_eval(pred_seg, gold_seg, result_dict=rv)
    # for k, v in rv.items():
    #     rv[k] = v / len(candidates)
    # print(rv)
    # print(post_evaluate_test_results_file("con_qa_test_results.txt"))
    # print(post_evaluate_test_results_file("con_crx_test_results.txt"))
