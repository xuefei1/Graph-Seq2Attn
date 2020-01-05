import copy
import math
import time
import torch
import torch.nn as nn
import numpy as np
from constants import *
from embeddings import *
from tqdm import tqdm
from utils.model_utils import device, model_checkpoint
from utils.misc_utils import write_line_to_file
from utils.lang_utils import make_std_mask
from evaluate import corpus_eval


def clones(module, n):
    "Produce n identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def build_sents_mask(data_dict, pad_idx=0, eos_idx=3):
    batch_sents_list = data_dict[DK_DOC_SENTS_WID]
    batch_size = data_dict[DK_BATCH_SIZE]
    batch_sents_mask = torch.ones(batch_size, 1, len(batch_sents_list)).type(torch.ByteTensor).to(device())
    for si, src in enumerate(batch_sents_list):
        assert src.shape[0] == batch_size
        src = src.to(device())
        if pad_idx is not None:
            b_exc_pad = (src != pad_idx)
        else:
            b_exc_pad = torch.ones(src.size()).type(torch.BoolTensor).to(device())
        if eos_idx is not None:
            b_exc_eos = (src != eos_idx)
        else:
            b_exc_eos = torch.ones(src.size()).type(torch.BoolTensor).to(device())
        b_exc_both = b_exc_pad & b_exc_eos
        b_exc_aggr = b_exc_both.sum(1)
        mask_vals = (b_exc_aggr != 0)
        batch_sents_mask[:, 0, si] = mask_vals
    return batch_sents_mask.to(device())


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # batch_size x n_heads x seq_len x seq_len, i.e. attn score on each word
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1) # batch_size x n_heads x seq_len x seq_len, softmax on last dimension, i.e. 3rd dimension attend on 4th dimension
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn # attended output, attention vec


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        rv = {}
        rv["_step"] = self._step
        rv["warmup"] = self.warmup
        rv["factor"] = self.factor
        rv["model_size"] = self.model_size
        rv["_rate"] = self._rate
        rv["opt_state_dict"] = self.optimizer.state_dict()
        return rv

    def load_state_dict(self, state_dict):
        self._step = state_dict["_step"]
        self.warmup = state_dict["warmup"]
        self.factor = state_dict["factor"]
        self.model_size = state_dict["model_size"]
        self._rate = state_dict["_rate"]
        self.optimizer.load_state_dict(state_dict["opt_state_dict"])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device())


class LabelSmoothing(nn.Module):
    "Label smoothing actually starts to penalize the model if it gets very confident about a given choice"

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        assert x.size(1) == self.size
        x = x.to(device())
        target = target.to(device())
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        indices = target.data.unsqueeze(1)
        true_dist.scatter_(1, indices, self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.shape[0] > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist)


class GraphSeq2Attn(nn.Module):

    def __init__(self, doc_gcn_kw_embed, doc_gcn_merge_embed, doc_sents_wid_embed, tgt_wid_embed,
                 doc_merge_gcn, doc_kw_dist_gcn,
                 sents_encoder, ctx_encoder, pool, decoder, generator,
                 pad_idx=0, sos_idx=2, eos_idx=3):
        super(GraphSeq2Attn, self).__init__()
        self.doc_gcn_kw_embed = doc_gcn_kw_embed
        self.doc_gcn_merge_embed = doc_gcn_merge_embed
        self.doc_sents_wid_embed = doc_sents_wid_embed
        self.tgt_wid_embed = tgt_wid_embed
        self.doc_merge_gcn = doc_merge_gcn
        self.doc_kw_dist_gcn = doc_kw_dist_gcn
        self.sents_encoder = sents_encoder
        self.ctx_encoder = ctx_encoder
        self.pool = pool
        self.decoder = decoder
        self.generator = generator
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(self, data_dict):
        sos = torch.ones(1, 1).type(torch.LongTensor).fill_(self.sos_idx).repeat(data_dict[DK_BATCH_SIZE], 1).to(device())
        tgt = data_dict[DK_TGT_GEN_WID].to(device())
        tgt = torch.cat([sos, tgt], dim=1)
        tgt = tgt[:, :-1]
        tgt_mask = make_std_mask(tgt, self.pad_idx)
        enc_list = self.encode(data_dict)
        decoder_out = self.decode(tgt, tgt_mask, enc_list)
        g_probs = self.generator(decoder_out)
        return g_probs

    def encode(self, data_dict):
        sents_encoded = []
        batch_sents_list = data_dict[DK_DOC_SENTS_WID]
        enc_hidden = None
        for src in batch_sents_list:
            src = src.to(device())
            enc_op, enc_hidden = self.encode_seq(src, enc_hidden)
            sents_encoded.append(enc_op)
        sents_encoded = torch.cat(sents_encoded, dim=1)
        batch_sents_attn_mask = build_sents_mask(data_dict, pad_idx=self.pad_idx, eos_idx=self.eos_idx)
        doc_merge_embedded, doc_merge_adj, _ = self.doc_gcn_merge_embed(data_dict, sents_encoded)
        doc_merge_layer_outputs, _ = self.doc_merge_gcn(doc_merge_embedded, doc_merge_adj)
        doc_graph_out = doc_merge_layer_outputs[-1]
        kw_dist_embedded, kw_dist_adj, kw_dist_mask = self.doc_gcn_kw_embed(data_dict)
        kw_dist_layer_outputs, _ = self.doc_kw_dist_gcn(kw_dist_embedded, kw_dist_adj)
        kw_dist_out = kw_dist_layer_outputs[-1]
        sent_lens_mask = build_sents_mask(data_dict, pad_idx=self.pad_idx, eos_idx=None)
        sent_lens = sent_lens_mask.squeeze(1).sum(1)
        doc_encoded, _ = self.ctx_encoder(sents_encoded, sent_lens)
        rv = [(doc_encoded, batch_sents_attn_mask), (doc_graph_out, batch_sents_attn_mask), (kw_dist_out, kw_dist_mask)]
        return rv

    def encode_seq(self, src, encoder_hidden=None, encoder_cell=None):
        if self.sents_encoder.rnn_type.lower() == "lstm": encoder_hidden = (encoder_hidden, encoder_cell)
        src_lens = (src != self.pad_idx).sum(1).to(device())
        src = self.doc_sents_wid_embed(src)
        encoder_op, encoder_hidden = self.sents_encoder(src, src_lens, encoder_hidden)
        return encoder_op[:,-1,:].unsqueeze(1), encoder_hidden

    def decode(self, tgt, tgt_mask, src_tup_list):
        tgt_embedded = self.tgt_wid_embed(tgt)
        mem_tup_list = []
        for t in src_tup_list:
            mem = self.pool(t[0])
            mem_mask = t[1]
            if mem_mask is not None and mem.shape[1] != mem_mask.shape[2]:
                mem_mask = None
            mem_tup_list.append((mem, mem_mask))
        decoder_out = self.decoder(tgt_embedded, tgt_mask, mem_tup_list)
        # decoder_out = self.decoder(tgt_embedded, tgt_mask, src_tup_list)
        return decoder_out

    def predict(self, decoder_out, topk=1, topk_dim=2):
        decoder_out = decoder_out[:, -1, :].unsqueeze(1)
        prob = self.generator(decoder_out)
        val, indices = prob.topk(topk, dim=topk_dim)
        return prob, val, indices


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h # dimesion of keys should be constrained by model hidden size?
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3) # d_model to d_model, attn key dimension downsize is achieved through reshaping
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.last_ff = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # batch_size x seq_len x d_model => batch_size x n_heads x seq_len x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) # batch_size x n_heads x seq_len x d_k
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # batch_size x n_heads x seq_len x d_k => batch_size x n_heads x seq_len x d_k
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # batch_size x n_heads x seq_len x d_k => batch_size x seq_len x d_model
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x = self.last_ff(x)
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.layer_1 = nn.Linear(d_model, d_ff)
        self.layer_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        l1_output = torch.relu(self.layer_1(x))
        l1_output = self.dropout(l1_output)
        l2_output = self.layer_2(l1_output)
        return l2_output


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_func):
        "Apply residual connection to any sublayer with the same size."
        layer_output = sublayer_func(x)
        residual_rv = x + self.dropout(layer_output)
        return self.norm(residual_rv)


class SequentialDecoderLayer(nn.Module):

    def __init__(self, size, attn, pos_ff, dropout, n_parallel_units=1):
        super(SequentialDecoderLayer, self).__init__()
        c = copy.deepcopy
        self.size = size
        self.pos_ff = c(pos_ff)
        self.self_attn = c(attn)
        self.p_attns = clones(attn, n_parallel_units)
        self.n_parallel_units = n_parallel_units
        self.sublayer = SublayerConnection(size, dropout)

    def forward(self, tgt, tgt_mask, attn_in_tup_list):
        rv = self.sublayer(tgt, lambda q: self.self_attn(q, q, q, tgt_mask))
        for i, t in enumerate(attn_in_tup_list):
            rv = self.sublayer(rv, lambda q: self.p_attns[i](q, t[0], t[0], t[1]))
        rv = self.sublayer(rv, self.pos_ff)
        return rv


class SequentialDecoder(nn.Module):

    def __init__(self, d_model, layer, n, pos_embed=None):
        super(SequentialDecoder, self).__init__()
        self.d_model = d_model
        self.layers = clones(layer, n)
        self.pos_embed = copy.deepcopy(pos_embed) if pos_embed is not None else None

    def forward(self, tgt, tgt_mask, attn_in_tup_list):
        rv = tgt
        if self.pos_embed is not None:
            rv = self.pos_embed(rv)
        for layer in self.layers:
            rv = layer(rv, tgt_mask, attn_in_tup_list)
        return rv


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).type(torch.FloatTensor).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).type(torch.FloatTensor) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # mark as not learnable parameters, but still part of the state

    def forward(self, x, pe_expand_dim=None):
        encoding_vals = self.pe
        if pe_expand_dim is not None:
            encoding_vals = self.pe.unsqueeze(pe_expand_dim)
        x = x + encoding_vals[:, :x.size(1), :] # just reads the first seq_len positional embedding values
        return x


class GCN(nn.Module):
    """
    A GCN/Contextualized GCN module operated on dependency graphs.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0, num_layers=2):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gcn_drop = nn.Dropout(dropout)

        # gcn layer
        self.W = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            input_dim = self.input_dim if layer_idx == 0 else self.hidden_dim
            output_dim = self.output_dim if layer_idx == self.num_layers-1 else self.hidden_dim
            self.W.append(nn.Linear(input_dim, output_dim))

    def forward(self, gcn_inputs, adj):
        """
        :param adj: batch_size * num_vertex * num_vertex
        :param gcn_inputs: batch_size * num_vertex * input_dim
        :return: gcn_outputs: list of batch_size * num_vertex * hidden_dim
                 mask: batch_size * num_vertex * 1. In mask, 1 denotes
                     this vertex is PAD vertex, 0 denotes true vertex.
        """
        # use out degree, assume undirected graph
        denom = adj.sum(2).unsqueeze(2) + 1
        adj_mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        gcn_outputs = []
        for l in range(self.num_layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs)  # self loop
            AxW = AxW / denom
            gAxW = torch.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.num_layers - 1 else gAxW
            gcn_outputs.append(gcn_inputs)

        return gcn_outputs, adj_mask


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, rnn_dir=2, dropout_prob=0.0, rnn_type="gru"):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn_dir = rnn_dir
        self.rnn_type = rnn_type
        self.dropout_prob = dropout_prob
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.input_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.n_layers,
                               dropout=self.dropout_prob if self.n_layers>1 else 0,
                               batch_first=True,
                               bidirectional=self.rnn_dir==2)
        else:
            self.rnn = nn.GRU(input_size=self.input_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.n_layers,
                               dropout=self.dropout_prob if self.n_layers>1 else 0,
                               batch_first=True,
                               bidirectional=self.rnn_dir==2)

    def forward(self, embedded, input_lengths, hidden=None):
        outputs, hidden = self.rnn(embedded, hidden)
        return outputs, hidden


class SeqMaxPoolLayer(nn.Module):
    def __init__(self, d_model, max_pool_factor=2, min_seq_len=4, fill_val=-1e9):
        super(SeqMaxPoolLayer, self).__init__()
        self.min_seq_len = min_seq_len
        self.max_pool_factor = max_pool_factor
        self.d_model = d_model
        self.max_pool = nn.MaxPool1d(self.max_pool_factor, stride=self.max_pool_factor)
        self.fill_val = fill_val

    def pad_to_max_pool_size(self, x, fill_val=None):
        fill_val = self.fill_val if fill_val is None else fill_val
        if x.shape[1] <= self.min_seq_len or self.max_pool_factor <= 1:
            return x
        if x.shape[1] % self.max_pool_factor != 0:
            new_size = x.shape[1] + (self.max_pool_factor - x.shape[1] % self.max_pool_factor)
            padded = torch.zeros(x.shape[0], new_size, x.shape[2]).type(torch.FloatTensor).to(device())
            padded.fill_(fill_val)
            padded[:, :x.shape[1], :] = x
        else:
            padded = x
        return padded

    def forward(self, x):
        padded_x = self.pad_to_max_pool_size(x)
        if x.shape[1] > self.min_seq_len:
            padded_x = padded_x.transpose(1, 2)
            padded_x = self.max_pool(padded_x)
            padded_x = padded_x.transpose(1, 2)
        return padded_x


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        logits = self.proj(x)
        probs = torch.log_softmax(logits, dim=-1)
        return probs


class BeamSearchResult:
    def __init__(self, i2w, idx_in_batch,
                 beam_width=4, sos_idx=2, eos_idx=3,
                 gamma=0.0, len_norm=0.0):
        self.idx_in_batch = idx_in_batch
        self.beam_width = beam_width
        self.gamma = gamma
        self.len_norm = len_norm
        self.eos_idx = eos_idx
        sos = torch.ones(1,1).fill_(sos_idx).type(torch.LongTensor).to(device())
        self.curr_candidates = [
            (sos, 1.0, [], [i2w[sos_idx]])
        ]
        self.completed_insts = []
        self.done = False

    def update(self, probs, next_vals, next_wids, next_words):
        assert len(next_wids) == len(self.curr_candidates)
        next_candidates = []
        for i, tup in enumerate(self.curr_candidates):
            prev_tgt = tup[0]
            score = tup[1]
            prev_prob_list = [t for t in tup[2]]
            prev_words = [t for t in tup[3]]
            preds = next_wids[i]
            vals = next_vals[i]
            pred_words = next_words[i]
            prev_prob_list.append(probs)
            for bi in range(len(preds)):
                wi = preds[bi]
                val = vals[bi]
                word = pred_words[bi]
                div_penalty = 0.0
                if bi > 0: div_penalty = self.gamma * (bi+1)
                new_score = score + val - div_penalty
                new_tgt = torch.cat([prev_tgt, torch.ones(1,1).type(torch.LongTensor).fill_(wi).to(device())], dim=1)
                new_words = [w for w in prev_words]
                new_words.append(word)
                if wi == self.eos_idx:
                    if self.len_norm > 0:
                        length_penalty = (self.len_norm + new_tgt.shape[1]) / (self.len_norm + 1)
                        new_score /= length_penalty ** self.len_norm
                    else:
                        new_score = new_score / new_tgt.shape[1] if new_tgt.shape[1] > 0 else new_score
                    ppl = 0 # TODO: add perplexity later
                    self.completed_insts.append((new_tgt, new_score, ppl, new_words))
                else:
                    next_candidates.append((new_tgt, new_score, prev_prob_list, new_words))
        next_candidates = sorted(next_candidates, key=lambda t: t[1], reverse=True)
        next_candidates = next_candidates[:self.beam_width]
        self.curr_candidates = next_candidates
        self.done = len(self.curr_candidates) == 0

    def get_curr_tgt(self):
        if len(self.curr_candidates) == 0: return None
        return torch.cat([tup[0] for tup in self.curr_candidates], dim=0).type(torch.LongTensor).to(device())

    def get_curr_candidate_size(self):
        return len(self.curr_candidates)

    def collect_results(self, topk=1):
        for cand in self.curr_candidates:
            self.completed_insts.append((cand[0], cand[1], 0, cand[3]))
        self.completed_insts = sorted(self.completed_insts, key=lambda t: t[1], reverse=True)
        self.completed_insts = self.completed_insts[:self.beam_width]
        return self.completed_insts[:topk]


def eval_graph_seq2attn(model, loader, params, desc="Eval"):
    start = time.time()
    exclude_tokens = [params.sos_token, params.eos_token, params.pad_token, ""]
    truth_rsp = []
    gen_rsp = []
    ofn = params.logs_dir + params.model_name + "_eval_out.txt"
    write_line_to_file("", ofn)
    for batch in tqdm(loader, mininterval=2, desc=desc, leave=False, ascii=True):
        batch = copy.deepcopy(batch)
        beam_rvs = graph_seq2attn_beam_decode_batch(model, batch, params.sos_idx, params.tgt_i2w,
                                                    len_norm=params.bs_len_norm, gamma=params.bs_div_gamma,
                                                    max_len=params.max_decoder_seq_len, beam_width=params.beam_width_eval)
        for bi in range(batch[DK_BATCH_SIZE]):
            best_rv = beam_rvs[bi][3]
            truth_rsp.append([[w for w in batch[DK_TGT_SEG_LISTS][bi] if w not in exclude_tokens]])
            gen_rsp.append([w for w in best_rv if w not in exclude_tokens])
            write_line_to_file("truth: " + " ".join([w for w in batch[DK_TGT_SEG_LISTS][bi] if w not in exclude_tokens]), ofn)
            write_line_to_file("pred: " + " ".join([w for w in best_rv if w not in exclude_tokens]), ofn)
    perf = corpus_eval(gen_rsp, truth_rsp)
    elapsed = time.time() - start
    info = "Eval result {} elapsed {}".format(str(perf), elapsed)
    print(info)
    write_line_to_file(info, params.logs_dir + params.model_name + "_train_info.txt")
    return perf[params.eval_metric]


def make_graph_seq2attn_model(src_w2v_mat, tgt_w2v_mat, params, src_vocab_size, tgt_vocab_size,
                              same_word_embedding=False):
    n_parallel_units = 3
    n_heads = params.graph_seq2attn_num_attn_heads
    d_model_sent_enc = params.graph_seq2attn_encoder_hidden_size
    d_model_ctx_enc = params.graph_seq2attn_context_hidden_size
    d_model_decoder = params.graph_seq2attn_context_hidden_size * params.graph_seq2attn_context_rnn_dir # TODO: small issue here
    d_model_dec_pos_ff = int(params.graph_seq2attn_decoder_hidden_size * params.graph_seq2attn_decoder_ff_ratio)
    d_output_gcn = d_model_decoder
    dec_attn = MultiHeadedAttention(n_heads, d_model_decoder)
    dec_ff = PositionwiseFeedForward(d_model_decoder, d_model_dec_pos_ff)
    dec_word_pos = PositionalEncoding(d_model_decoder)
    # embeddings
    if src_w2v_mat is None:
        src_word_embed_layer = TrainableEmbedding(params.word_embedding_dim, src_vocab_size)
    else:
        src_word_embed_layer = PreTrainedWordEmbedding(src_w2v_mat, params.word_embedding_dim,
                                                       allow_further_training=params.src_embed_further_training)
    if tgt_w2v_mat is None:
        tgt_word_embed_layer = TrainableEmbedding(params.word_embedding_dim, tgt_vocab_size)
    else:
        tgt_word_embed_layer = PreTrainedWordEmbedding(tgt_w2v_mat, params.word_embedding_dim,
                                                       allow_further_training=params.tgt_embed_further_training)
    if same_word_embedding and src_vocab_size == tgt_vocab_size:
        tgt_word_embed_layer = src_word_embed_layer
    doc_gcn_kw_embed = DocGCNKWDistDictEmbedding(params.word_embedding_dim, src_word_embed_layer)
    doc_gcn_merge_embed = DocGCNDictMergeEmbedding()
    tgt_wid_embed = ResizeWrapperEmbedding(d_model_decoder, tgt_word_embed_layer)
    # high-level components
    doc_merge_gcn = GCN(d_model_sent_enc * params.graph_seq2attn_encoder_rnn_dir, params.graph_seq2attn_doc_merge_gcn_hidden_size, d_output_gcn,
                        params.graph_seq2attn_doc_merge_gcn_dropout_prob, num_layers=params.graph_seq2attn_doc_merge_gcn_layers)
    doc_kw_dist_gcn = GCN(params.word_embedding_dim, params.graph_seq2attn_doc_kw_dist_gcn_hidden_size, d_output_gcn,
                          params.graph_seq2attn_doc_kw_dist_gcn_dropout_prob, num_layers=params.graph_seq2attn_doc_kw_dist_gcn_layers)
    sent_encoder = EncoderRNN(params.word_embedding_dim, d_model_sent_enc, n_layers=params.graph_seq2attn_num_encoder_layers,
                              dropout_prob=params.graph_seq2attn_encoder_dropout_prob,
                              rnn_dir=params.graph_seq2attn_encoder_rnn_dir, rnn_type=params.graph_seq2attn_encoder_type)
    doc_encoder = EncoderRNN(d_model_sent_enc * params.graph_seq2attn_encoder_rnn_dir, d_model_ctx_enc, n_layers=params.graph_seq2attn_num_context_layers,
                              dropout_prob=params.graph_seq2attn_context_dropout_prob,
                              rnn_dir=params.graph_seq2attn_context_rnn_dir, rnn_type=params.graph_seq2attn_context_type)
    max_pool = SeqMaxPoolLayer(d_model_decoder, max_pool_factor=params.graph_seq2attn_pool_factor, min_seq_len=1)
    decoder_layer = SequentialDecoderLayer(d_model_decoder, dec_attn, dec_ff, params.graph_seq2attn_decoder_dropout_prob,
                                           n_parallel_units=n_parallel_units)
    decoder = SequentialDecoder(d_model_decoder, decoder_layer, params.graph_seq2attn_num_decoder_layers, pos_embed=dec_word_pos)
    generator = Generator(d_model_decoder, tgt_vocab_size)

    # model
    model = GraphSeq2Attn(doc_gcn_kw_embed, doc_gcn_merge_embed, src_word_embed_layer, tgt_wid_embed,
                          doc_merge_gcn, doc_kw_dist_gcn, sent_encoder, doc_encoder, max_pool, decoder, generator,
                          pad_idx=params.pad_idx, sos_idx=params.sos_idx, eos_idx=params.eos_idx)
    for p in filter(lambda pa: pa.requires_grad, model.parameters()):
        if p.dim() == 1:
            p.data.normal_(0, math.sqrt(6 / (1 + p.size(0))))
        else:
            nn.init.xavier_normal_(p, math.sqrt(3))
    return model


def train_graph_seq2attn(params, model, train_loader, criterion_gen, optimizer,
                         completed_epochs=0, dev_loader=None,
                         best_eval_result=0, best_eval_epoch=0, past_eval_results=[],
                         checkpoint=True):
    model = model.to(device())
    criterion_gen = criterion_gen.to(device())
    for epoch in range(params.epochs):
        report_epoch = epoch + completed_epochs + 1
        model.train()
        run_graph_seq2attn_epoch(model, train_loader, criterion_gen,
                                 curr_epoch=report_epoch, optimizer=optimizer,
                                 max_grad_norm=params.max_gradient_norm,
                                 desc="Train", pad_idx=params.pad_idx,
                                 model_name=params.model_name,
                                 logs_dir=params.logs_dir)
        if dev_loader is not None: # fast eval
            model.eval()
            with torch.no_grad():
                if report_epoch >= params.full_eval_start_epoch and \
                   report_epoch % params.full_eval_every_epoch == 0: # full eval
                    eval_bleu_4 = eval_graph_seq2attn(model, dev_loader, params)
                    if eval_bleu_4 > best_eval_result:
                        best_eval_result = eval_bleu_4
                        best_eval_epoch = report_epoch
                        print("Model best checkpoint with score {}".format(eval_bleu_4))
                        fn = params.saved_models_dir + params.model_name + "_best.pt"
                        if checkpoint:
                            model_checkpoint(fn, report_epoch, model, optimizer, params,
                                             past_eval_results, best_eval_result, best_eval_epoch)
                    info = "Best {} so far {} from epoch {}".format(params.eval_metric, best_eval_result, best_eval_epoch)
                    print(info)
                    write_line_to_file(info, params.logs_dir + params.model_name + "_train_info.txt")
                    if hasattr(optimizer, "update_learning_rate"):
                        optimizer.update_learning_rate(eval_bleu_4)
                    past_eval_results.append(eval_bleu_4)
                    if len(past_eval_results) > params.past_eval_scores_considered:
                        past_eval_results = past_eval_results[1:]
        fn = params.saved_models_dir + params.model_name + "_latest.pt"
        if checkpoint:
            model_checkpoint(fn, report_epoch, model, optimizer, params,
                             past_eval_results, best_eval_result, best_eval_epoch)
        print("")
    return best_eval_result, best_eval_epoch


def run_graph_seq2attn_epoch(model, loader, criterion, curr_epoch=0, max_grad_norm=5.0, optimizer=None,
                             desc="Train", pad_idx=0, model_name="graph_seq2attn", logs_dir=""):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_correct = 0
    for batch in tqdm(loader, mininterval=2, desc=desc, leave=False, ascii=True):
        g_log_wid_probs = model(batch)
        gen_targets = batch[DK_TGT_GEN_WID].to(device())
        n_tokens = batch[DK_TGT_N_TOKEN].item()
        g_log_wid_probs = g_log_wid_probs.view(-1, g_log_wid_probs.size(-1))
        loss = criterion(g_log_wid_probs, gen_targets.contiguous().view(-1))
        # compute acc
        tgt = copy.deepcopy(gen_targets.view(-1, 1).squeeze(1))
        g_preds_i = copy.deepcopy(g_log_wid_probs.max(1)[1])
        n_correct = g_preds_i.data.eq(tgt.data)
        n_correct = n_correct.masked_select(tgt.ne(pad_idx).data).sum()
        total_loss += loss.item()
        total_correct += n_correct.item()
        total_tokens += n_tokens
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_grad_norm)
            optimizer.step()
            if torch.isnan(loss).any():
                assert False, "nan detected after step()"
    loss_report = total_loss / total_tokens
    acc = total_correct / total_tokens
    elapsed = time.time() - start
    info = desc + " epoch %d loss %f, acc %f ppl %f elapsed time %f" % (curr_epoch, loss_report, acc,
                                                                        math.exp(loss_report), elapsed)
    print(info)
    write_line_to_file(info, logs_dir + model_name + "_train_info.txt")
    return loss_report, acc


def graph_seq2attn_beam_decode_batch(model, batch, start_idx, i2w, max_len,
                                     gamma=0.0, pad_idx=0, beam_width=4, eos_idx=3, len_norm=1.0, topk=1):
    batch_size = batch[DK_BATCH_SIZE]
    model = model.to(device())
    enc_list = model.encode(batch)
    batch_results = [
        BeamSearchResult(idx_in_batch=bi, i2w=i2w,
                         beam_width=beam_width, sos_idx=start_idx,
                         eos_idx=eos_idx, gamma=gamma, len_norm=len_norm)
        for bi in range(batch_size)
    ]
    final_ans = []
    for i in range(max_len):
        curr_actives = [b for b in batch_results if not b.done]
        if len(curr_actives) == 0: break
        b_tgt_list = [b.get_curr_tgt() for b in curr_actives]
        b_tgt = torch.cat(b_tgt_list, dim=0)
        b_tgt_mask = make_std_mask(b_tgt, pad_idx)
        b_cand_size_list = [b.get_curr_candidate_size() for b in curr_actives]
        b_enc_list = []
        for tup in enc_list:
            mem, mask = tup
            b_mem = torch.cat([mem[b.idx_in_batch, :, :].unsqueeze(0).repeat(b.get_curr_candidate_size(), 1, 1) for b in curr_actives], dim=0)
            b_mask = None
            if mask is not None:
                b_mask = torch.cat([mask[b.idx_in_batch, :, :].unsqueeze(0).repeat(b.get_curr_candidate_size(), 1, 1) for b in curr_actives], dim=0)
            b_enc_list.append((b_mem, b_mask))
        dec_out = model.decode(b_tgt, b_tgt_mask, b_enc_list)
        gen_wid_probs, _, _ = model.predict(dec_out)
        beam_i = 0
        for bi, size in enumerate(b_cand_size_list):
            g_probs = gen_wid_probs[beam_i:beam_i + size, :].view(size, -1, gen_wid_probs.size(-1))
            vt, it = g_probs.topk(beam_width)
            next_vals, next_wids, next_words = [], [], []
            for ci in range(size):
                vals, wis, words = [], [], []
                for idx in range(beam_width):
                    vals.append(vt[ci, 0, idx].item())
                    wi = it[ci, 0, idx].item()
                    word = i2w[wi]
                    wis.append(wi)
                    words.append(word)
                next_vals.append(vals)
                next_wids.append(wis)
                next_words.append(words)
            curr_actives[bi].update(g_probs, next_vals, next_wids, next_words)
            beam_i += size
    for b in batch_results:
        final_ans += b.collect_results(topk=topk)
    return final_ans

