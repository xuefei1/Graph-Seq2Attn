import os
import math
import copy
import torch
import random
import pickle
import numpy as np
from constants import *
from tqdm import tqdm
from utils.model_utils import device
from utils.lang_utils import gen_word2idx_vec_rep


def build_sent_depend_edges(sent_depend_list, word_filter_dict=None):
    edges = []
    for tup in sent_depend_list:
        if len(tup) != 5: continue
        curr_word = tup[1]
        comp_word = tup[2]
        curr_idx = tup[3]
        comp_idx = tup[4]
        if curr_word == comp_word: continue
        if word_filter_dict is not None:
            if curr_word not in word_filter_dict or comp_word not in word_filter_dict: continue
        edge = {
            GK_EDGE_TYPE: GK_EDGE_UNDIR,
            GK_EDGE_WORD_PAIR: (curr_word, comp_word),
            GK_EDGE_GV_IDX_PAIR: (curr_idx, comp_idx),
            GK_EDGE_WEIGHT: 1.0,
        }
        edges.append(edge)
    return edges


def build_doc_sent_word_overlap_percent_edges(doc_sent_words, threshold=None):
    edges = []
    max_weight = 0.0
    for curr_i, curr_words in enumerate(doc_sent_words):
        for comp_i, comp_words in enumerate(doc_sent_words):
            if comp_i <= curr_i: continue
            total_words_count = len(curr_words + comp_words)
            overlapping_words_count = len(list(set(curr_words) - set(comp_words)))
            weight = overlapping_words_count / total_words_count
            if threshold is not None:
                weight = 0.0 if weight < threshold else weight
            if weight > 0:
                max_weight = max(max_weight, weight)
                edge = {
                    GK_EDGE_TYPE: GK_EDGE_UNDIR,
                    GK_EDGE_GV_IDX_PAIR: (curr_i, comp_i),
                    GK_EDGE_WEIGHT: weight,
                }
                edges.append(edge)
    if max_weight > 0:
        for ed in edges:
            ed[GK_EDGE_WEIGHT] = ed[GK_EDGE_WEIGHT] * (1.0 / max_weight)
    return edges


def shortest_keywords_dist_in_doc(doc_w_list, kw1, kw2):
    if len(doc_w_list)==0: return 0
    if kw1 == kw2: assert False, "Keywords shouldn't be the same"
    kw1_indices = [idx for idx, w in enumerate(doc_w_list) if w == kw1]
    kw2_indices = [idx for idx, w in enumerate(doc_w_list) if w == kw2]
    dists = [abs(w1i-w2i) for w1i in kw1_indices for w2i in kw2_indices]
    if len(dists)==0: return 0
    dist = sum(dists)/len(dists)
    if dist < 1e-5: return 0
    return 1/dist


def build_doc_keyword_dist_edges(doc_w_list, keywords, threshold=None):
    edges = []
    max_weight = 0.0
    for curr_i, curr_word in enumerate(keywords):
        for comp_i, comp_word in enumerate(keywords):
            if curr_word == comp_word: continue
            weight = shortest_keywords_dist_in_doc(doc_w_list, curr_word, comp_word)
            max_weight = max(max_weight, weight)
            if weight < 1e-7: continue
            if threshold is not None and weight < threshold: continue
            edge = {
                GK_EDGE_TYPE: GK_EDGE_UNDIR,
                GK_EDGE_WORD_PAIR: (curr_word, comp_word),
                GK_EDGE_GV_IDX_PAIR: (curr_i, comp_i),
                GK_EDGE_WEIGHT: weight,
            }
            edges.append(edge)
    if max_weight > 0:
        for ed in edges:
            ed[GK_EDGE_WEIGHT] = ed[GK_EDGE_WEIGHT] * (1.0 / max_weight)
    return edges


def build_batch_tensor_from_edges(batch_edges_list, max_vertex_count):
    batch_size = len(batch_edges_list)
    adj_tensor = torch.zeros(batch_size, max_vertex_count, max_vertex_count).type(torch.FloatTensor)
    for bi in range(batch_size):
        for ed in batch_edges_list[bi]:
            gv_idx_pair = ed[GK_EDGE_GV_IDX_PAIR]  # make sure this idx pair correctly corresponds to wid_tensor
            weight = ed[GK_EDGE_WEIGHT]
            adj_tensor[bi, int(gv_idx_pair[0]), int(gv_idx_pair[1])] = weight
            adj_tensor[bi, int(gv_idx_pair[1]), int(gv_idx_pair[0])] = weight
    return adj_tensor.to(device())


def gen_cpy_np(ctx_word_seg_lists, ans_word_seg_lists, max_tgt_len, w2i):
    assert len(ctx_word_seg_lists) == len(ans_word_seg_lists)
    cpy_wids = np.zeros((len(ctx_word_seg_lists), max_tgt_len))
    cpy_gates = np.zeros((len(ctx_word_seg_lists), max_tgt_len))
    for bi, ctx_word_seg_list in enumerate(ctx_word_seg_lists):
        ans_word_seg_list = ans_word_seg_lists[bi]
        for ci, cw in enumerate(ctx_word_seg_list):
            for ai, aw in enumerate(ans_word_seg_list):
                if aw in w2i: continue  # only allow copy for OOV words
                if cw == aw:
                    cpy_gates[bi,ai] = 1
                    cpy_wids[bi,ai] = ci
    return cpy_wids, cpy_gates


def make_batch_size(batch_size, batch_list):
    if len(batch_list) >= batch_size:
        return batch_list
    assert len(batch_list) > 0, "batch list cannot be empty"
    rv = []
    i = 0
    term_len = len(batch_list)
    while len(rv) < batch_size:
        new_inst = copy.deepcopy(batch_list[i])
        rv.append(new_inst)
        i += 1
        i %= term_len
    assert len(rv) == batch_size
    return rv


class BatchIDGraph:

    def __init__(self, pad_indices=[0]):
        self.batch_size = 0
        self.batch_vertex_wids = []
        self.batch_edges_lists = []
        self.max_n_vertices = 0
        self.pad_indices = pad_indices

    def add_inst(self, vertex_wids, edges):
        self.batch_vertex_wids.append(vertex_wids)
        self.batch_edges_lists.append(edges)
        self.batch_size += 1
        self.max_n_vertices = max(self.max_n_vertices, len(vertex_wids))

    def get_tensors(self, max_verts_override=None):
        assert self.batch_size > 0, "Cannot get tensor from empty graph"
        if max_verts_override is not None:
            self.max_n_vertices = max(self.max_n_vertices, max_verts_override)
        id_tensor = torch.zeros(self.batch_size, self.max_n_vertices).type(torch.LongTensor)
        for bi in range(self.batch_size):
            for idx, wid in enumerate(self.batch_vertex_wids[bi]):
                id_tensor[bi, idx] = wid
        adj_tensor = build_batch_tensor_from_edges(self.batch_edges_lists, self.max_n_vertices)
        wid_mask = torch.ones(id_tensor.size()).type(torch.ByteTensor).unsqueeze(1)
        for pad_idx in self.pad_indices:
            mask = (id_tensor != pad_idx).type(torch.ByteTensor).unsqueeze(1)
            wid_mask = mask & wid_mask
        return id_tensor.to(device()), adj_tensor.to(device()), wid_mask.to(device())


class DocGraphDataLoader:

    def __init__(self, batch_size, src_vocab, tgt_vocab, data, cache_file_prefix,
                 lazy_build=False, cache_split=1):
        self.batch_size = batch_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.batches = []
        self.curr_batch_idx = 0
        self.lazy_build = lazy_build
        self.data = None
        if self.lazy_build:
            self.data = data
        self.num_batches = int(len(data)/batch_size) + 1
        if not self.lazy_build:
            self.cache_split = max(cache_split,1)
            cache_file_meta = cache_file_prefix + "_info.txt"
            if not os.path.isfile(cache_file_meta):
                self.split_build(data, cache_file_prefix)
            self.split_load(cache_file_prefix)

    def split_load(self, cache_file_prefix):
        with open(cache_file_prefix + "_info.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
        split = int(lines[0])
        for i in range(split):
            cache_file = cache_file_prefix + "_{}.pkl".format(i+1)
            with open(cache_file, "rb") as f:
                curr_batches = pickle.load(f)
            self.batches += curr_batches
        self.shuffle()

    def split_build(self, data, cache_file_prefix):
        group_size = int(len(data) / self.cache_split)
        for i in range(self.cache_split):
            cache_file = cache_file_prefix + "_{}.pkl".format(i+1)
            curr_group = data[i * group_size:i * group_size + group_size]
            batches = self._build_batches(curr_group)
            with open(cache_file,"wb") as f:
                pickle.dump(batches, f)
        with open(cache_file_prefix + "_info.txt", "w", encoding="utf-8") as f:
            f.write(str(self.cache_split))

    def _build_batches(self, insts):
        rv = []
        idx = 0
        bar = tqdm(total=math.ceil(len(insts) / self.batch_size), desc="Building batches", ascii=True)
        while idx < len(insts):
            batch_list = insts[idx: idx+self.batch_size]
            target_seg_lists = []
            doc_seg_lists = []
            doc_sents_seg_lists = []
            doc_kw_seg_lists = []
            for inst in batch_list:
                target_seg_list = [w for w in inst[0] if len(w) > 0] + [self.tgt_vocab.eos_token] # target
                if len(target_seg_list) == 1: continue
                doc_seg_list = inst[1] # document words
                doc_seg_list = [w for w in doc_seg_list if len(w) > 0]
                doc_kw_seg_list = [w for w in inst[2] if len(w) > 0] # document keywords
                if len(doc_kw_seg_list) == 0: continue
                doc_sents_seg_list = inst[3] # document sentences
                target_seg_lists.append(target_seg_list)
                doc_seg_lists.append(doc_seg_list)
                doc_sents_seg_lists.append(doc_sents_seg_list)
                doc_kw_seg_lists.append(doc_kw_seg_list)
            tmp = list(zip(target_seg_lists, doc_seg_lists, doc_sents_seg_lists, doc_kw_seg_lists))
            tmp = sorted(tmp, key=lambda t: (len(t[1]), len(t[0]), len(t[2]), len(t[3])), reverse=True)
            target_seg_lists, doc_seg_lists, doc_sents_seg_lists, doc_kw_seg_lists = zip(*tmp)

            max_n_sents = max([len(sents) for sents in doc_sents_seg_lists])
            sents_tensors_list = []
            tmp = copy.deepcopy(doc_sents_seg_lists)
            for sents in tmp:
                while len(sents) < max_n_sents: sents.append([self.tgt_vocab.eos_token])
            for si in range(max_n_sents):
                batch_sent_i_list = []
                max_n_sent_i_words = 0
                for di in range(len(tmp)):
                    sent_i_in_doc = tmp[di][si]
                    batch_sent_i_list.append(sent_i_in_doc)
                    max_n_sent_i_words = max(max_n_sent_i_words, len(sent_i_in_doc))
                sents_vec = gen_word2idx_vec_rep(batch_sent_i_list, self.src_vocab.w2i, max_n_sent_i_words,
                                                 pad_idx=self.src_vocab.pad_token_idx,
                                                 oov_idx=self.src_vocab.oov_token_idx)
                sents_vec = torch.from_numpy(sents_vec).type(torch.LongTensor)
                sents_tensors_list.append(sents_vec)

            title_vec = gen_word2idx_vec_rep(target_seg_lists, self.tgt_vocab.w2i, max([len(l) for l in target_seg_lists]),
                                             pad_idx=self.tgt_vocab.pad_token_idx, oov_idx=self.tgt_vocab.oov_token_idx)
            doc_vec = gen_word2idx_vec_rep(doc_seg_lists, self.src_vocab.w2i, max([len(l) for l in doc_seg_lists]),
                                           pad_idx=self.src_vocab.pad_token_idx, oov_idx=self.src_vocab.oov_token_idx)
            cpy_title_wids, cpy_title_gates = gen_cpy_np(doc_seg_lists, target_seg_lists, title_vec.shape[1], self.tgt_vocab.w2i)

            doc_wid = torch.from_numpy(doc_vec).type(torch.LongTensor)
            target_g_wid = torch.from_numpy(title_vec).type(torch.LongTensor)
            target_c_wid = torch.from_numpy(cpy_title_wids).type(torch.LongTensor)
            target_c_gate = torch.from_numpy(cpy_title_gates).type(torch.FloatTensor)
            doc_wid_mask = (doc_wid != self.src_vocab.pad_token_idx).type(torch.ByteTensor).unsqueeze(1)
            target_n_tokens = (target_g_wid != self.tgt_vocab.pad_token_idx).data.sum()

            # doc keywords distance graph
            batch_doc_kw_dist_graph = BatchIDGraph(pad_indices=[self.src_vocab.pad_token_idx])
            for bi, doc_kws in enumerate(doc_kw_seg_lists):
                kw_wids = gen_word2idx_vec_rep([doc_kws], self.src_vocab.w2i, len(doc_kws),
                                               pad_idx=self.src_vocab.pad_token_idx,
                                               oov_idx=self.src_vocab.oov_token_idx,
                                               return_lists=True)[0]
                doc_words = doc_seg_lists[bi]
                edges = build_doc_keyword_dist_edges(doc_words, doc_kws)
                batch_doc_kw_dist_graph.add_inst(kw_wids, edges)

            # doc sents word overlap graph
            batch_doc_sent_word_overlap_graph = BatchIDGraph(pad_indices=[self.src_vocab.pad_token_idx])
            for bi, doc_sents in enumerate(doc_sents_seg_lists):
                edges = build_doc_sent_word_overlap_percent_edges(doc_sents, threshold=0.1)
                # this vertices is not used, instead, we dynamically supply sentence embeddings
                vertices = [idx+1 for idx in range(len(doc_sents))]
                batch_doc_sent_word_overlap_graph.add_inst(vertices, edges)

            batch = {
                DK_BATCH_SIZE: target_g_wid.shape[0],
                DK_PAD: self.tgt_vocab.pad_token_idx,
                DK_DOC_WID: doc_wid,
                DK_DOC_WID_MASK: doc_wid_mask,
                DK_DOC_SENTS_WID: sents_tensors_list,

                DK_SRC_WID: doc_wid,
                DK_SRC_WID_MASK: doc_wid_mask,
                DK_SRC_SEG_LISTS: doc_seg_lists,
                DK_TGT_GEN_WID: target_g_wid,
                DK_TGT_CPY_WID: target_c_wid,
                DK_TGT_CPY_GATE: target_c_gate,
                DK_TGT_N_TOKEN: target_n_tokens,
                DK_TGT_SEG_LISTS: target_seg_lists,

                DK_DOC_SEG_LISTS: doc_seg_lists,
                DK_TQ_SEG_LISTS: target_seg_lists,
                DK_DOC_KW_DIST_GRAPH: batch_doc_kw_dist_graph,
                DK_DOC_SENT_WORD_OVERLAP_GRAPH: batch_doc_sent_word_overlap_graph,
            }
            rv.append(batch)
            bar.update(1)
            idx += self.batch_size
        bar.close()
        return rv

    def shuffle(self):
        if self.lazy_build:
            random.shuffle(self.data)
        else:
            random.shuffle(self.batches)

    @property
    def n_batches(self):
        if self.lazy_build:
            return self.num_batches
        else:
            return len(self.batches)

    @property
    def src_vocab_size(self):
        return len(self.src_vocab.w2i)

    @property
    def tgt_vocab_size(self):
        return len(self.tgt_vocab.w2i)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        if self.lazy_build:
            return self.num_batches
        else:
            return len(self.batches)

    def next(self):
        if self.curr_batch_idx >= self.n_batches:
            self.shuffle()
            self.curr_batch_idx = 0
            raise StopIteration()
        if self.lazy_build:
            next_group_idx = self.curr_batch_idx * self.batch_size
            batch_insts = self.data[next_group_idx:next_group_idx+self.batch_size]
            next_batch = self._build_batches(batch_insts)[0]
        else:
            next_batch = self.batches[self.curr_batch_idx]
        self.curr_batch_idx += 1
        return next_batch

    def get_overlapping_data(self, loader):
        if not isinstance(loader, DocGraphDataLoader):
            print("type mismatch, no overlaps by default")
            return []
        overlapped = []
        my_data = {}
        if self.lazy_build:
            print("overlap checking is not available in lazy build mode")
            return overlapped
        for batch in self:
            for i, ctx in enumerate(batch[DK_DOC_SEG_LISTS]):
                rsp = batch[DK_TQ_SEG_LISTS][i]
                key = "|".join([" ".join(ctx), " ".join(rsp)])
                if key not in my_data:
                    my_data[key] = 0
                my_data[key] += 1
        for batch in loader:
            for i, ctx in enumerate(batch[DK_DOC_SEG_LISTS]):
                rsp = batch[DK_TQ_SEG_LISTS][i]
                key = "|".join([" ".join(ctx), " ".join(rsp)])
                if key in my_data:
                    overlapped.append(key)
        return overlapped
