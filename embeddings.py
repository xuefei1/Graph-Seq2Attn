import math
import torch
import torch.nn as nn
from constants import *
from utils.model_utils import device


class TrainableEmbedding(nn.Module):

    def __init__(self, d_model, vocab, padding_idx=0):
        super(TrainableEmbedding, self).__init__()
        self.embed_layer = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        if torch.isnan(x).any():
            print(x)
            assert False, "NaN detected in input"
        rv = self.embed_layer(x)
        if torch.isnan(rv).any():
            torch.set_printoptions(threshold=10000)
            print(torch.isnan(self.embed_layer.weight).any())
            print(self.embed_layer.weight)
            assert False, "NaN detected in embedding"
        return rv


class PreTrainedWordEmbedding(nn.Module):

    def __init__(self, word_mat, d_model, allow_further_training=False):
        super(PreTrainedWordEmbedding, self).__init__()
        assert word_mat.shape[1] == d_model
        self.d_model = d_model
        self.embedding_layer = nn.Embedding.from_pretrained(word_mat, freeze=not allow_further_training)

    def forward(self, x):
        rv = self.embedding_layer(x)
        return rv


class IOVPreTrainedEmbeddings(nn.Module):

    def __init__(self, w2v_mat, d_model, oov_ids):
        super(IOVPreTrainedEmbeddings, self).__init__()
        self.d_model = d_model
        self.embedding_dim = w2v_mat.shape[1]
        self.embedding_layer = nn.Embedding.from_pretrained(w2v_mat, freeze=True)
        self.oov_idx_map = {}
        for i, oov_id in enumerate(oov_ids):
            self.oov_idx_map[oov_id] = i
        self.oov_vocab_size = len(self.oov_idx_map)
        self.oov_embed_layer = nn.Embedding(self.oov_vocab_size, self.embedding_dim)

    def forward(self, x):
        # x must be batch_size x seq_len
        rv = torch.zeros(x.shape[0], x.shape[1], self.embedding_dim).type(torch.FloatTensor).to(device())
        for bi in range(x.shape[0]):
            for wi in range(x.shape[1]):
                w_idx = x[bi, wi].cpu().item()
                if w_idx in self.oov_idx_map:
                    oov_idx = torch.ones(1,1).fill_(self.oov_idx_map[w_idx]).type(torch.LongTensor).to(device())
                    embed = self.oov_embed_layer(oov_idx).squeeze()
                else:
                    embed = self.embedding_layer(x[bi, wi]).squeeze()
                rv[bi, wi, :] = embed
        return rv


class DictWIDEmbedding(nn.Module):

    def __init__(self, embed_layer):
        super(DictWIDEmbedding, self).__init__()
        self.embedding_layer = embed_layer

    def forward(self, data_dict):
        x = data_dict[DK_SRC_WID].to(device())
        rv = self.embedding_layer(x)
        return rv


class ResizeWrapperEmbedding(nn.Module):

    def __init__(self, d_model, embed_layer):
        super(ResizeWrapperEmbedding, self).__init__()
        self.embed_layer = embed_layer
        self.d_model = d_model
        self.resize_layer = nn.Linear(embed_layer.d_model, self.d_model, bias=False)

    def forward(self, x):
        rv = self.embed_layer(x)
        rv = self.resize_layer(rv)
        return rv


class ResizeDictWrapperEmbedding(nn.Module):

    def __init__(self, d_model, embed_layer):
        super(ResizeDictWrapperEmbedding, self).__init__()
        self.embed_layer = embed_layer
        self.d_model = d_model
        self.resize_layer = nn.Linear(embed_layer.d_model, self.d_model, bias=False)

    def forward(self, data_dict):
        w_id = data_dict[DK_SRC_WID].to(device())
        w_embed = self.embed_layer(w_id)
        rv = self.resize_layer(w_embed)
        return rv


class SQuADQGDictEmbedding(nn.Module):

    def __init__(self, d_model, word_embed, iob_embed, pos_embed, ner_embed, resize=False):
        super(SQuADQGDictEmbedding, self).__init__()
        self.d_model = d_model
        self.word_embed = word_embed
        self.iob_embed = iob_embed
        self.pos_embed = pos_embed
        self.ner_embed = ner_embed
        self.resize_layer = nn.Linear(word_embed.d_model+iob_embed.d_model+pos_embed.d_model+ner_embed.d_model,
                                      self.d_model) if resize else None

    def forward(self, data_dict):
        w_id = data_dict[DK_SRC_WID].to(device())
        w_embed = self.word_embed(w_id)
        iob = data_dict[DK_SRC_IOB].to(device())
        iob_embed = self.iob_embed(iob)
        pos = data_dict[DK_SRC_POS].to(device())
        pos_embed = self.pos_embed(pos)
        ner = data_dict[DK_SRC_NER].to(device())
        ner_embed = self.ner_embed(ner)
        rv = torch.cat([w_embed, iob_embed, pos_embed, ner_embed], dim=2)
        if self.resize_layer is not None:
            rv = self.resize_layer(rv)
        assert rv.shape[2] == self.d_model, "Embedding size does not match expect encoder input size"
        return rv


class WIDDictEmbedding(nn.Module):

    def __init__(self, d_model, word_embed, resize=False):
        super(WIDDictEmbedding, self).__init__()
        self.d_model = d_model
        self.word_embed = word_embed
        self.resize_layer = nn.Linear(word_embed.d_model, self.d_model) if resize else None

    def forward(self, data_dict):
        w_id = data_dict[DK_SRC_WID].to(device())
        w_embed = self.word_embed(w_id)
        if self.resize_layer is not None:
            w_embed = self.resize_layer(w_embed)
        assert w_embed.shape[2] == self.d_model, "Embedding size does not match expect encoder input size"
        return w_embed


class ConceptDictEmbedding(nn.Module):

    def __init__(self, d_model, word_embed, resize=False):
        super(ConceptDictEmbedding, self).__init__()
        self.d_model = d_model
        self.word_embed = word_embed
        self.resize_layer = nn.Linear(word_embed.d_model, self.d_model) if resize else None

    def forward(self, data_dict):
        w_id = data_dict[DK_SRC_WID].to(device())
        w_embed = self.word_embed(w_id)
        if self.resize_layer is not None:
            w_embed = self.resize_layer(w_embed)
        assert w_embed.shape[2] == self.d_model, "Embedding size does not match expect encoder input size"
        return w_embed


class DialogDictEmbedding(nn.Module):

    def __init__(self, d_model, word_embed, resize=False):
        super(DialogDictEmbedding, self).__init__()
        self.d_model = d_model
        self.word_embed = word_embed
        self.resize_layer = nn.Linear(word_embed.d_model, self.d_model) if resize else None

    def forward(self, data_dict):
        w_id = data_dict[DK_SRC_WID].to(device())
        w_embed = self.word_embed(w_id)
        if self.resize_layer is not None:
            w_embed = self.resize_layer(w_embed)
        assert w_embed.shape[2] == self.d_model, "Embedding size does not match expect encoder input size"
        return w_embed


class DocGCNDictMergeEmbedding(nn.Module):

    def __init__(self, pos_embed=None):
        super(DocGCNDictMergeEmbedding, self).__init__()
        self.pos_embed = pos_embed

    def forward(self, data_dict, merge_tensor):
        overlap_graph = data_dict[DK_DOC_SENT_WORD_OVERLAP_GRAPH]
        # overlap_graph = data_dict[DK_DOC_SENT_PAIR_TFIDF_SIM_GRAPH]
        # overlap_graph = data_dict[DK_DOC_SENT_MEAN_TFIDF_SIM_GRAPH]
        _, overlap_adj, batch_sents_mask = overlap_graph.get_tensors()
        assert merge_tensor.shape[1] == overlap_adj.shape[1]
        if self.pos_embed is not None:
            merge_tensor = self.pos_embed(merge_tensor)
        return merge_tensor, overlap_adj, batch_sents_mask


class DocGCNKWDistDictEmbedding(nn.Module):

    def __init__(self, d_model, word_embed):
        super(DocGCNKWDistDictEmbedding, self).__init__()
        self.d_model = d_model
        self.word_embed = word_embed

    def forward(self, data_dict):
        kw_dist_graph = data_dict[DK_DOC_KW_DIST_GRAPH]
        kwids, kw_dist_adj, mask = kw_dist_graph.get_tensors()
        kw_embed = self.word_embed(kwids)
        return kw_embed, kw_dist_adj, mask
