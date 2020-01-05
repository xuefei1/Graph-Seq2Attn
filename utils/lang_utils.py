import gensim
import torch
import numpy as np
import os
import copy
from nltk.corpus import stopwords
from utils.model_utils import device


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subseq_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') # upper triangular matrix
    return torch.from_numpy(subseq_mask) == 0


def make_std_mask(tgt, pad):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask.to(device())


class W2IDict:
    def __init__(self, pad_token="<PAD>", pad_idx=0,
                 oov_token="<OOV>", oov_idx=1,
                 sos_token="<SOS>", sos_idx=2,
                 eos_token="<EOS>", eos_idx=3,
                 ):
        self.pad_token = pad_token
        self.oov_token = oov_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token_idx = pad_idx
        self.oov_token_idx = oov_idx
        self.sos_token_idx = sos_idx
        self.eos_token_idx = eos_idx
        self.w2i = {
            self.pad_token : self.pad_token_idx,
            self.oov_token : self.oov_token_idx,
            self.sos_token : self.sos_token_idx,
            self.eos_token : self.eos_token_idx,
        }
        self.i2w = {v:k for k, v in self.w2i.items()}
        self._next_idx = len(self.w2i)

    def add_words_from_seg_words_list(self, seg_words_list):
        for word in seg_words_list:
            self._add_new_word(word)
        assert len(self.w2i) == len(self.i2w)

    def add_words_from_word2count_dict(self, word2count):
        for word, _ in word2count.items():
            self._add_new_word(word)
        assert len(self.w2i) == len(self.i2w)

    def _add_new_word(self, word):
        if word not in self.w2i:
            while self._next_idx in self.i2w:
                self._next_idx += 1
            self.w2i[word] = self._next_idx
            self.i2w[self._next_idx] = word
            self._next_idx += 1

    def add_words_from_w2v(self, w2v):
        if hasattr(w2v, "index2word"):
            for word in w2v.index2word:
                self._add_new_word(word)
        else:
            for w, _ in w2v.items():
                self._add_new_word(w)

    def index2word(self, index):
        if index in self.i2w:
            return self.i2w[index]
        else:
            return self.oov_token

    def items(self):
        return self.w2i.items()

    def __contains__(self, key):
        return key in self.w2i

    def __getitem__(self, word):
        if word in self.w2i:
            return self.w2i[word]
        else:
            return self.oov_token_idx

    def __setitem__(self, word, val):
        raise ValueError("Direct set is not allowed")

    def __len__(self):
        return len(self.w2i)

    def __repr__(self):
        return "W2I dict, OOV:{}, PAD:{}, SOS:{}, EOS:{}".format(self.oov_token_idx, self.pad_token_idx,
                                                                 self.sos_token_idx, self.eos_token_idx)


class W2VDict:
    def __init__(self, w2v, embedding_dim=300,
                 pad_token="<PAD>", pad_idx=0,
                 oov_token="<UNK>", oov_idx=1,
                 sos_token="<SOS>", sos_idx=2,
                 eos_token="<EOS>", eos_idx=3,
                 ):
        self._embedding_dim = embedding_dim
        self._vocab_size = 0
        self.w2v = w2v
        self.w2i = {}
        self.i2w = {}
        self.i2v = {}
        self.c2i = {}
        self.i2c = {}
        self.w2v_mat = None
        self.c2v_mat = None
        self.pad_token = pad_token
        self.oov_token = oov_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token_idx = pad_idx
        self.oov_token_idx = oov_idx
        self.sos_token_idx = sos_idx
        self.eos_token_idx = eos_idx
        # modify this to contain all special tokens
        self.special_token_indices = [pad_idx, oov_idx, sos_idx, eos_idx]
        self.trainable_oov_token_indices = [oov_idx, sos_idx, eos_idx]

    def build_vocab(self, custom_w2c=None):
        """
        Builds w2i, i2w, i2v
        :param custom_w2c: dictionary of word to count, if not None, will shrink make w2i, i2w, i2v match this vocab
        """
        self.w2i = {}
        self.i2w = {}
        self.i2v = {}
        self.w2i[self.pad_token] = self.pad_token_idx
        self.w2i[self.oov_token] = self.oov_token_idx
        self.w2i[self.sos_token] = self.sos_token_idx
        self.w2i[self.eos_token] = self.eos_token_idx
        trainable_oov_embed = np.random.randn(self._embedding_dim) # init to rand, should be trained afterwards
        trainable_sos_embed = np.random.randn(self._embedding_dim)
        trainable_eos_embed = np.random.randn(self._embedding_dim)
        trainable_eou_embed = np.random.randn(self._embedding_dim)
        self.i2v[self.pad_token_idx] = np.zeros(self._embedding_dim)
        self.i2v[self.oov_token_idx] = np.zeros(self._embedding_dim)
        self.i2v[self.sos_token_idx] = np.zeros(self._embedding_dim)
        self.i2v[self.eos_token_idx] = np.zeros(self._embedding_dim)
        if custom_w2c is not None:
            print("building vocab from custom w2c, vocab size: " + str(len(custom_w2c)))
            idx = len(self.special_token_indices)
            for w, c in custom_w2c.items():
                # if w in self.w2v: # save only words that are in the pretrained w2v dictionary, this will reduce the overall vocab size
                # or still record this word, but in i2v we map it to oov embedding, this allows this index to be trained
                self.w2i[w] = idx
                idx += 1
            print("vocab size of dictionary: " + str(len(self.w2i)))
        else:
            if hasattr(self.w2v, "index2word"):
                for k, v in enumerate(self.w2v.index2word):
                    self.w2i[v] = k+len(self.special_token_indices)
            else: # w2v should be a dict at least
                idx = len(self.special_token_indices)
                for w, _ in self.w2v.items():
                    self.w2i[w] = idx
                    idx += 1
        self.i2w = {v: k for k, v in self.w2i.items()}
        oov_count=0
        for i, w in self.i2w.items():
            if i in self.i2v: continue
            if w in self.w2v:
                self.i2v[i] = self.w2v[w]
            else:
                # assert False, "All words in i2w must already be in w2v"
                # give oov embed to this word
                oov_count += 1
                self.i2v[i] = self.i2v[self.oov_token_idx]
                self.trainable_oov_token_indices.append(i)
        print("{} words mapped to OOV when building vocab".format(oov_count))
        self._vocab_size = len(self.w2i)
        self.w2v_mat = build_i2v_mat(self._vocab_size, self._embedding_dim, self.i2v)
        self.w2v_mat = torch.from_numpy(self.w2v_mat).type(torch.FloatTensor)
        self._build_character_embedding()

    def _build_character_embedding(self, embedding_dim=200):
        # should be called at the end of build_vocab()
        self.c2i = {chr(i):i for i in range(128)} # TODO: for now only do ascii, and use 200d
        self.i2c = {v:k for k, v in self.c2i.items()}
        self.c2v_mat = np.random.randn(128, embedding_dim)
        self.c2v_mat = torch.from_numpy(self.c2v_mat).type(torch.FloatTensor)

    def get_word_idx(self, word):
        return self.w2i if word in self.w2i else self.oov_token_idx

    def get_word_from_idx(self, idx):
        return self.i2w if idx in self.i2w else self.oov_token

    def items(self):
        return self.w2i.items()

    def __len__(self):
        return self._vocab_size

    def __repr__(self):
        return "Vocab with additional tokens like OOV:{}, PAD:{}, SOS:{}, EOS:{}".format(self.oov_token, self.pad_token, self.sos_token, self.eos_token)


def expand_w2c_dict(tgt_dict, ref_dict, tgt_size):
    tgt_dict = copy.deepcopy(tgt_dict)
    ref_order = sorted([(k,v) for k,v in ref_dict.items()],key=lambda t:t[1],reverse=True)
    ref_idx = 0
    while len(tgt_dict) < tgt_size:
        w,c = ref_order[ref_idx]
        if w not in tgt_dict:
            tgt_dict[w] = 0
        tgt_dict[w] += c
        ref_idx += 1
        if ref_idx >= len(ref_order): break
    return tgt_dict


def build_i2v_mat(vocab_size, embedding_dim, i2v):
    rv = np.zeros((vocab_size, embedding_dim))
    assert len(i2v) == vocab_size
    for i, v in i2v.items():
        assert isinstance(i, int)
        if i >= vocab_size: assert False, "idx {} OOV".format(i)
        rv[i, :] = i2v[i]
    return rv


def get_char_ids_tensor(c2i, seg_lists, pad_id=0, oov_id=1):
    """
    return batch_size x seq_len x 16
    """
    max_seq_len = max([len(l) for l in seg_lists])
    rv = np.zeros((len(seg_lists), max_seq_len, 16))
    rv.fill(pad_id)
    for i, seg_list in enumerate(seg_lists):
        for j, word in enumerate(seg_list):
            for k, c in enumerate(word):
                if k >= rv.shape[2]:
                    break
                if c in c2i:
                    rv[i,j,k] = c2i[c]
                else:
                    rv[i,j,k] = oov_id
    return torch.from_numpy(rv).type(torch.LongTensor)


def _read_qa_seg_cache_file(path, delim=","):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    seg_rv = []
    for line in lines:
        line = line.rstrip()
        if line:
            seg_rv.append(line.split(delim))
    return seg_rv


def read_qa_corpus_file(path, read_lines_limit=None):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    raw_msg = []
    raw_rsp = []
    fetch_msg = True
    if read_lines_limit:
        print('Read lines limit: ' + str(read_lines_limit))
    pairs_count = 0
    for line in lines:
        if read_lines_limit and pairs_count >= read_lines_limit:
            break
        line = line.rstrip()
        if line:
            if fetch_msg:
                raw_msg.append(line)
            else:
                pairs_count += 1
                raw_rsp.append(line)
            fetch_msg = not fetch_msg
    return raw_msg, raw_rsp


def load_word_vectors(w2v_file, w2v_cache_file=None, read_delim=" ",):
    word2idx = {}
    import pickle
    if os.path.isfile(w2v_cache_file):
        fileObject = open(w2v_cache_file, 'rb')
        model = pickle.load(fileObject)
        fileObject.close()
    else:
        model = gensim.models.KeyedVectors.load_word2vec_format(
            w2v_file, binary=False)
        fileObject = open(w2v_cache_file, 'wb')
        pickle.dump(model, fileObject)
        fileObject.close()
    if hasattr(model, "index2word"):
        word2idx = {v: k for k, v in enumerate(model.index2word)}
    else:
        with open(w2v_file, encoding='utf-8') as f:
            lines = f.readlines()
        idx = 0
        for line in lines:
            tokens = line.split(read_delim)
            if len(tokens) < 10:
                continue
            word2idx[tokens[0]] = idx
            idx+=1
    idx2word = {v: k for k, v in word2idx.items()}
    return model, word2idx, idx2word


def pad_until_len(word_seg_list, target_len, pad_word="</s>"):
    for seg_list in word_seg_list:
        while len(seg_list) < target_len:
            seg_list.append(pad_word)


def one_hot_encode_word(word, word2idx, vocab_size):
    classes = np.zeros(vocab_size)
    if word in word2idx:
        idx = word2idx[word]
        classes[idx] = 1
    return classes


def _gen_vec_rep_for_tokenized_sent(word_list, max_sent_len, word_vec, embed_dim):
    rv = np.zeros((max_sent_len, embed_dim))
    i = 0
    for word in word_list:
        if i >= max_sent_len:
            break
        if word in word_vec:
            rv[i,:] = word_vec[word]
        i += 1
    return rv


def handle_OOV_for_words(words_list, dictionary, oov_token=None):
    rv = []
    for w in words_list:
        if w in dictionary:
            rv.append(w)
        elif oov_token is not None and oov_token in dictionary:
            rv.append(oov_token)
    return rv


def _read_sent_w2i_cache_file(file):
    npz_file = np.load(file)
    return npz_file['arr_0']


def remove_stop_words_en(seg_list, ):
    eng_stops = set(stopwords.words('english'))
    sent_list = [w for w in seg_list if w not in eng_stops]
    return sent_list

def _sent_word2idx_lists(seg_sent_list, word2idx, max_sent_len, oov_idx=None):
    sent_words2idx = []
    for seg_sent in seg_sent_list:
        j = 0
        tmp = []
        for word in seg_sent:
            if j >= max_sent_len:
                break
            if word in word2idx:
                tmp.append(word2idx[word])
            elif oov_idx is not None:
                tmp.append(oov_idx)
            j += 1
        sent_words2idx.append(tmp)
    return sent_words2idx


def _sent_word2idx_np(seg_sent_list, word2idx, max_sent_len, pad_idx=0, oov_idx=1):
    sent_words2idx = np.zeros((len(seg_sent_list), max_sent_len))
    if pad_idx is not None: sent_words2idx.fill(pad_idx)
    i = 0
    for seg_sent in seg_sent_list:
        j = 0
        for word in seg_sent:
            if j >= max_sent_len:
                break
            if word in word2idx:
                sent_words2idx[i,j] = word2idx[word]
            else:
                sent_words2idx[i,j] = oov_idx
            j += 1
        i += 1
    return sent_words2idx


def sents_words2idx_to_text(w2i, w2v, eos_idx=0, oov_token="<OOV>", delim=" "):
    rv = []
    max_sent_len = get_valid_vec_rep_length(w2i.reshape(-1, 1))
    for i in range(max_sent_len):
        word_idx = int(w2i[i])
        if word_idx == eos_idx >= 0:
           break
        if 0 <= word_idx <= len(w2v.index2word):
            word = w2v.index2word[word_idx]
            rv.append(word)
        else:
            rv.append(oov_token)
    return delim.join(rv)


def sents_words2idx_to_one_hot(batch_size, w2i, vocab_size, max_sent_len_idx=1):
    max_sent_len = w2i.shape[max_sent_len_idx]
    rv = np.zeros((max_sent_len, batch_size, vocab_size))
    for i in range(batch_size):
        for j in range(max_sent_len):
            idx = int(w2i[i, j])
            if idx < 0:
                continue
            rv[j, i, idx] = 1
    return rv


def _word2idx_to_one_hot(word_idx, vocab_size):
    rv = np.zeros(vocab_size)
    rv[word_idx] = 1
    return rv


def append_to_seg_sents_list(seg_sents_list, token, to_front=True):
    for sent_word_list in seg_sents_list:
        if to_front:
            sent_word_list.insert(0, token)
        else:
            sent_word_list.append(token)


def handle_msg_rsp_OOV(msg_seg_list, rsp_seg_list, dictionary, oov_token=None):
    msg_seg_rv = []
    rsp_seg_rv = []
    for i in range(len(msg_seg_list)):
        msg_words = handle_OOV_for_words(msg_seg_list[i], dictionary, oov_token=oov_token)
        rsp_words = handle_OOV_for_words(rsp_seg_list[i], dictionary, oov_token=oov_token)
        if len(msg_words) == 0 or len(rsp_words) == 0:
            continue
        msg_seg_rv.append(msg_words)
        rsp_seg_rv.append(rsp_words)
    return msg_seg_rv, rsp_seg_rv


def get_valid_vec_rep_length(vec):
    """
    Input must be embedding_size
    """
    n_instances = vec.shape[0]
    rv = 0
    for i in range(n_instances):
        val = vec[i,:]
        if np.all(val==0) or np.all(val==-1):
            return rv
        rv += 1
    return rv


def gen_word_embedding_vec_rep(seg_sents_list, embedding_size, w2v, max_sent_len,
                               time_major=False,
                               word_embedding_vec_cache_file=None,):
    if word_embedding_vec_cache_file is not None and os.path.isfile(word_embedding_vec_cache_file):
        npz_file = np.load(word_embedding_vec_cache_file)
        return npz_file['arr_0']
    n_instances = len(seg_sents_list)
    if time_major:
        rv = np.zeros((max_sent_len, n_instances, embedding_size))
    else:
        rv = np.zeros((n_instances, max_sent_len, embedding_size))
    for i in range(n_instances):
        vec_rep = _gen_vec_rep_for_tokenized_sent(seg_sents_list[i], max_sent_len, w2v, embedding_size)
        if time_major:
            rv[:, i, :] = vec_rep
        else:
            rv[i, :, :] = vec_rep
    if word_embedding_vec_cache_file:
        np.savez(word_embedding_vec_cache_file, rv)
    return rv


def gen_word2idx_vec_rep(seg_sents_list, word2idx, max_sent_len, return_lists=False,
                         pad_idx=0, oov_idx=1):
    if return_lists:
        sent_word2idx = _sent_word2idx_lists(seg_sents_list, word2idx, max_sent_len, oov_idx)
    else:
        sent_word2idx = _sent_word2idx_np(seg_sents_list, word2idx, max_sent_len, pad_idx=pad_idx, oov_idx=oov_idx)
    return sent_word2idx


def truncate_str_upto(s, upto_char, include=True):
    sp = s.split(upto_char)
    if len(sp) == 1:
        return s
    return sp[0]+upto_char if include else sp[0]

