import pickle
import regex as re
import utils.model_utils as mutil
from utils.lang_utils import *
from loader import DocGraphDataLoader
from data.read_data import read_doc_data
from params import prepare_params
from model import *
from utils.model_utils import model_load


def main(params):
    params.model_name = "graph_seq2attn"
    mutil.DEVICE_STR_OVERRIDE = params.device_str

    data_train, w2c_doc, w2c_query = read_doc_data("data/train_data_example.txt",
                                                   w2c_doc_limit=params.src_vocab_w2c_limit,
                                                   w2c_target_limit=params.tgt_vocab_w2c_limit)
    data_dev, _, _ = read_doc_data("data/dev_data_example.txt")
    data_test, _, _ = read_doc_data("data/test_data_example.txt")

    src_vocab_cache_file = "cache/news_vocab.pkl"
    tgt_vocab_cache_file = "cache/query_vocab.pkl"
    print("Loading dictionary")
    if os.path.isfile(src_vocab_cache_file):
        with open(src_vocab_cache_file, "rb") as f:
            src_vocab = pickle.load(f)
    else:
        src_vocab = W2IDict(
            oov_token=params.oov_token, oov_idx=params.oov_idx,
            pad_token=params.pad_token, pad_idx=params.pad_idx,
            sos_token=params.sos_token, sos_idx=params.sos_idx,
            eos_token=params.eos_token, eos_idx=params.eos_idx
        )
        src_vocab.add_words_from_word2count_dict(w2c_doc)
        with open(src_vocab_cache_file, "wb") as f:
            pickle.dump(src_vocab, f)
    params.src_vocab_size = len(src_vocab.w2i)
    print("src vocab size: ", params.src_vocab_size)
    if os.path.isfile(tgt_vocab_cache_file):
        with open(tgt_vocab_cache_file, "rb") as f:
            tgt_vocab = pickle.load(f)
    else:
        tgt_vocab = W2IDict(
            oov_token=params.oov_token, oov_idx=params.oov_idx,
            pad_token=params.pad_token, pad_idx=params.pad_idx,
            sos_token=params.sos_token, sos_idx=params.sos_idx,
            eos_token=params.eos_token, eos_idx=params.eos_idx,
        )
        tgt_vocab.add_words_from_word2count_dict(w2c_query)
        with open(tgt_vocab_cache_file, "wb") as f:
            pickle.dump(tgt_vocab, f)
    if params.same_word_embedding:
        print("Setting same vocab for src and tgt, pre-built tgt_vocab ignored!")
        tgt_vocab = src_vocab
    params.tgt_vocab_size = len(tgt_vocab.w2i)
    print("tgt vocab size: ", params.tgt_vocab_size)
    params.src_w2i = src_vocab.w2i
    params.src_i2w = src_vocab.i2w
    params.tgt_w2i = tgt_vocab.w2i
    params.tgt_i2w = tgt_vocab.i2w

    print("Preparing data loaders")
    train_loader = DocGraphDataLoader(params.batch_size, src_vocab, tgt_vocab, data_train,
                                      cache_file_prefix="cache/train_loader")
    dev_loader = DocGraphDataLoader(params.batch_size, src_vocab, tgt_vocab, data_dev,
                                    cache_file_prefix="cache/dev_loader")
    test_loader = DocGraphDataLoader(params.batch_size, src_vocab, tgt_vocab, data_test,
                                     cache_file_prefix="cache/test_loader")

    print("{} overlapped train/test instances detected".format(len(train_loader.get_overlapping_data(test_loader))))
    print("{} overlapped train/valid instances detected".format(len(train_loader.get_overlapping_data(dev_loader))))
    print("{} overlapped valid/test instances detected".format(len(dev_loader.get_overlapping_data(test_loader))))

    print("Initializing model")
    model = make_graph_seq2attn_model(None, None,
                                      params, params.src_vocab_size, params.tgt_vocab_size,
                                      same_word_embedding=params.same_word_embedding)

    criterion = LabelSmoothing(size=params.tgt_vocab_size, padding_idx=params.pad_idx, smoothing=params.smoothing_const)
    model_opt = NoamOpt(params.graph_seq2attn_decoder_hidden_size, params.noam_factor, params.noam_warm_up_steps,
                        torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), betas=(params.adam_betas_1, params.adam_betas_2),
                                         eps=params.adam_eps, weight_decay=params.adam_l2))

    completed_epochs = 0
    best_eval_result = 0
    best_eval_epoch = 0
    past_eval_results = []
    ct = params.continue_training
    smf = params.saved_model_file
    fse = params.full_eval_start_epoch
    ep = params.epochs
    if os.path.isfile(params.saved_model_file):
        print("Found saved model {}, loading".format(params.saved_model_file))
        sd = model_load(params.saved_model_file)
        params = sd[CHKPT_PARAMS]
        model.load_state_dict(sd[CHKPT_MODEL])
        model_opt.load_state_dict(sd[CHKPT_OPTIMIZER])
        best_eval_result = sd[CHKPT_BEST_EVAL_RESULT]
        best_eval_epoch = sd[CHKPT_BEST_EVAL_EPOCH]
        past_eval_results = sd[CHKPT_PAST_EVAL_RESULTS]
        completed_epochs = sd[CHKPT_COMPLETED_EPOCHS]
    params.epochs = ep
    params.continue_training = ct
    params.saved_model_file = smf
    params.full_eval_start_epoch = fse

    print(model)
    print("Model name: {}".format(params.model_name))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}".format(n_params))

    if not os.path.isfile(params.saved_model_file) or \
            (os.path.isfile(params.saved_model_file) and params.continue_training):
        print("Training")
        try:
            train_graph_seq2attn(params, model, train_loader, criterion, model_opt,
                                 completed_epochs=completed_epochs, best_eval_result=best_eval_result,
                                 best_eval_epoch=best_eval_epoch, past_eval_results=past_eval_results,
                                 dev_loader=dev_loader)
        except KeyboardInterrupt:
            print("Training interrupted")

    if len(test_loader) > 0:
        # load best model if possible
        fn = params.saved_models_dir + params.model_name + "_best.pt"
        exclude_tokens = [params.sos_token, params.eos_token, params.pad_token, "", " "]
        if os.path.isfile(fn):
            sd = model_load(fn)
            completed_epochs = sd[CHKPT_COMPLETED_EPOCHS]
            model.load_state_dict(sd[CHKPT_MODEL])
            print("Loaded best model after {} epochs of training".format(completed_epochs))
        with torch.no_grad():
            model.eval()
            write_line_to_file("doc|pred|truth|ppl", f_path=params.model_name + "_test_results.txt")
            for batch in tqdm(test_loader, desc="Test", ascii=True):
                beam_rvs = graph_seq2attn_beam_decode_batch(model, batch, tgt_vocab.sos_token_idx, tgt_vocab.i2w,
                                                            len_norm=params.bs_len_norm, gamma=params.bs_div_gamma,
                                                            max_len=params.max_decoder_seq_len, beam_width=params.beam_width_test)
                for bi in range(batch[DK_BATCH_SIZE]):
                    msg_str = "".join(batch[DK_DOC_SEG_LISTS][bi])
                    truth_rsp_seg = [w for w in batch[DK_TGT_SEG_LISTS][bi] if w not in exclude_tokens]
                    truth_rsp_str = " ".join(truth_rsp_seg)
                    truth_rsp_str = re.sub(" +", " ", truth_rsp_str)
                    best_rv = [w for w in beam_rvs[bi][3] if w not in exclude_tokens]  # word seg list
                    ppl = beam_rvs[bi][2]
                    rsp = " ".join(best_rv)
                    write_line_to_file(msg_str + "|" + rsp + "|" + truth_rsp_str + "|" + str(ppl),
                                       f_path=params.model_name + "_test_results.txt")


if __name__ == "__main__":
    args = prepare_params()
    main(args)
    print("done")
