import os
import argparse
import torch

def prepare_params():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    d = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(d + os.path.sep + "cache"): os.makedirs(d + os.path.sep + "cache")
    if not os.path.exists(d + os.path.sep + "saved_models"): os.makedirs(d + os.path.sep + "saved_models")
    parser = argparse.ArgumentParser(description="Input control args for the model")

    # General
    parser.add_argument("-batch_size", help="Batch size", type=int,
                        default=128, required=False)
    parser.add_argument("-epochs", help="Training epochs", type=int,
                        default=40, required=False)
    parser.add_argument("-word_embedding_dim", help="Embedding dimension", type=int,
                        default=300, required=False)
    parser.add_argument("-max_decoder_seq_len", help="Max decoder sequence len", type=int,
                        default=10, required=False)
    parser.add_argument("-eval_metric", help="Determining metric name in eval to index performance dict", type=str,
                        default="rouge_1", required=False)
    parser.add_argument("-device_str", help="Device string", type=str,
                        default=None, required=False)

    # Adam
    parser.add_argument("-adam_betas_1", help="Beta 1 for Adam optimizer", type=float,
                        default=0.9, required=False)
    parser.add_argument("-adam_betas_2", help="Beta 2 for Adam optimizer", type=float,
                        default=0.999, required=False)
    parser.add_argument("-adam_eps", help="Epsilon for Adam optimizer", type=float,
                        default=1e-8, required=False)
    parser.add_argument("-adam_l2", help="L2 penalty for Adam optimizer", type=float,
                        default=0.000, required=False)

    # NoamOptimizer
    parser.add_argument("-noam_warm_up_steps", help="Warm up steps of the Noam optimizer", type=int,
                        default=2000, required=False)
    parser.add_argument("-noam_factor", help="Factor for Noam optimizer", type=float,
                        default=1.0, required=False)

    # label smoothing criterion
    parser.add_argument("-smoothing_const", help="Label smoothing constant", type=float,
                        default=0.1, required=False)

    # LRDecayOptimizer
    parser.add_argument("-lrd_initial_lr", help="Training initial lr", type=float,
                        default=0.001, required=False)
    parser.add_argument("-lrd_min_lr", help="Training min lr", type=float,
                        default=0.0001, required=False)
    parser.add_argument("-lrd_lr_decay_factor", help="LR decay factor", type=float,
                        default=0.5, required=False)
    parser.add_argument("-lrd_past_lr_scores_considered", help="Past lr loss considered", type=int,
                        default=1, required=False)
    parser.add_argument("-lrd_max_fail_limit", help="Max bad count for lr update", type=int,
                        default=1, required=False)

    # gcn
    parser.add_argument("-sent_gcn_hidden_size", help="sentence dependency gcn hidden", type=int,
                        default=64, required=False)
    parser.add_argument("-sent_gcn_output_size", help="sentence dependency gcn output", type=int,
                        default=8, required=False)
    parser.add_argument("-sent_gcn_output_aggregated_size", help="output size aggregated over all sentences", type=int,
                        default=2048, required=False)
    parser.add_argument("-doc_merge_gcn_input_size", help="doc merge gcn input", type=int,
                        default=512, required=False)
    parser.add_argument("-doc_merge_gcn_hidden_size", help="doc merge gcn hidden", type=int,
                        default=64, required=False)
    parser.add_argument("-doc_gcn_output_size", help="doc gcn output", type=int,
                        default=128, required=False)
    parser.add_argument("-doc_kw_dist_gcn_hidden_size", help="doc kw dist gcn hidden", type=int,
                        default=64, required=False)
    parser.add_argument("-sent_gcn_layers", help="sentence dependency gcn layers", type=int,
                        default=2, required=False)
    parser.add_argument("-doc_kw_dist_gcn_layers", help="doc kw dist gcn layers", type=int,
                        default=2, required=False)
    parser.add_argument("-doc_merge_gcn_layers", help="doc merge gcn layers", type=int,
                        default=2, required=False)
    parser.add_argument("-sent_gcn_dropout_prob", help="sentence dependency dropout prob", type=float,
                        default=0.1, required=False)
    parser.add_argument("-doc_kw_dist_gcn_dropout_prob", help="doc kw dist gcn dropout prob", type=float,
                        default=0.1, required=False)
    parser.add_argument("-doc_merge_gcn_dropout_prob", help="doc merge gcn dropout prob", type=float,
                        default=0.1, required=False)

    parser.add_argument("-graph_seq2attn_num_attn_heads", help="Encoder hidden size", type=int,
                        default=8, required=False)
    parser.add_argument("-graph_seq2attn_teacher_forcing_ratio", help="Teacher forcing ratio", type=float,
                        default=1.0, required=False)
    parser.add_argument("-graph_seq2attn_encoder_hidden_size", help="Encoder hidden size", type=int,
                        default=128, required=False)
    parser.add_argument("-graph_seq2attn_context_hidden_size", help="Context hidden size", type=int,
                        default=128, required=False)
    parser.add_argument("-graph_seq2attn_decoder_hidden_size", help="Decoder hidden size", type=int,
                        default=128, required=False)
    parser.add_argument("-graph_seq2attn_decoder_ff_ratio", help="FF hidden ratio", type=float,
                        default=4.0, required=False)
    parser.add_argument("-graph_seq2attn_context_ff_ratio", help="FF hidden ratio", type=float,
                        default=4.0, required=False)
    parser.add_argument("-graph_seq2attn_encoder_dropout_prob", help="Encoder dropout prob", type=float,
                        default=0.0, required=False)
    parser.add_argument("-graph_seq2attn_context_dropout_prob", help="Context dropout prob", type=float,
                        default=0.1, required=False)
    parser.add_argument("-graph_seq2attn_decoder_dropout_prob", help="Decoder dropout prob", type=float,
                        default=0.1, required=False)
    parser.add_argument("-graph_seq2attn_encoder_type", help="Encoder RNN type", type=str,
                        default="gru", required=False)
    parser.add_argument("-graph_seq2attn_context_type", help="Context RNN type", type=str,
                        default="gru", required=False)
    parser.add_argument("-graph_seq2attn_num_encoder_layers", help="Encoder number of layers", type=int,
                        default=1, required=False)
    parser.add_argument("-graph_seq2attn_num_context_layers", help="Context number of layers", type=int,
                        default=1, required=False)
    parser.add_argument("-graph_seq2attn_num_decoder_layers", help="Decoder number of layers", type=int,
                        default=2, required=False)
    parser.add_argument("-graph_seq2attn_encoder_rnn_dir", help="RNN direction", type=int,
                        default=2, required=False)
    parser.add_argument("-graph_seq2attn_context_rnn_dir", help="RNN direction", type=int,
                        default=2, required=False)
    parser.add_argument("-graph_seq2attn_doc_merge_gcn_hidden_size", help="doc merge gcn hidden", type=int,
                        default=128, required=False)
    parser.add_argument("-graph_seq2attn_doc_kw_dist_gcn_hidden_size", help="doc kw dist gcn hidden", type=int,
                        default=128, required=False)
    parser.add_argument("-graph_seq2attn_doc_kw_dist_gcn_dropout_prob", help="doc kw dist gcn dropout prob", type=float,
                        default=0.1, required=False)
    parser.add_argument("-graph_seq2attn_doc_merge_gcn_dropout_prob", help="doc merge gcn dropout prob", type=float,
                        default=0.1, required=False)
    parser.add_argument("-graph_seq2attn_doc_kw_dist_gcn_layers", help="doc kw dist gcn layers", type=int,
                        default=2, required=False)
    parser.add_argument("-graph_seq2attn_doc_merge_gcn_layers", help="doc merge gcn layers", type=int,
                        default=2, required=False)
    parser.add_argument("-graph_seq2attn_pool_factor", help="pool factor", type=int,
                        default=2, required=False)

    # beam search
    parser.add_argument("-beam_width_test", help="Beam search width", type=int,
                        default=4, required=False)
    parser.add_argument("-beam_width_eval", help="Beam search width", type=int,
                        default=2, required=False)
    parser.add_argument("-beam_width", help="Beam search width", type=int,
                        default=4, required=False)
    parser.add_argument("-bs_len_norm", help="Beam search length norm", type=float,
                        default=0.0, required=False)
    parser.add_argument("-bs_div_gamma", help="Gamma controlling beam search diversity", type=float,
                        default=0.0, required=False)

    # embedding
    parser.add_argument("-use_pretrained_embedding", help="Use pre-trained word vectors", type=str,
                        default="true", required=False)
    parser.add_argument("-same_word_embedding", help="Same source target word embedding", type=str,
                        default="false", required=False)
    parser.add_argument("-src_embed_further_training", help="Further train source word embedding", type=str,
                        default="true", required=False)
    parser.add_argument("-tgt_embed_further_training", help="Further train target word embedding", type=str,
                        default="true", required=False)
    parser.add_argument("-src_vocab_w2c_limit", help="Source vocab limit", type=int,
                        default=40000, required=False)
    parser.add_argument("-tgt_vocab_w2c_limit", help="Source vocab limit", type=int,
                        default=20000, required=False)

    # params that usually do not change once set
    parser.add_argument("-seed", help="Seed", type=int,
                        default=12345, required=False)
    parser.add_argument("-test_amount", help="How many instances from training data are used for testing", type=int,
                        default=128, required=False)
    parser.add_argument("-validation_amount", help="How many instances from training data are used for validation", type=int,
                        default=128, required=False)
    parser.add_argument("-max_gradient_norm", help="Max grad norm", type=float,
                        default=5.0, required=False)
    parser.add_argument("-oov_idx", help="Out-Of-Vocab token", type=int,
                        default=1, required=False)
    parser.add_argument("-sos_idx", help="Start-Of-Sequence token", type=int,
                        default=2, required=False)
    parser.add_argument("-eos_idx", help="End-Of-Sequence token", type=int,
                        default=3, required=False)
    parser.add_argument("-eou_idx", help="End-Of-Sequence token", type=int,
                        default=4, required=False)
    parser.add_argument("-pad_idx", help="Padding token", type=int,
                        default=0, required=False)
    parser.add_argument("-oov_token", help="Out-Of-Vocab token", type=str,
                        default="<UNK>", required=False)
    parser.add_argument("-sos_token", help="Start-Of-Sequence token", type=str,
                        default="<SOS>", required=False)
    parser.add_argument("-eos_token", help="End-Of-Sequence token", type=str,
                        default="<EOS>", required=False)
    parser.add_argument("-pad_token", help="Padding token", type=str,
                        default="<PAD>", required=False)

    # bool args
    parser.add_argument("-plot", help="Plot loss with visdom", required=False, action='store_true')
    parser.add_argument("-verbose", help="Verbose when training", required=False, action='store_true')
    parser.add_argument("-test_play", help="Enables test_play", required=False, action='store_true')
    parser.add_argument("-output_to_file", help="Enables outputting important console prints to a file", required=False, action='store_true')
    parser.add_argument("-continue_training", help="Enables continue training", required=False, action='store_true')
    parser.add_argument("-invalidate_data_cache", help="Reload all data loaders", required=False, action='store_true')
    parser.add_argument("-multi_turn_eval", help="Controls what evaluation functions to call", type=str,
                        default="false", required=False)
    parser.add_argument("-stop_train_by_score", help="Stops training by checking past scores", type=str,
                        default="true", required=False)
    parser.add_argument("-best_performing_checkpoint", help="Always save the best model", type=str,
                        default="true", required=False)
    parser.add_argument("-most_recent_checkpoint", help="Always save the most recently trained model", type=str,
                        default="true", required=False)

    # value args
    parser.add_argument("-dataset_name", help="For selecting datesets", type=str,
                        default="dataset", required=False)
    parser.add_argument("-model_name", help="For checkpoint purposes", type=str,
                        default="model", required=False)
    parser.add_argument("-test_play_mode", help="Test play mode", type=str,
                        default="demo", required=False)
    parser.add_argument("-word_delim", help="word delimiter", type=str,
                        default=" ", required=False)
    parser.add_argument("-logs_dir", help="logs directory", type=str,
                        default="logs"+os.path.sep, required=False)
    parser.add_argument("-saved_model_file", help="Checkpoint file name", type=str,
                        default=d+os.path.sep+"saved_models"+os.path.sep+"default_name.pt", required=False)
    parser.add_argument("-saved_criterion_file", help="Checkpoint file name", type=str,
                        default=d + os.path.sep + "saved_models" + os.path.sep + "default_name.pt", required=False)
    parser.add_argument("-saved_optimizer_file", help="Checkpoint file name", type=str,
                        default=d + os.path.sep + "saved_models" + os.path.sep + "default_name.pt", required=False)
    parser.add_argument("-saved_models_dir", help="Checkpoint file folder", type=str,
                        default=d+os.path.sep+"saved_models"+os.path.sep, required=False)
    parser.add_argument("-full_eval_start_epoch", help="Full decoding start at which epoch", type=int,
                        default=2, required=False)
    parser.add_argument("-full_eval_every_epoch", help="Full decoding every how many epochs", type=int,
                        default=1, required=False)
    parser.add_argument("-checkpoint_every_epoch", help="When to checkpoint", type=int,
                        default=1, required=False)
    parser.add_argument("-checkpoint_init_epoch", help="Checkpoint file name start", type=int,
                        default=0, required=False)
    parser.add_argument("-past_eval_scores_considered", help="Number of past eval scores considered", type=int,
                        default=10, required=False)
    parser.add_argument("-word_vec_file", help="W2V file", type=str,
                        default=d+os.path.sep+"word_vecs"+os.path.sep+"glove_6B_300d_w2v.txt", required=False)
    parser.add_argument("-vocab_cache_file", help="Vocab cache file", type=str,
                        default=d+os.path.sep+"cache"+os.path.sep+"vocab.pkl", required=False)
    parser.add_argument("-train_loader_cache_file", help="Train loader cache file", type=str,
                        default=d+os.path.sep+"cache"+os.path.sep+"train_loader.pkl", required=False)
    parser.add_argument("-valid_loader_cache_file", help="Valid loader cache file", type=str,
                        default=d+os.path.sep+"cache"+os.path.sep+"valid_loader.pkl", required=False)
    parser.add_argument("-test_loader_cache_file", help="Test loader cache file", type=str,
                        default=d+os.path.sep+"cache"+os.path.sep+"test_loader.pkl", required=False)
    parser.add_argument("-model_load_map_location", help="map_location argument in torch.load()", type=str,
                        default="auto", required=False)

    rv = parser.parse_args()

    # map_location determines where to load the model
    # map_location=lambda storage, loc: storage # to CPU
    # map_location=lambda storage, loc: storage.cuda(1) # to GPU 1
    # map_location = {"cuda1":"cuda0"} # from GPU 0 to GPU 1
    if rv.model_load_map_location == "cpu":
        rv.map_location = lambda storage, loc:storage
    elif rv.model_load_map_location == "cuda0":
        rv.map_location = lambda storage, loc: storage.cuda(0)
    elif rv.model_load_map_location == "cuda1":
        rv.map_location = lambda storage, loc: storage.cuda(1)
    else:
        rv.map_location = None

    rv.use_pretrained_embedding = str2bool(rv.use_pretrained_embedding)
    rv.multi_turn_eval = str2bool(rv.multi_turn_eval)
    rv.stop_train_by_score = str2bool(rv.stop_train_by_score)
    rv.best_performing_checkpoint = str2bool(rv.best_performing_checkpoint)
    rv.most_recent_checkpoint = str2bool(rv.most_recent_checkpoint)
    rv.same_word_embedding = str2bool(rv.same_word_embedding)
    rv.src_embed_further_training = str2bool(rv.src_embed_further_training)
    rv.tgt_embed_further_training = str2bool(rv.tgt_embed_further_training)

    if torch.cuda.is_available() and rv.seed > 0:
        torch.cuda.manual_seed(rv.seed)
        print('My cuda seed is {0}'.format(torch.cuda.initial_seed()))
    torch.manual_seed(rv.seed)
    print('My seed is {0}'.format(torch.initial_seed()))
    return rv
