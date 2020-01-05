import torch
from constants import *


DEVICE_STR_OVERRIDE = None


def device(device_str=DEVICE_STR_OVERRIDE, ref_tensor=None):
    if ref_tensor is not None:
        return ref_tensor.get_device()
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_str)


def _save_model(file_path, epoch, model, optimizer, metadata={}):
    sv = {
        CHKPT_COMPLETED_EPOCHS: epoch,
        CHKPT_MODEL: model.state_dict(),
        CHKPT_OPTIMIZER: optimizer.state_dict(),
        CHKPT_METADATA: metadata
    }
    torch.save(sv, file_path)


def load_transformer_dict(file_path):
    return torch.load(file_path, map_location=lambda storage,loc:storage)


def model_checkpoint(file_path, epoch, model, optimizer, params,
                     past_eval_results, best_eval_result, best_eval_epoch, metadata={}):
    sv = {
        CHKPT_COMPLETED_EPOCHS: epoch,
        CHKPT_MODEL: model.state_dict(),
        CHKPT_OPTIMIZER: optimizer.state_dict(),
        CHKPT_PARAMS: params,
        CHKPT_PAST_EVAL_RESULTS: past_eval_results,
        CHKPT_BEST_EVAL_RESULT: best_eval_result,
        CHKPT_BEST_EVAL_EPOCH: best_eval_epoch,
        CHKPT_METADATA: metadata
    }
    torch.save(sv, file_path)


def model_load(file_path):
    return torch.load(file_path, map_location=lambda storage,loc:storage)

