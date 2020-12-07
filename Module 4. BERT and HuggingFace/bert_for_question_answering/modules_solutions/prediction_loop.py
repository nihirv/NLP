from .utils import set_hardware_acceleration, format_time, gpu_memory_usage
from typing import Optional, Tuple
from tqdm import tqdm
from time import time
import torch
import logging


logger = logging.getLogger(__name__)


def predict(
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        model: torch.nn.Module,
        batch_size: int,
        device_: Optional[str] = None,  # if None, it automatically detects if a GPU is available, if not uses a CPU
        disable_progress_bar: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a trained model and unseen data, performs the predictions and returns the results.
    Unlike in the fine-tuning and training stages, during prediction there's no need to build a dataloader which
    splits the set into train and validation, and randomly shuffles the training samples. We can just pass the items
    directly one by one. As we're not training, there are no training epochs either.
    :param input_ids: torch.tensor of shape (N, max_len) representing the ids of each token of the N encoded sequence
           pairs, with padding at the end up to max_len. If decoded, the input_ids will consist of a "[CLS]" token,
           followed by the question's tokens, followed by a "[SEP]" token, followed by the context's tokens, followed
           by a "[SEP]" token, followed by "[PAD]" tokens, if relevant, up to max_len.
    :param token_type_ids: torch.tensor of shape (N, max_len) where each Nth dimension is filled with 1 for token
           positions in the context text, 0 elsewhere (i.e. in question and padding)
    :param attention_masks: torch.tensor of shape (N, max_len) where each Nth dimension is filled with 1 for
           non-"[PAD]" tokens, 0 for "[PAD]" tokens.
    :param model: the model to use (must be instance of torch.nn.Module). As we're performing predictions, this must
           be a trained model.
    :param batch_size: the batch size to use for predictions. Batching samples speeds up processing.
    :param device_: if specified, the device used for the computations. Can be one of cpu, cuda, mkldnn, opengl,
           opencl, ideep, hip, msnpu. If set to None, it will default to GPU (cuda) if one is available, else it will
           use a CPU. Default: None
    :param disable_progress_bar: bool; whether to disable the tqdm progress bar. When used in production for quickly
           returning answers to a single or small set of questions, the bar might be distracting. Default: False.
    :return: pred_start: torch.tensor of shape (N) with the predicted indices of the first answer token for each answer
             pred_end: torch.tensor of shape (N) with the predicted indices of the last answer token for each answer
    """
    assert input_ids.shape == token_type_ids.shape == attention_masks.shape, "Some input shapes are wrong"

    device = set_hardware_acceleration(default=device_)
    model = model.to(device)
    model.eval()

    pred_start = torch.tensor([], dtype=torch.long, device=device)  # initialising tensors for storing results
    pred_end = torch.tensor([], dtype=torch.long, device=device)

    t_i = time()
    # batch the samples to speed up processing. We do batching manually here to avoid using DataLoader
    for batch_i in tqdm(range(0, len(input_ids), batch_size), disable=disable_progress_bar):
        batch_input_ids = input_ids[batch_i:batch_i + batch_size, :].to(device)
        batch_token_type_ids = token_type_ids[batch_i:batch_i + batch_size, :].to(device)
        batch_attention_masks = attention_masks[batch_i:batch_i + batch_size, :].to(device)
        with torch.no_grad():
            start_logits, end_logits = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                token_type_ids=batch_token_type_ids,
            )  # if we don't pass it start_positions and end_positions it won't return the loss, unlike during training

            pred_start_positions = torch.argmax(start_logits, dim=1)
            pred_end_positions = torch.argmax(end_logits, dim=1)

            pred_start = torch.cat((pred_start, pred_start_positions))
            pred_end = torch.cat((pred_end, pred_end_positions))
        if torch.cuda.is_available():
            logger.debug(f"GPU memory usage: \n{gpu_memory_usage()}")

    logger.info(f"All predictions calculated in {format_time(time() - t_i)}.")
    if torch.cuda.is_available():
        logger.info(f"GPU memory usage: \n{gpu_memory_usage()}")

    return pred_start, pred_end

