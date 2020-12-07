from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from .utils import set_hardware_acceleration, format_time, gpu_memory_usage
from typing import Optional, Union, Tuple, Dict
import json
from tqdm import tqdm
from time import time
import torch
import logging


logger = logging.getLogger(__name__)


def _build_dataloaders(
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
        batch_size: Tuple[int, int],
        train_ratio: float = 0.9,
) -> Tuple[DataLoader, DataLoader]:
    """
    Takes the pre-processed input data and returns the train and validation dataloaders with a customizable split, for
    input into the training loop.
    :param input_ids: as described in the fine_tune_train_and_eval function documentation.
    :param token_type_ids: idem
    :param attention_masks: idem
    :param start_positions: idem
    :param end_positions: idem
    :param batch_size: idem
    :param train_ratio: idem
    :return: train_dataloader: the Dataloader for the train dataset.
             valid_dataloader: the Dataloader for the validation dataset.
    """
    dataset = TensorDataset(
        input_ids, token_type_ids, attention_masks, start_positions, end_positions
    )
    train_size = int(train_ratio * len(dataset))
    valid_size = len(dataset) - train_size
    logger.info(
        f"The input dataset has {len(dataset)} input samples, which have been split into {train_size} training "
        f"samples and {valid_size} validation samples."
    )
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size[0], sampler=RandomSampler(train_dataset))  # could do with shuffle=True instead?
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size[1], sampler=SequentialSampler(valid_dataset))
    logger.info(f"There are {len(train_dataloader)} training batches and {len(valid_dataloader)} validation batches.")
    return train_dataloader, valid_dataloader


def fine_tune_train_and_eval(
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
        batch_size: Tuple[int, int],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_ratio: float = 0.9,
        training_epochs: int = 3,
        lr_scheduler_warmup_steps: int = 0,
        save_model_path: Optional[str] = None,
        save_stats_dict_path: Optional[str] = None,
        device_: Optional[str] = None  # if None, it automatically detects if a GPU is available, if not uses a CPU
) -> Tuple[torch.nn.Module, Dict[str, Dict[str, Union[float, str]]]]:
    """
    Performs the fine tuning of the model and returns the trained model as well as a dictionary with evaluation
    statistics at each epochs which can be used to check overfitting and training time.
    :param input_ids: torch.tensor of shape (N, max_len) representing the ids of each token of the N encoded sequence
           pairs, with padding at the end up to max_len. If decoded, the input_ids will consist of a "[CLS]" token,
           followed by the question's tokens, followed by a "[SEP]" token, followed by the context's tokens, followed
           by a "[SEP]" token, followed by "[PAD]" tokens, if relevant, up to max_len.
    :param token_type_ids: torch.tensor of shape (N, max_len) where each Nth dimension is filled with 1 for token
           positions in the context text, 0 elsewhere (i.e. in question and padding)
    :param attention_masks: torch.tensor of shape (N, max_len) where each Nth dimension is filled with 1 for
           non-"[PAD]" tokens, 0 for "[PAD]" tokens.
    :param start_positions: torch.tensor of shape (N) containing the index of the first answer token for each answer
    :param end_positions: torch.tensor of shape (N) containing the index of the last answer token for each answer
    :param batch_size: a tuple of 2 integers, representing the batch size of the train and validation dataloaders
           respectively.
    :param model: the model to use (must be instance of torch.nn.Module). For question answering,
           transformers.BertForQuestionAnswering is recommended.
    :param optimizer: the optimizer to use for the model (must be instance of torch.optim.Optimizer).
    :param train_ratio: the train / (train + validation) split ratio. Default: 0.9 (i.e. 90% of the input data will
           go to the train dataloader and 10% to the validation dataloader). The split is random.
    :param training_epochs: the number of training epochs. Default: 3.
    :param lr_scheduler_warmup_steps: the number of warmup steps of the learning rate scheduler. Default: 0.
           Note: the purpose of this scheduler is to update the learning rate over the course of the training. It is
           preferable for the learning rate to gradually get smaller and smaller so that training makes gradually
           finer adjustments to the weights as the loss gets smaller.
    :param save_model_path: if specified, the path where to save the model (should have '.pt' extension). Default: None.
    :param save_stats_dict_path: if specified, the path where to save the dictionary of statistics (should have
           '.json' extension). Default: None.
    :param device_: if specified, the device used for the computations. Can be one of cpu, cuda, mkldnn, opengl,
           opencl, ideep, hip, msnpu. If set to None, it will default to GPU (cuda) if one is available, else it will
           use a CPU. Default: None
    :return: model: the fine tuned model.
             training_stats: a dictionary with a number of statistics. For each epoch, the training loss, validation
             loss, validation accuracy, training time and validation time are included.
    """
    assert all([isinstance(i, torch.Tensor) for i in [
        input_ids, token_type_ids, attention_masks, start_positions, end_positions
    ]]), "Some inputs are not tensors. When training, start_positions and end_positions must be tensors, not lists."
    assert input_ids.shape == token_type_ids.shape == attention_masks.shape, "Some input shapes are incompatible."
    assert input_ids.shape[0] == len(start_positions) == len(end_positions), "Some input shapes are incompatible"

    train_dataloader, valid_dataloader = _build_dataloaders(
        input_ids, token_type_ids, attention_masks, start_positions, end_positions, batch_size, train_ratio
    )
    training_steps = training_epochs * len(train_dataloader)  # epochs * number of batches
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=lr_scheduler_warmup_steps, num_training_steps=training_steps
    )
    device = set_hardware_acceleration(default=device_)
    model = model.to(device)
    training_stats = {}
    for epoch in (range(training_epochs)):
        logger.info(f"Training epoch {epoch + 1} of {training_epochs}. Running training.")
        t_i = time()
        model.train()
        cumulative_train_loss_per_epoch = 0.
        for batch_num, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            logger.debug(f"Running training batch {batch_num + 1} of {len(train_dataloader)}.")
            batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_start_positions, batch_end_positions = \
                batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
            model.zero_grad()
            #  model.zero_grad() and optimizer.zero_grad() are the same IF all model parameters are in that optimizer.
            #  It could be safer to call model.zero_grad() if you have two or more optimizers for one model.
            loss, start_logits, end_logits = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                token_type_ids=batch_token_type_ids,
                start_positions=batch_start_positions,
                end_positions=batch_end_positions
            )  # BertForQuestionAnswering uses CrossEntropyLoss by default, no need to calculate explicitly

            cumulative_train_loss_per_epoch += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            # clipping the norm of the gradients to 1.0 to help prevent the "exploding gradients" issues.
            optimizer.step()  # update model parameters
            lr_scheduler.step()  # update the learning rate

        average_training_loss_per_batch = cumulative_train_loss_per_epoch / len(train_dataloader)
        training_time = format_time(time() - t_i)
        logger.info(f"Epoch {epoch + 1} took {training_time} to train.")
        logger.info(f"Average training loss: {average_training_loss_per_batch}. \n Running validation.")
        if torch.cuda.is_available():
            logger.info(f"GPU memory usage: \n{gpu_memory_usage()}")

        t_i = time()
        model.eval()

        pred_start = torch.tensor([], dtype=torch.long, device=device)  # initialising tensors for storing results
        pred_end = torch.tensor([], dtype=torch.long, device=device)
        true_start = torch.tensor([], dtype=torch.long, device=device)
        true_end = torch.tensor([], dtype=torch.long, device=device)

        cumulative_eval_loss_per_epoch = 0
        cumulative_eval_accuracy_per_epoch = 0  # WE DO THIS DIFFERENTLY. SHALL WE REMOVE THIS?

        for batch_num, batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
            logger.info(f"Running validation batch {batch_num + 1} of {len(valid_dataloader)}.")
            batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_start_positions, batch_end_positions = \
                batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
            with torch.no_grad():
                loss, start_logits, end_logits = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_masks,
                    token_type_ids=batch_token_type_ids,
                    start_positions=batch_start_positions,
                    end_positions=batch_end_positions
                )  # if we pass it the true labels, i.e. start_positions and end_positions it will also return the loss
                cumulative_eval_loss_per_epoch += loss.item()
                # SHALL WE MOVE THE BELOW TO CPU AND NUMPY OR KEEP GPU AND PYTORCH?

                pred_start_positions = torch.argmax(start_logits, dim=1)
                pred_end_positions = torch.argmax(end_logits, dim=1)

                pred_start = torch.cat((pred_start, pred_start_positions))
                pred_end = torch.cat((pred_end, pred_end_positions))
                true_start = torch.cat((true_start, batch_start_positions))
                true_end = torch.cat((true_end, batch_end_positions))
            if torch.cuda.is_available():
                logger.debug(f"GPU memory usage: \n{gpu_memory_usage()}")

        total_correct_start = int(sum(pred_start == true_start))
        total_correct_end = int(sum(pred_end == true_end))
        total_correct = total_correct_start + total_correct_end
        total_indices = len(true_start) + len(true_end)

        average_validation_accuracy_per_epoch = total_correct / total_indices
        average_validation_loss_per_batch = cumulative_eval_loss_per_epoch / len(valid_dataloader)
        valid_time = format_time(time() - t_i)
        logger.info(f"Epoch {epoch + 1} took {valid_time} to validate.")
        logger.info(f"Average validation loss: {average_validation_loss_per_batch}.")
        logger.info(f"Average validation accuracy (out of 1): {average_validation_accuracy_per_epoch}.")
        if torch.cuda.is_available():
            logger.info(f"GPU memory usage: \n{gpu_memory_usage()}")

        training_stats[f"epoch_{epoch + 1}"] = {
            "training_loss": average_training_loss_per_batch,
            "valid_loss": average_validation_loss_per_batch,
            "valid_accuracy": average_validation_accuracy_per_epoch,
            "training_time": training_time,
            "valid_time": valid_time
        }
    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)
    if save_stats_dict_path is not None:
        with open(save_stats_dict_path, "w") as file:
            json.dump(training_stats, file)
    return model, training_stats
