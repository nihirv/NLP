import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Union
from transformers import PreTrainedTokenizerBase
import logging
from time import time
from .utils import format_time, set_hardware_acceleration


logger = logging.getLogger(__name__)


class DatasetEncoder:
    """
    This class handles all the preprocessing steps to convert the raw labelled dataset (consisting, at a minimum, of
    context-question-answer triplets plus other optional metadata) to the tensor inputs into the BERT model for
    fine-tuning on question answering tasks, or for predictions.
    The class can be initialised using a ready-made input dataset, or via the from_dict_of_paragraphs classmethod
    using a SQuAD-like dictionary dataset.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, input_dataset: List[Dict]) -> None:
        """
        :param tokenizer: the tokenizer used to tokenize the text. Must be a class derived from PreTrainedTokenizerBase.
        :param input_dataset: a list where each element is a dictionary with at least 2 items, namely
               - 'context_text': a str with the full reference text in which the answer can be found,
               - 'question_text': a str with the question text,
               Other optional, frequently used keys, are:
               - 'answers': a list of dictionaries, where each dictionary contains two str keys, namely 'text' with
                 the answer text and 'answer_start' with the index of the first answer character. Some datasets will
                 have one answer per question, whereas others might have multiple valid answers. An answer is always
                 needed for training and testing, but not when using the model in production and the answer is unknown.
               - 'qas_id': optional, a hash-like str which is unique to each question-answer pair,
               - 'title': optional (for SQuAD only): a str with the title of the article where the context is taken
                 from. This is not used directly and is for reference purposes only.
        """
        expected_keys = ['context_text', 'question_text']
        assert all([key in dict_.keys() for key in expected_keys for dict_ in input_dataset]), \
            f"Each dictionary item in the input_dataset list must contain the following keys: {expected_keys}."
        self._tokenizer = tokenizer
        self._input_dataset = input_dataset

    def __len__(self):
        return len(self._input_dataset)

    def __getitem__(self, item):
        return self._input_dataset[item]

    @classmethod
    def from_dict_of_paragraphs(cls, tokenizer: PreTrainedTokenizerBase, input_dataset: Dict):
        """
        A classmethod to instantiate the class from a SQuAD-like dictionary dataset.
        :param tokenizer: the tokenizer used to tokenize the text. Must be a class derived from PreTrainedTokenizerBase.
        :param input_dataset: passed as argument of _create_training_samples_from_dict_of_paragraphs
        :return: an instance of DatasetEncoder ready for use.
        """
        assert 'data' in input_dataset.keys(), "SQuAD input dataset must have 'data' key."
        assert all([key in art.keys() for key in ['title', 'paragraphs'] for art in input_dataset['data']]), \
            "Input data don't match SQuAD structure. Keys 'title' and 'paragraphs' must be in each item in 'data' list."

        training_samples = cls(tokenizer, cls._create_training_samples_from_dict_of_paragraphs(input_dataset))
        return training_samples

    @staticmethod
    def _create_training_samples_from_dict_of_paragraphs(input_dict: Dict) -> List[Dict]:
        """
        This is called by the from_dict_of_paragraphs class method when instantiating the class with a SQuAD-like
        dataset, and it converts the dataset into a format which is more readily usable for our fine-tuning.
        :param input_dict: a dictionary with two keys: "data" and "version". The first value is a list where each
               element corresponds to a paragraph and all its related questions and answers.
        :return: training_samples: a list where each element is a dictionary with 5 or 6 items, including a question, a
                 context and the answer. The context is the reference text in which the answer can be found.
        """
        training_samples = []
        for article in input_dict['data']:
            for paragraph in article['paragraphs']:
                for qas in paragraph['qas']:  # each paragraph has multiple questions and answers associated
                    sample_dict = {
                        'answers': qas['answers'],
                        'context_text': paragraph['context'],
                        'qas_id': qas['id'],
                        'question_text': qas['question'],
                        'title': article['title']
                    }
                    training_samples.append(sample_dict)
        return training_samples

    def tokenize_and_encode(
            self,
            with_answers: bool,
            max_len: int,
            start_end_positions_as_tensors: bool = True,
            log_interval: Optional[int] = None,
            device_: Optional[str] = None  # if None, it automatically detects if a GPU is available, if not uses a CPU
    ) -> Tuple:
        """
        This method converts the input dataset into  a number of tensors ready to train the BERT model for question
        answering or be used for predictions. It takes as input the maximum length to pad the text to. Any sample
        where the answer falls outside (or partially outside) the question + context sentence pair after its truncation
        to max_len is dropped from the dataset. The remaining N samples are tokenized and encoded.
        If using a test dataset, multiple valid answers can be provided, in which case some tensors will be returned
        as lists instead (see details below). If with_answer=True, this will also return the tensors of start and end
        indices for each answer.

        :param with_answers: bool; whether an answer is provided for each question. This is usually True for training
               and testing datasets, and False for real life production text when the answer is unknown.
        :param max_len: an int; the maximum length to pad the question + answer sentence pair sequence to. Training
               time is quadratic with max_len, however if max_len is too low, more answers will fall outside the limit
               and will be truncated, making those samples unusable and therefore hurting accuracy due to the loss of
               information. GPU or CPU memory limits also need be taken into account when finding the best trade-off.
        :param start_end_positions_as_tensors: a boolean specifying whether start_positions and end_positions should be
               returned as tensors. Default: True. If False, they will be returned as lists. Please note: if True, only
               one valid answer per question must be provided (usually this is the case during training). If multiple
               valid answers are provided (usually during testing), set value to False.
        :param log_interval: the interval when to log the encoding status. Default: None
        :param device_: if specified, the device used for the computations. Can be one of cpu, cuda, mkldnn, opengl,
               opencl, ideep, hip, msnpu. If set to None, it will default to GPU (cuda) if one is available, else it
               will use a CPU. Default: None
        :return:
        - if with_answer=True: Tuple of 5 elements
            input_ids: torch.tensor of shape (N, max_len) representing the ids of each token of the N encoded
            sequence pairs, with padding at the end.
            token_type_ids: torch.tensor of shape (N, max_len) where each Nth dimension is filled with 1 for token
            positions in the context text, 0 elsewhere (i.e. in question and padding)
            attention_masks: torch.tensor of shape (N, max_len) where each Nth dimension is filled with 1 for
            non-"[PAD]" tokens, 0 for "[PAD]" tokens.
            start_positions: if start_end_positions_as_tensors=True, this is a torch.tensor of shape (N)
            containing the index of the first answer token for each answer. Otherwise, this is a list of lists,
            where each inner list contains m torch.tensors, where m is the number of possible correct answers.
            Note that m can vary for each inner list depending on how many valid answers each question has.
            Given this variability, it is not possible to convert the outer list to a tensor as the inner lists
            of tensors have variable lengths. Note these represent ground truth values, not predictions.
            end_positions: same as start_positions but for the last answer token for each answer.
            dropped_samples: int, the number of samples dropped from the dataset due to the answer (or at least
            one of the possible answers, if multiple valid answers are given) falling outside (or partially
            outside) the question + answer sentence pair after truncation to max_len. For N encoded sequence pairs,
            dropped_samples = len(training_samples) - N
        - if with_answer=False: Tuple of 3 elements
            input_ids: as above
            token_type_ids: as above
            attention_masks: as above
            It does NOT return start_positions and end_positions and ground truth answer values are not provided.
        """
        if with_answers:
            assert all(['answers' in dict_.keys() for dict_ in self._input_dataset]), \
                "Not all questions provided contain an answer. If you do not intend to use ground truth answer " \
                "values for training or testing, please set with_answers=False ."
            return self._tokenize_and_encode_with_answer(max_len, start_end_positions_as_tensors, log_interval, device_)
        else:
            if not start_end_positions_as_tensors:
                logger.warning("Setting start_end_positions_as_tensors=False has no effect when with_answers=False.")
            if log_interval is not None:
                logger.warning("Setting log_interval has no effect when with_answers=False")
            return self._tokenize_and_encode_without_answer(max_len, device_)

    def _tokenize_and_encode_with_answer(
            self,
            max_len: int,
            start_end_positions_as_tensors: bool = True,
            log_interval: Optional[int] = None,
            device_: Optional[str] = None  # if None, it automatically detects if a GPU is available, if not uses a CPU
    ) -> Tuple[Tensor, Tensor, Tensor, Union[List[List[Tensor]], Tensor], Union[List[List[Tensor]], Tensor], int]:
        """
        Called by tokenize_and_encode when with_answers=True
        """
        device = set_hardware_acceleration(default=device_)

        dropped_samples = 0
        all_encoded_dicts = []
        all_q_start_positions = []
        all_q_end_positions = []

        t_i = time()  # initial time
        for i, sample in enumerate(self._input_dataset):
            if start_end_positions_as_tensors and len(sample['answers']) != 1:
                raise IndexError(
                    f"In order to return torch tensors for training, each question must have only one possible "
                    f"answers. If tokenizing questions with multiple valid answers for testing, please set "
                    f"start_end_positions_as_tensors=False."
                )
            if log_interval is not None and i % log_interval == 0 and i != 0:
                logger.info(
                    f"Encoding sample {i} of {len(self._input_dataset)}. Elapsed: {format_time(time() - t_i)}. "
                    f"Remaining: {format_time((time() - t_i) / i * (len(self._input_dataset) - i))}."
                )
            possible_starts = []
            possible_ends = []
            # in dev sets with more than one possible answer, it records if some but not all valid answers are truncated
            for possible_answer in sample['answers']:
                answer_tokens = self._tokenizer.tokenize(possible_answer['text'])
                answer_replacement = " ".join(["[MASK]"] * len(answer_tokens))
                start_position_character = possible_answer['answer_start']
                end_position_character = possible_answer['answer_start'] + len(possible_answer['text'])
                context_with_replacement = sample['context_text'][:start_position_character] + answer_replacement + \
                    sample['context_text'][end_position_character:]
                encoded_dict = self._tokenizer.encode_plus(
                    sample['question_text'],
                    context_with_replacement,
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]' tokens
                    max_length=max_len,
                    padding='max_length',  # Pad or truncates sentences to `max_length`
                    truncation=True,
                    return_attention_mask=True,  # Construct attention masks.
                    return_tensors='pt',  # Return pytorch tensors.
                ).to(device)
                '''A dictionary containing the sequence pair and additional information. There are 3 keys, each value 
                is a torch.tensor of shape (1, max_len) and can be converted to just (max_len) by applying .squeeze():
                - 'input_ids': the ids of each token of the encoded sequence pair, with padding at the end
                - 'token_type_ids': 1 for token positions in  answer text, 0 elsewhere (i.e. in question and padding)
                - 'attention_mask': 1 for non "[PAD]" token, 0 for "[PAD]" tokens.'''
                is_mask_token = encoded_dict['input_ids'].squeeze() == self._tokenizer.mask_token_id
                mask_token_indices = is_mask_token.nonzero(as_tuple=False)
                if len(mask_token_indices) != len(answer_tokens):
                    continue  # ignore cases where start or end of answer exceed max_len and have been truncated
                answer_start_index, answer_end_index = mask_token_indices[0], mask_token_indices[-1]
                possible_starts.append(answer_start_index)
                possible_ends.append(answer_end_index)
                answer_token_ids = self._tokenizer.encode(
                    possible_answer['text'],
                    add_special_tokens=False,
                    return_tensors='pt'
                ).to(device)
            if len(sample['answers']) != len(possible_starts) or len(sample['answers']) != len(possible_ends):
                dropped_samples += 1  # we drop sample due to answer being truncated
                continue
            encoded_dict['input_ids'][0, answer_start_index:answer_end_index + 1] = answer_token_ids
            # Finally, replace the "[MASK]" tokens with the actual answer tokens
            all_encoded_dicts.append(encoded_dict)
            all_q_start_positions.append(possible_starts)
            all_q_end_positions.append(possible_ends)

        assert len(all_encoded_dicts) == len(self._input_dataset) - dropped_samples, "Lengths check failed!"
        input_ids = torch.cat([encoded_dict['input_ids'] for encoded_dict in all_encoded_dicts], dim=0)
        token_type_ids = torch.cat([encoded_dict['token_type_ids'] for encoded_dict in all_encoded_dicts], dim=0)
        attention_masks = torch.cat([encoded_dict['attention_mask'] for encoded_dict in all_encoded_dicts], dim=0)
        if start_end_positions_as_tensors:
            all_q_start_positions = torch.tensor(all_q_start_positions).squeeze().to(device)
            all_q_end_positions = torch.tensor(all_q_end_positions).squeeze().to(device)
        if dropped_samples > 0:
            logger.warning(
                f"Dropped {dropped_samples} question+context pair samples from the dataset because the start or end "
                f"token of the answer was at an unreachable position exceeding the max_len ({max_len})."
            )
        return input_ids, token_type_ids, attention_masks, all_q_start_positions, all_q_end_positions, dropped_samples

    def _tokenize_and_encode_without_answer(
            self,
            max_len: int,
            device_: Optional[str] = None  # if None, it automatically detects if a GPU is available, if not uses a CPU
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Called by tokenize_and_encode when with_answers=False.
        """
        device = set_hardware_acceleration(default=device_)

        all_encoded_dicts = []
        for i, sample in enumerate(self._input_dataset):
            encoded_dict = self._tokenizer.encode_plus(
                sample['question_text'],
                sample['context_text'],
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]' tokens
                max_length=max_len,
                padding='max_length',  # Pad or truncates sentences to `max_length`
                truncation=True,
                return_attention_mask=True,  # Construct attention masks.
                return_tensors='pt',  # Return pytorch tensors.
            ).to(device)
            all_encoded_dicts.append(encoded_dict)
        input_ids = torch.cat([encoded_dict['input_ids'] for encoded_dict in all_encoded_dicts], dim=0)
        token_type_ids = torch.cat([encoded_dict['token_type_ids'] for encoded_dict in all_encoded_dicts], dim=0)
        attention_masks = torch.cat([encoded_dict['attention_mask'] for encoded_dict in all_encoded_dicts], dim=0)
        return input_ids, token_type_ids, attention_masks
