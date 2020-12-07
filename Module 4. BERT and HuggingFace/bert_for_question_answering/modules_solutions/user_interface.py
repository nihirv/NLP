import torch.nn as nn
from typing import Union, List
from transformers import PreTrainedTokenizerBase
from .prediction_loop import predict
from .preprocess_dataset import DatasetEncoder


class ChatBot:
    """
    This class is designed as a user interface, where a context, question/s, a pre-trained model and a pre-trained
    tokenizer are given, and the unknown answer/s to the question/s is/are returned. This is for use in production
    where the answer/s is/are not known, and this class does not support handling of ground truths answers.
    """
    def __init__(self, context: str, tokenizer: PreTrainedTokenizerBase, model: nn.Module, max_len: int = 500) -> None:
        """

        :param context: str; the context, i.e. the reference text in which the answer/s can be found.
        :param tokenizer: the tokenizer used to tokenize the text. Must be a class derived from PreTrainedTokenizerBase.
        :param model: the model to use (must be instance of torch.nn.Module). This must be a pre-trained model.
        :param max_len: an int; the maximum length to pad the question + context sentence pair sequence to.
               Computational time is quadratic with max_len, however if max_len is too low, more answers will fall
               outside the limit and will be truncated, making the predicted answers incorrect. For production mode
               predictions, it is recommended to keep max_len long enough to contain the full question + context
               sentence pair. Default: 500.
        """
        self.context = context
        self.tokenizer = tokenizer
        self.model = model
        self.max_len = max_len

    def answer(self, questions: Union[str, List[str]], disable_progress_bar: bool = True) -> List[str]:
        """
        Takes a question, or list of questions, whose answer/s can be found in the context used to instantiate the
        class, and returns the answer or list of answers.
        :param questions: a string with a single question, or list of strings with multiple questions. All questions
               must be answerable with the given context.
        :param disable_progress_bar: bool; whether to disable the tqdm bar showing computational progress as the bot
               calculates the answers. The bar may be useful to track progress if many questions are asked.
               Default: True.
        :return: predicted_answers: A list of strings with all the predicted answers in order.
        """
        if isinstance(questions, str):
            questions = [questions]  # convert to list if a single question is given as string
        encoder = DatasetEncoder(
            tokenizer=self.tokenizer,
            input_dataset=[{'context_text': self.context, 'question_text': question} for question in questions]
        )
        input_ids, token_type_ids, attention_masks = encoder.tokenize_and_encode(
            max_len=self.max_len, with_answers=False
        )
        pred_start, pred_end = predict(
            input_ids, token_type_ids, attention_masks, self.model, 1, disable_progress_bar=disable_progress_bar
        )
        predicted_answers = [
            self.tokenizer.decode(input_ids[i, pred_start_i:pred_end_i + 1])
            for i, (pred_start_i, pred_end_i) in enumerate(zip(pred_start, pred_end))
        ]
        return predicted_answers



