import torch
import numpy as np
from typing import List, Union, Tuple


def exact_match_rate(
        real_start: Union[List[List[torch.Tensor]], torch.Tensor],
        real_end: Union[List[List[torch.Tensor]], torch.Tensor],
        pred_start: torch.Tensor,
        pred_end: torch.Tensor
) -> Tuple[int, int, float]:
    """
    Takes as input the real and predicted start and end tokens of N answers and returns the number of correct
    predictions and the total match rate. An answer where only either the start or end token has been predicted
    correctly will count as 50% match rate for that answer. While only one predicted start and end can be given for
    an answer, multiple start-end pairs can be given for the ground truth values, as multiple correct answers may be
    acceptable. If multiple possible ground truth answers are provided, a prediction is considered correct if its
    values match exactly at least one of the ground truth values, and wrong otherwise.
    :param real_start: the ground truth value of the N answers' start token indices. If only one ground truth value per
           answer is given, this can be a torch.tensor if shape (N). If multiple valid answers are provided,  this
           must be a list of N lists, where each inner list contains m torch.tensors, where m is the number of
           possible correct answers. Note that m can vary for each inner list depending on how many valid answers each
           question has. Given this variability, it is not possible to convert the outer list to a tensor as the
           inner lists of tensors have variable lengths.
    :param real_end: the ground truth value of the N answers' end token indices. If only one ground truth value per
           answer is given, this can be a torch.tensor if shape (N). If multiple valid answers are provided,  this
           must be a list of N lists, where each inner list contains m torch.tensors, where m is the number of
           possible correct answers. Note that m can vary for each inner list depending on how many valid answers each
           question has. Given this variability, it is not possible to convert the outer list to a tensor as the
           inner lists of tensors have variable lengths.
    :param pred_start: the predicted values of the N answers' start token indices. Must be a torch.tensor of shape (N),
           with only one predicted token per answer.
    :param pred_end: the predicted values of the N answers' end token indices. Must be a torch.tensor of shape (N),
           with only one predicted token per answer.
    :return: correct: the total number of correct predictions out of the total number of predictions which is 2*N (i.e.
             N starts and N ends).
             total_indices: The total number of predictions, i.e. 2*N (N starts and N ends)
             match_rate: the exact match rate defined as the ratio correct/total_indices
    """
    assert len(real_start) == len(real_end), "real_start and real_end shapes do not match."
    assert pred_start.shape == pred_end.shape, "pred_start and pred_end lengths do not match."
    assert len(real_start) == len(pred_start), \
        f"Datasets mismatch: {len(real_start)} correct labels and {len(pred_start)} predictions were provided."

    correct = 0
    total_indices = len(pred_start) + len(pred_end)
    for i, (pred_start_sample, pred_end_sample) in enumerate(zip(pred_start, pred_end)):
        '''The list below list will store how many correct predictions (start+end) the algorithm makes for every
        correct possible answer. E.g. if there are 3 possible correct answers, and the algorithm predicts start+end 
        correctly for the first answer, only correct start of the second possible answer, and not the third 
        possible answer the list will be [2, 1, 0]. We'll take take the max (in this case 2), as any correct possible
        answer means our model made a correct prediction.'''
        match_options = []
        # each sample might have j correct possible answers
        for real_start_sample, real_end_sample in zip(real_start[i], real_end[i]):
            matches = 0
            if pred_start_sample == real_start_sample:
                matches += 1
            if pred_end_sample == real_end_sample:
                matches += 1
            match_options.append(matches)
        correct += max(match_options)
    match_rate = correct / total_indices
    return correct, total_indices, match_rate


def f1_score(
        real_start: Union[List[List[torch.Tensor]], torch.Tensor],
        real_end: Union[List[List[torch.Tensor]], torch.Tensor],
        pred_start: torch.Tensor,
        pred_end: torch.Tensor
) -> Tuple[List[float], float]:
    """
    Takes as input the real and predicted start and end tokens of N answers and returns the F1 scores (average and
    per individual answer). While only one predicted start and end can be given for an answer, multiple start-end
    pairs can be given for the ground truth values, as multiple correct answers may be acceptable. If multiple
    possible ground truth answers are provided, the F1 score is calculated for each answer against all provided
    ground truth values, and the highest of those F1 scores is taken as the final F1 score for that answer.
    :param real_start: the ground truth value of the N answers' start token indices. If only one ground truth value per
           answer is given, this can be a torch.tensor if shape (N). If multiple valid answers are provided,  this
           must be a list of N lists, where each inner list contains m torch.tensors, where m is the number of
           possible correct answers. Note that m can vary for each inner list depending on how many valid answers each
           question has. Given this variability, it is not possible to convert the outer list to a tensor as the
           inner lists of tensors have variable lengths.
    :param real_end: the ground truth value of the N answers' end token indices. If only one ground truth value per
           answer is given, this can be a torch.tensor if shape (N). If multiple valid answers are provided,  this
           must be a list of N lists, where each inner list contains m torch.tensors, where m is the number of
           possible correct answers. Note that m can vary for each inner list depending on how many valid answers each
           question has. Given this variability, it is not possible to convert the outer list to a tensor as the
           inner lists of tensors have variable lengths.
    :param pred_start: the predicted values of the N answers' start token indices. Must be a torch.tensor of shape (N),
           with only one predicted token per answer.
    :param pred_end: the predicted values of the N answers' end token indices. Must be a torch.tensor of shape (N),
           with only one predicted token per answer.
    :return: all_f1: a list containing all the F1 scores for each predicted answer.
             average_f1: the average F1 score on the whole input dataset of N answers.
    """
    all_f1 = []
    for i, (pred_start_sample, pred_end_sample) in enumerate(zip(pred_start, pred_end)):
        '''The list below list will store how many correct predictions (start+end) the algorithm makes for every
        correct possible answer. E.g. if there are 3 possible correct answers, and the algorithm predicts start+end 
        correctly for the first answer, only correct start of the second possible answer, and not the third 
        possible answer the list will be [2, 1, 0]. We'll take take the max (in this case 2), as any correct possible
        answer means our model made a correct prediction.'''

        pred_indices = set(range(pred_start_sample, pred_end_sample + 1))

        f1_options = []
        # each sample might have j correct possible answers
        for real_start_sample, real_end_sample in zip(real_start[i], real_end[i]):

            real_indices = set(range(real_start_sample, real_end_sample + 1))  # consider adding int() around tensors
            correctly_pred_indices = real_indices.intersection(pred_indices)
            if correctly_pred_indices == set():
                f1_options.append(0)
                continue  # f1 is 0 if there's no overlap. Loop cannot continue to avoid division by zero error.

            precision = len(correctly_pred_indices) / len(pred_indices)
            recall = len(correctly_pred_indices) / len(real_indices)
            f1_sample = (2 * precision * recall) / (precision + recall)
            f1_options.append(f1_sample)
        all_f1.append(float(max(f1_options)))  # float() just so we avoid a mix of int and float in the list

    average_f1 = np.mean(all_f1)
    return all_f1, average_f1


