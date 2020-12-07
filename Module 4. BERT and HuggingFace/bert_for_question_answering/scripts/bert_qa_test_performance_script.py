import json
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from modules_solutions.preprocess_dataset import DatasetEncoder
from modules_solutions.prediction_loop import predict
from modules_solutions.scores import exact_match_rate, f1_score

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
    # NOTE: dev-v1.1-small-subset.json should only be used to test the code works. When doing the actual scoring,
    # the full dev-v1.1.json should be used instead.
    with open("data/dev-v1.1-small-subset.json", "r") as f:
        dev = json.load(f)
    tok_enc = DatasetEncoder.from_dict_of_paragraphs(tokenizer, dev)
    input_ids, token_type_ids, attention_masks, start_positions, end_positions, dropped_samples = \
        tok_enc.tokenize_and_encode(
            max_len=384, log_interval=1000, start_end_positions_as_tensors=False, with_answers=True
        )
    for i in [input_ids, token_type_ids, attention_masks, start_positions, end_positions]:
        try:
            print(i.shape)
        except AttributeError:
            print(len(i))
    print(dropped_samples, " samples dropped.")
    model = BertForQuestionAnswering.from_pretrained("bert-base-cased")
    print(model.load_state_dict(torch.load("models/trained_model_state_dict.pt", map_location=torch.device('cpu'))))
    # This is an inplace function, but we're printing it because if all goes well it should output
    # '<All keys matched successfully>'
    pred_start, pred_end = predict(input_ids, token_type_ids, attention_masks, model, batch_size=16)
    correct, total_indices, match_rate = exact_match_rate(start_positions, end_positions, pred_start, pred_end)
    print(correct, total_indices, match_rate)
    all_f1, average_f1 = f1_score(start_positions, end_positions, pred_start, pred_end)
    print(average_f1)
