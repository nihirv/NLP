import json
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from modules_solutions.preprocess_dataset import DatasetEncoder
from modules_solutions.fine_tuning import fine_tune_train_and_eval

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
    # NOTE: train-v1.1-small-subset.json should only be used to test the code works. When doing the actual training,
    # the full train-v1.1.json should be used instead.
    with open("data/train-v1.1-small-subset.json", "r") as f:
        train = json.load(f)
    tok_enc = DatasetEncoder.from_dict_of_paragraphs(tokenizer, train)
    input_ids, token_type_ids, attention_masks, start_positions, end_positions, dropped_samples = \
        tok_enc.tokenize_and_encode(
            max_len=384, start_end_positions_as_tensors=True, log_interval=1000, with_answers=True
        )
    for i in [input_ids, token_type_ids, attention_masks, start_positions, end_positions]:
        try:
            print(i.shape)
        except AttributeError:
            print(len(i))
    print(dropped_samples, " samples dropped.")

    model = BertForQuestionAnswering.from_pretrained(
        "bert-base-cased",  # Use the 12-layer BERT model with pre-trained weights, with a cased vocab.
        output_attentions=False,
        output_hidden_states=False,
    )
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  # defaults: lr=5e-5, eps=1e-8

    model, training_stats = fine_tune_train_and_eval(
        input_ids,
        token_type_ids,
        attention_masks,
        start_positions,
        end_positions,
        batch_size=(16, 16),
        model=model,
        optimizer=optimizer,
        train_ratio=0.9,
        training_epochs=3,
        lr_scheduler_warmup_steps=0,
        save_model_path="models/trained_model.pt",
        save_stats_dict_path="models/statistics.json"
    )
