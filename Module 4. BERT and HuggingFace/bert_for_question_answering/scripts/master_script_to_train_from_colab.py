"""
This is a sample script which could be run as is on Google Colab and will do all the training and testing.
In order for this to work, you will need to mount your Google Drive first by doing

from google.colab import drive
drive.mount('/content/drive')

and inserting your verification code when prompted. You will need to log in if you haven't already.
You will also need to save all the modules_solutions files on your Drive and save the SQuAD jsons on a
"/content/drive/My Drive/Colab Notebooks/" folder on your Drive. Pleas ensure directories are set up correctly before
running. You will also need to manually set your Notebook to using a GPU accelerator, otherwise a CPU will be used.
"""

import json
import torch
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from modules_solutions.preprocess_dataset import DatasetEncoder
from modules_solutions.fine_tuning import fine_tune_train_and_eval
from modules_solutions.prediction_loop import predict
from modules_solutions.scores import exact_match_rate, f1_score

tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
with open("/content/drive/My Drive/Colab Notebooks/train-v1.1.json", "r") as f:
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
    save_model_path="/content/drive/My Drive/Colab Notebooks/trained_model.pt",
    save_stats_dict_path="/content/drive/My Drive/Colab Notebooks/statistics.json"
)


with open("/content/drive/My Drive/Colab Notebooks/dev-v1.1.json", "r") as f:
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
model = torch.load("/content/drive/My Drive/Colab Notebooks/trained_model.pt", map_location=torch.device('cpu'))
pred_start, pred_end = predict(input_ids, token_type_ids, attention_masks, model, batch_size=16)
correct, total_indices, match_rate = exact_match_rate(start_positions, end_positions, pred_start, pred_end)
all_f1, average_f1 = f1_score(start_positions, end_positions, pred_start, pred_end)

scores = {
    "correct": correct,
    "total_indices": total_indices,
    "match_rate": match_rate,
    "all_f1": all_f1,
    "average_f1": average_f1
}

with open("/content/drive/My Drive/Colab Notebooks/scores.json", "w") as file:
    json.dump(scores, file)
