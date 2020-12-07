WARNING

------------------------------------

The code in the current folder `bert_for_question_answering` works with `transformers==3.3.1`. All other libraries 
needed for this project are listed in 
the `requirements.txt` file with their required versions. As of 30 November 2020, the version of `transformers==4.0.0` 
was released. This version 
is not backward compatible, and this code is known to break if this version is used. Please be aware of this, and 
run this code on the version `3.3.1`.

In particular, this forward pass is known to break the code (from `modules/fine_tuning.py`):
```python
loss, start_logits, end_logits = model(
    input_ids=batch_input_ids,
    attention_mask=batch_attention_masks,
    token_type_ids=batch_token_type_ids,
    start_positions=batch_start_positions,
    end_positions=batch_end_positions
)
```
I have not yet investigated the fix for how to make this compatible for the 4.0.0 version. PRs are welcomed.



