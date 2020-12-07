if [ -d "Chapter 5. NLP/Module 4. BERT and HuggingFace/bert_for_question_answering" ]
then
  cd Chapter\ 5.\ NLP/Module\ 4.\ BERT\ and\ HuggingFace/bert_for_question_answering/
  python -m scripts.bert_qa_preprocess_and_finetune_script
elif [ -d "../scripts" ]
then
  cd ..
  python -m scripts.bert_qa_preprocess_and_finetune_script
elif [ -d "scripts" ]
then
  python -m scripts.bert_qa_preprocess_and_finetune_script
else
  echo "Could not find the scripts directory. Please ensure you're in the home directory of the Practical-ML-DS repository or at Chapter 5. NLP/Module 4. BERT and HuggingFace/bert_for_question_answering"
fi

