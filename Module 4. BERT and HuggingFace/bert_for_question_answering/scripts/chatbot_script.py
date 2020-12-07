import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from modules_solutions.user_interface import ChatBot


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
    model = BertForQuestionAnswering.from_pretrained("bert-base-cased")
    print(model.load_state_dict(torch.load("models/trained_model_state_dict.pt", map_location=torch.device('cpu'))))
    # This is an inplace function, but we're printing it because if all goes well it should output
    # '<All keys matched successfully>'
    context = """
    Boris Johnson is a British politician, author, and former  journalist who has served as Prime Minister of the United 
    Kingdom and Leader of the Conservative Party since 2019. 
    He was Foreign Secretary from 2016 to 2018 and Mayor of London from 2008 to 2016. Johnson was Member of 
    Parliament (MP) for Henley from 2001 to 2008 and has been MP for Uxbridge and South Ruislip since 2015. 
    Ideologically, he identifies as a one-nation conservative.
    """.replace("\n", " ")

    bot = ChatBot(context, tokenizer, model, max_len=300)
    questions = [
        "Who is Boris Johnson?",
        "From when was Boris Johnson Foreign Secretary?",
        "Which party does Boris Johnson belong to?",
    ]

    answers = bot.answer(questions, disable_progress_bar=False)
    for answer in answers:
        print("Answer is: ", answer)
