# import essential libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import math
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, layers, PAD_IDX=1, bidirectional=False, dropout=0.1):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.bidirectional = bidirectional
        self.PAD_IDX = PAD_IDX

        # Initialise dropout
        self.dropout = nn.Dropout(dropout)

        # If we use a bidirectional encoder to encode both forward and backward context,
        # the dimension of the hidden state will double
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            ff_input_dim = 2 * hidden_dim
        else:
            ff_input_dim = hidden_dim

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(emb_dim, self.hidden_dim, layers, dropout=dropout,
                           bidirectional=bidirectional, bias=False, batch_first=True)

        # Our decoder will take something of shape [B, H].
        # If we use bidirectional RNNs, we will be returning something of [B, 2H].
        # So we apply a non-linearity to reduce our encoder dimensions to [B, H]
        # Apply a linear layer followed by a tanh activation
        self.ff = nn.Sequential(
            nn.Linear(ff_input_dim, hidden_dim),
            nn.Tanh()
        )

    # x: (T, B)
    def forward(self, x):
        # x: (B, T)
        x = x.permute(1, 0)

        # embed the input, and then apply dropout
        x = self.dropout(self.embedding(x))
        # x: (B, T, E)

        outputs, (h_n, c_n) = self.rnn(x)
        # outputs: (B, T, H*directions)
        # h_n: (layers*directions, B, H)

        if self.bidirectional:
            # concatenate the forward and backward hidden states
            h_n = torch.cat((h_n[0::2, :, :], h_n[1::2, :, :]), dim=-1)
            c_n = torch.cat((c_n[0::2, :, :], c_n[1::2, :, :]), dim=-1)

        # reduce dimensionality of our final hidden and cell state
        h_n = self.ff(h_n)
        c_n = self.ff(c_n)
        # h_n: (layers, B, H)
        # c_n: (layers, B, H)

        # outputs: ()
        return outputs, (h_n, c_n)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, layers, PAD_IDX=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.layers = layers

        # initialize dropout
        self.dropout = nn.Dropout(dropout)

        # initialize embedding
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)

        # initialize LSTM (set batch_first=True)
        # we DON'T set bidirectional=True here
        self.rnn = nn.LSTM(emb_dim, hidden_dim, layers, dropout=dropout,
                           batch_first=True)  # we don't set bidirectional here

        self.out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    # x: (B)

    def forward(self, x, hidden):
        # we expand the dim of sequence length
        x = x.unsqueeze(1)
        # x: (B, 1)

        # apply the embedding and dropout to x
        embed = self.dropout(self.embedding(x))
        # embed: (B, 1, E)

        # run the LSTM
        # We initialize the hidden & cell state of the decoder to the hidden & cell state of the encoder.
        # Read https://pytorch.org/docs/stable/nn.html#lstm to see how this is done
        output, hidden = self.rnn(embed, hidden)
        # output: (B, 1, H), h_n: (layers, B, H)

        # Run the output layer
        output = self.out(output)
        # output = (B, 1, O)

        # Remove the dimension where there is "1"
        output = output.squeeze(1)
        # output = (B, O)

        return output, hidden


class Seq2seq(nn.Module):

    def __init__(self, encoder, decoder, device='cpu', with_attn=False):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.with_attn = with_attn

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        output_dim = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, output_dim).to(self.device)

        enc_outputs, hidden = self.encoder(src)

        # initialize output sequence with '<sos>'
        dec_output = trg[0, :]

        # decoder token by token
        for t in range(1, max_len):
            if self.with_attn:
                dec_output, hidden, _ = self.decoder(dec_output, hidden, enc_outputs)
            else:
                dec_output, hidden = self.decoder(dec_output, hidden)

            outputs[t] = dec_output
            teacher_force = random.random() < teacher_forcing_ratio

            pred_next = dec_output.argmax(1)

            dec_output = (trg[t] if teacher_force else pred_next)
        return outputs

    # greedy search for actual translation
    def greedy_search(self, src, sos_idx, max_len=50, return_attention=False):
        src = src.to(self.device)
        batch_size = src.shape[1]
        src_len = src.shape[0]

        outputs = torch.zeros(max_len, batch_size).to(self.device)

        enc_outputs, hidden = self.encoder(src)

        dec_output = torch.zeros(batch_size, dtype=torch.int64).to(device)
        dec_output.fill_(sos_idx)

        outputs[0] = dec_output

        attentions = torch.zeros(max_len, batch_size, src_len).to(self.device)

        for t in range(1, max_len):
            if self.with_attn:
                dec_output, hidden, attention_score = self.decoder(dec_output, hidden, enc_outputs)
                attentions[t] = attention_score
            else:
                dec_output, hidden = self.decoder(dec_output, hidden)

            dec_output = dec_output.argmax(1)

            outputs[t] = dec_output

        if return_attention:
            return outputs, attentions
        else:
            return outputs


# torchtext will pre-process the data, including tokenization, padding, stoi, etc.
SRC = Field(tokenize="spacy",
            tokenizer_language="de",
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

TRG = Field(tokenize="spacy",
            tokenizer_language="en",
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                    fields=(SRC, TRG))
# print the number of examples in train/valid/test sets
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")


# build a vocab of our training set, ignoring word with frequency less than 2
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)


# build train/valid/test iterators, which will batch the data for us
BATCH_SIZE = 128
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

x = vars(test_data.examples[0])['src']
y = vars(test_data.examples[0])['trg']
print("Source example:", " ".join(x))
print("Target example:", " ".join(y))
print("Padded target:", TRG.pad([y]))
print("Tensorized target:", TRG.process([y]))


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
EMB_DIM = 256
HIDDEN_DIM = 512
LAYERS = 1
DROPOUT = 0.5
BIDIRECTIONAL = True


# padding token
SRC_PAD = SRC.vocab.stoi['<pad>']
TRG_PAD = TRG.vocab.stoi['<pad>']

# build model
enc = Encoder(INPUT_DIM, EMB_DIM, HIDDEN_DIM, LAYERS, PAD_IDX=SRC_PAD, bidirectional=BIDIRECTIONAL, dropout=DROPOUT)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HIDDEN_DIM, LAYERS, PAD_IDX=TRG_PAD, dropout=DROPOUT)
model = Seq2seq(enc, dec, device).to(device)


# initialize weights
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.1, 0.1)


model.apply(init_weights)
print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')


LR = 0.001
# set optimizer and learning rate
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD)


def train(model, iterator, optimizer, criterion, grad_clip, num_epoch):
    model.train()

    total_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        outputs = model(src, trg)

        # exclude <sos> token
        # outputs: (seq_len * batch_size, output_dim)
        # trg : (seq_len * batch_size)
        outputs = outputs[1:].view(-1, outputs.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(outputs, trg)

        writer.add_scalar('training loss',
                          loss.item(),
                          num_epoch * len(iterator) + i)

        if i % 50 == 0:
            print('Batch:\t {0} / {1},\t loss: {2:2.3f}'.format(i, len(iterator), loss.item()))

        loss.backward()
        # clip grad to avoid gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(iterator)


def eval(model, iterator, criterion):
    # In eval model, layers such as Dropout, BatchNorm will work in eval model
    model.eval()

    total_loss = 0
    # this prevents the back-propagation
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            # during test time, we have no correct trg so we turn off teacher forcing
            outputs = model(src, trg, teacher_forcing_ratio=0)

            outputs = outputs[1:].view(-1, outputs.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(outputs, trg)
            total_loss += loss.item()
    return total_loss / len(iterator)


# Helper function, converting a batch of tensors to the text form
def get_text_from_tensor(tensor, field, eos='<eos>'):
    batch_output = []
    for i in range(tensor.shape[1]):
        sequence = tensor[:, i]
        words = []
        for tok_idx in sequence:
            tok_idx = int(tok_idx)
            token = field.vocab.itos[tok_idx]

            if token == '<sos>':
                continue
            elif token == '<eos>' or token == '<pad>':
                break
            else:
                words.append(token)
        words = " ".join(words)
        batch_output.append(words)
    return batch_output


import sacrebleu


def test_bleu(model, iterator, trg_field, with_attention=False):
    model.eval()

    ref = []
    hyp = []

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            outputs = model.greedy_search(src, trg_field.vocab.stoi['<sos>'], return_attention=with_attention)

            hyp += get_text_from_tensor(outputs, trg_field)
            ref += get_text_from_tensor(trg, trg_field)

    # expand dim of reference list
    # sys = ['translation_1', 'translation_2']
    # ref = [['truth_1', 'truth_2'], ['another truth_1', 'another truth_2']]
    ref = [ref]
    return sacrebleu.corpus_bleu(hyp, ref, force=True)


EPOCH = 30
CLIP = 1

best_bleu = 0

for i in range(EPOCH):
    print('Start training Epoch {}:'.format(i + 1))
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP, i)
    valid_loss = eval(model, valid_iterator, criterion)
    bleu = test_bleu(model, valid_iterator, TRG)

    writer.add_scalar('valid loss',
                      valid_loss,
                      i)
    writer.add_scalar('valid ppl',
                      math.exp(valid_loss),
                      i)
    writer.add_scalar('valid BLEU',
                      bleu.score,
                      i)

    if bleu.score > best_bleu:
        best_bleu = bleu.score
        torch.save(model.state_dict(), 'checkpoint_best-seq2seq.pt')

    print('Epoch {0} train loss: {1:.3f} | Train PPL: {2:7.3f}'.format(i + 1, train_loss, math.exp(train_loss)))
    print('Epoch {0} valid loss: {1:.3f} | Valid PPL: {2:7.3f}'.format(i + 1, valid_loss, math.exp(valid_loss)))
    print('Epoch {0} valid BLEU: {1:3.3f}'.format(i + 1, bleu.score))


model.load_state_dict(torch.load('checkpoint_best-seq2seq.pt', map_location=torch.device(device)))
print(test_bleu(model, test_iterator, TRG))
