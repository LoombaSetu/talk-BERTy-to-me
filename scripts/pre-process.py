import torch
import nltk
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from load_data import TextCorpus
nltk.download("punkt")
nltk.download("stopwords")

def create_seq2seq_labels(text, window_size=8):
    if text is None:
        return []
    tokens = tokenizer(text)
    sequence_pairs = []
    for idx in range(len(tokens) - window_size):
        sequence_pairs.append(
            (tokens[idx : idx + window_size], tokens[idx + 1 : idx + window_size + 1])
        )
    return sequence_pairs

def collate_batch(batch):
    titles, genres, seq_pair = zip(*batch)
    titles = [tokenizer(t) for t in titles]
    title_tensor = pad_sequence([torch.tensor(vocab.lookup_indices(t)) for t in titles],
                                 batch_first=True)
    genres = [tokenizer(str(t)) for t in genres]
    genre_tensor = pad_sequence([torch.tensor(vocab.lookup_indices(t)) for t in genres],
                                batch_first=True)
    input_seq, output_seq = zip(*seq_pair)
    input_tensor = pad_sequence(
        [torch.tensor(vocab.lookup_indices(t)) for t in input_seq], batch_first=True
    )
    output_tensor = pad_sequence(
        [torch.tensor(vocab.lookup_indices(t)) for t in output_seq], batch_first=True
    )
    return title_tensor, input_tensor, output_tensor, genre_tensor


corpus = TextCorpus("dev", True)
tokenizer = get_tokenizer("basic_english")
train_df = corpus.data
train_df.loc[:, "seq_pairs"] = train_df["text"].apply(create_seq2seq_labels)
train_df.drop(columns=["text"], inplace=True)
train_df = train_df.explode("seq_pairs").reset_index(drop=True)
print(train_df.columns)
print(train_df.loc[:, "seq_pairs"][:1])
vocab = corpus.vocab

dataloader = torch.utils.data.DataLoader(
    train_df.values, batch_size=8, shuffle=False, collate_fn=collate_batch
)
for i, (title, input_seq, output_seq, genre) in enumerate(dataloader):
    print(title)
    print(input_seq)
    print(output_seq)
    print(genre)
    if i == 0:
        break
