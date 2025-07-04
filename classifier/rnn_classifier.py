import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from common import getDataFrame, strip_outer_quotes

BATCH_SIZE = 64 # 32
TRAIN_EPOCHS = 15
LEARNING_RATE = 0.002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class RNN(nn.Module):
  def __init__(self, n_vocab, embed_dim, n_hidden, n_rnnlayers, n_classes):
    super(RNN, self).__init__()
    self.V = n_vocab
    self.D = embed_dim
    self.M = n_hidden
    self.K = n_classes
    self.L = n_rnnlayers

    self.embed = nn.Embedding(self.V, self.D)
    self.rnn = nn.LSTM(
        input_size=self.D,
        hidden_size=self.M,
        num_layers=self.L,
        batch_first=True)
    self.fc = nn.Linear(self.M, self.K)

  def forward(self, X):
    # Initial hidden states
    h0 = torch.zeros(self.L, X.size(0), self.M).to(device)
    c0 = torch.zeros(self.L, X.size(0), self.M).to(device)

    # embedding layer
    # turns word indices into word vectors
    out = self.embed(X)

    # get RNN unit output
    out, _ = self.rnn(out, (h0, c0))

    # max pool
    out, _ = torch.max(out, 1)

    # we only want h(T) at the final step
    # fully connected layer
    out = self.fc(out)
    return out

def batch_gd(model, criterion, optimizer, epochs, train_gen, test_gen):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        t0 = datetime.now()
        train_loss = []
        model.train()  # Set model to training mode
        for inputs, targets in train_gen():
            targets = targets.view(-1, 1).float()
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)

        test_loss = []
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for inputs, targets in test_gen():
                targets = targets.view(-1, 1).float()
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        train_losses[it], test_losses[it] = train_loss, test_loss
        dt = datetime.now() - t0
        print(f"Epoch {it+1}/{epochs}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Duration: {dt}")

    return train_losses, test_losses

def tokenize_test_data(df_test, word2idx):
    # convert data into word indices
    test_sents = []
    for i, row in df_test.iterrows():
        tokens = row['clean_text'].lower().split()
        sentences_as_ints = [word2idx[token] for token in tokens if token in word2idx]
        test_sents.append(sentences_as_ints)
    return test_sents

def tokenize_train_data(df_train):
    idx = 1
    word2idx= {'<PAD>':0}
    for i, row in df_train.iterrows():
        tokens = row['clean_text'].lower().split()  # simple tokenization
        for token in tokens:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1
    return word2idx

def convert_to_indices(df_train, word2idx):
    # convert data into word indices
    train_sents = []
    for i, row in df_train.iterrows():
        tokens = row['clean_text'].lower().split()
        sentences_as_ints = [word2idx[token] for token in tokens]
        train_sents.append(sentences_as_ints)
    return train_sents

def data_generator(X, y, batch_size=BATCH_SIZE):
  X, y = shuffle(X, y)
  n_batches = int(np.ceil(len(y) / batch_size))
  for i in range(n_batches):
    start = i * batch_size
    end = min((i+1) * batch_size, len(y))

    X_batch = X[start:end]
    y_batch = y[start:end]

    # pad X_batch to be N x T
    max_len = np.max([len(x) for x in X_batch])

    for j in range(len(X_batch)):
      x = X_batch[j]
      pad = [0] * (max_len - len(x))
      X_batch[j] = pad + x

    # convert to tensor
    X_batch = torch.from_numpy(np.array(X_batch)).long()
    y_batch = torch.from_numpy(np.array(y_batch)).long()

    yield X_batch, y_batch

def plot_loss(train_losses, test_losses):
    # Plot the train loss and test loss per iteration
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.legend()
    plt.show()

def accuracy(model, train_gen, test_gen):
    # Accuracy
    n_correct = 0.
    n_total = 0.
    for inputs, targets in train_gen():
      inputs = inputs.to(device)
      targets = targets.view(-1,1).float().to(device)
      outputs = model(inputs)

      # Get prediction
      predictions = (outputs > 0)
      n_correct += (predictions == targets).sum().item()
      n_total += targets.shape[0]
    train_acc = n_correct / n_total
    print(f"Train accuracy: {train_acc:.4f}")


    # Accuracy
    n_correct = 0.
    n_total = 0.
    for inputs, targets in test_gen():
      inputs = inputs.to(device)
      targets = targets.to(device)
      outputs = model(inputs)

      # Get prediction
      predictions = (outputs > 0)
      n_correct += (predictions == targets).sum().item()
      n_total += targets.shape[0]
    test_acc = n_correct / n_total
    print(f"Test accuracy: {test_acc:.4f}")

def main():
    """Main function of the program."""

    begin_time_main = datetime.now()
    print("Begin time: ", begin_time_main.strftime("%m/%d/%Y, %H:%M:%S"))

    df = getDataFrame("data/24_04_24_21_01_clean_training.csv")
    df["clean_text"] = df["clean_text"].map(strip_outer_quotes)
    print(df.head())
    df = df.drop(['star_rating'], axis=1)
    df.columns = ['clean_text', 'human_tag']
    df_train, df_test = train_test_split(df, test_size=0.33, random_state=123)
    # clean_text,star_rating,human_tag
    print(df_train.shape)
    print(df_test.shape)
    word2idx = tokenize_train_data(df_train)
    print(len(word2idx))
    train_sents = convert_to_indices(df_train, word2idx)
    test_sents = tokenize_test_data(df_test, word2idx)
    print(f"Train Length: {len(train_sents)}")
    print(f"Test Length: {len(test_sents)}")
    for inputs, targets in data_generator(train_sents, df_train['human_tag']):
        print("train inputs", inputs, "shape:", inputs.shape)
        print("train targets", targets, "shape:", targets.shape)
        break
    for inputs, targets in data_generator(test_sents, df_test['human_tag']):
        print("test inputs", inputs, "shape:", inputs.shape)
        print("test targets", targets, "shape:", targets.shape)
        break
    model = RNN(n_vocab=len(word2idx), embed_dim=20, n_hidden=15, n_rnnlayers=1, n_classes=1)
    model.to(device)
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_gen = lambda: data_generator(train_sents, df_train['human_tag'])
    test_gen = lambda: data_generator(test_sents, df_test['human_tag'])
    train_losses, test_losses = batch_gd(model, criterion, optimizer, TRAIN_EPOCHS, train_gen, test_gen)
    plot_loss(train_losses, test_losses)
    accuracy(model, train_gen, test_gen)

    end_time = datetime.now()
    print("End time: ", end_time.strftime("%m/%d/%Y, %H:%M:%S"))
    print("Duration: " + str(end_time - begin_time_main))

if __name__ == "__main__":
    main()
