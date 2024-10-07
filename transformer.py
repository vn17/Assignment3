# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        super(Transformer, self).__init__()
        # print("VNVN Vocab Size: " + vocab_size)

        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions)

        # Transformer layers
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])

        # Output layer
        self.output_layer = nn.Linear(d_model, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, indices):
        # Embed the input indices and add positional encodings
        x = self.embedding(indices)
        # x = self.positional_encoding(x)

        # Store attention maps from each layer
        attention_maps = []

        # Pass through the transformer layers
        for layer in self.layers:
            x, attn_map = layer(x)
            attention_maps.append(attn_map)

        # Output layer to predict 0, 1, or 2 classes for each position
        output = self.output_layer(x)
        log_probs = self.softmax(output)

        # Return log probabilities and attention maps
        return log_probs, attention_maps

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        super(TransformerLayer, self).__init__()
        # Self-attention mechanism
        self.query = nn.Linear(d_model, d_internal)
        self.key = nn.Linear(d_model, d_internal)
        self.value = nn.Linear(d_model, d_internal)
        self.softmax = nn.Softmax(dim=-1)

        # Projection layer to project attention_output back to d_model
        self.proj = nn.Linear(d_internal, d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_internal),
            nn.ReLU(),
            nn.Linear(d_internal, d_model)
        )

    def forward(self, input_vecs):
        # Self-attention mechanism
        queries = self.query(input_vecs)
        keys = self.key(input_vecs)
        values = self.value(input_vecs)

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(queries.shape[-1])
        attention_probs = self.softmax(attention_scores)

        # Compute the attention output
        attention_output = torch.matmul(attention_probs, values)

        # Project attention output back to d_model for residual connection
        attention_output = self.proj(attention_output)

        # Add residual connection and apply feed-forward network
        output = attention_output + input_vecs  # Residual connection
        output = self.ffn(output)

        # Return the output and attention map for visualization
        return output, attention_probs
    

# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        super().__init__()
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: [batch_size, seq_len, embedding_dim] if batched
                  [seq_len, embedding_dim] otherwise
        :return: a tensor of the same size with positional embeddings added in
        """
        seq_len = x.shape[-2]  # Second-to-last dimension is the sequence length
        indices_to_embed = torch.arange(seq_len).type(torch.LongTensor).to(x.device)
        
        if self.batched:
            # Add positional encoding to each sequence in the batch
            emb = self.emb(indices_to_embed).unsqueeze(0)  # [1, seq_len, d_model]
            return x + emb
        else:
            # print("VNVN Indices: " + str(indices_to_embed.shape))
            return x + self.emb(indices_to_embed)

def train_classifier(args, train, dev):
    model = Transformer(vocab_size=27, num_positions=20, d_model=128, d_internal=64, num_classes=3, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fcn = nn.NLLLoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        random.shuffle(train)
        for example in train:
            optimizer.zero_grad()
            
            # Run the model and unpack the log_probs and attention maps
            log_probs, _ = model(example.input_tensor)  # Ignore attention maps during training

            # Compute the loss using only the log_probs
            loss = loss_fcn(log_probs, example.output_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

    model.eval()
    return model

####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
