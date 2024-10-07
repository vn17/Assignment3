import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformer import PositionalEncoding, TransformerLayer

class LanguageModel(nn.Module):  # Inherit from nn.Module
    def __init__(self, vocab_size):
        super(LanguageModel, self).__init__()
        self.vocab_size = vocab_size

    def get_next_char_log_probs(self, context) -> np.ndarray:
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        super(UniformLanguageModel, self).__init__(voc_size)
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.vocab_size]) * np.log(1.0 / self.vocab_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):  # Inherit from LanguageModel
    def __init__(self, vocab_size, d_model, d_internal, num_layers, vocab_index, max_seq_len=20):
        super(NeuralLanguageModel, self).__init__(vocab_size)
        self.d_model = d_model
        self.vocab_index = vocab_index

        # Define layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.transformer_layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)  # Define softmax

    def forward(self, indices):
        # Embed the input indices
        x = self.embedding(indices)
        sh = x.shape
        
        # Apply positional encoding
        x = self.positional_encoding(x)
        if x.shape != sh:
            print(f"Shape of x after embedding: {sh}")  # Debugging shape after embedding
            print(f"Shape of x after positional encoding: {x.shape}")  # Debugging shape after positional encoding

        attention_maps = []
        for layer in self.transformer_layers:  # Use the correct name here
            x, attn_map = layer(x)
            attention_maps.append(attn_map)

        # Final output layer
        output = self.output_layer(x)
        log_probs = self.softmax(output)  # Apply softmax to get log probabilities
        return log_probs, attention_maps

    def get_next_char_log_probs(self, context):
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            if len(context) == 0:
                return np.log(np.ones([self.vocab_size]) / self.vocab_size)

            # Convert context to tensor indices
            indices = []
            for c in context:
                index = self.vocab_index.index_of(c)
                if index >= self.vocab_size:
                    raise ValueError(f"Character '{c}' is mapped to index {index}, which is out of bounds for vocab size {self.vocab_size}.")
                indices.append(index)

            input_indices = torch.LongTensor(indices).unsqueeze(0)

            # Ensure the input length does not exceed max_seq_len
            if input_indices.shape[1] > 20:  # max_seq_len
                input_indices = input_indices[:, -20:]  # Take the last 20 characters

            log_probs, _ = self.forward(input_indices)  # Ignore attention maps during inference
            return log_probs.squeeze(0)[-1].detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        log_prob_sum = 0.0
        current_context = context
        for char in next_chars:
            log_probs = self.get_next_char_log_probs(current_context)
            char_index = self.vocab_index.index_of(char)
            log_prob_sum += log_probs[char_index]
            current_context += char  # Update context
        return log_prob_sum


def train_lm(args, train_text, dev_text, vocab_index):
    model = NeuralLanguageModel(
        vocab_size=len(vocab_index),
        d_model=128,
        d_internal=256,
        num_layers=2,
        vocab_index=vocab_index,
        max_seq_len=20
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.NLLLoss()

    # Prepare the dataset
    train_data = [train_text[i:i + 20] for i in range(len(train_text) - 20)]
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            input_sequences = [torch.LongTensor([vocab_index.index_of(c) for c in seq]) for seq in batch]
            input_tensor = torch.stack(input_sequences)

            # Shift inputs to predict next characters
            target_tensor = input_tensor[:, 1:]
            input_tensor = input_tensor[:, :-1]

            # Run model forward pass
            log_probs, _ = model(input_tensor)  # Include attention maps here if needed

            # Compute loss
            loss = loss_fn(log_probs.reshape(-1, len(vocab_index)), target_tensor.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

    model.eval()
    return model