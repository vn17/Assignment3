import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformer import PositionalEncoding, TransformerLayer

class LanguageModel(nn.Module):  # Inherit from nn.Module
    def __init__(self, vocab_size):
        super(LanguageModel, self).__init__()  # Initialize nn.Module
        self.vocab_size = vocab_size

    def get_next_char_log_probs(self, context) -> np.ndarray:
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        super(UniformLanguageModel, self).__init__(voc_size)  # Call the base class initializer
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):  # Inherit from LanguageModel
    def __init__(self, vocab_size, d_model, d_internal, num_layers, vocab_index, max_seq_len=20):
        super(NeuralLanguageModel, self).__init__(vocab_size)  # Call the base class initializer
        self.d_model = d_model
        self.vocab_index = vocab_index  # Store vocab_index as an instance variable

        # Define layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.transformer_layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, input_indices):
        # Embed the input and add positional encodings
        x = self.embedding(input_indices)
        x = self.positional_encoding(x)

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x, _ = layer(x)  # Ignore attention maps for simplicity

        # Project to the vocabulary size and return log probabilities
        logits = self.output_layer(x)
        return nn.LogSoftmax(dim=-1)(logits)  # Calculate log probabilities here

    def get_next_char_log_probs(self, context):
        """Get log probabilities for the next character given the context."""
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            input_indices = torch.LongTensor([self.vocab_index.index_of(c) for c in context]).unsqueeze(0)
            log_probs = self.forward(input_indices)
            return log_probs.squeeze(0)[-1].detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        """Score a sequence of next characters given a context."""
        log_prob_sum = 0.0
        current_context = context
        for char in next_chars:
            log_probs = self.get_next_char_log_probs(current_context)
            char_index = self.vocab_index.index_of(char)  # Use stored vocab_index
            log_prob_sum += log_probs[char_index]
            current_context += char  # Update context
        return log_prob_sum


def train_lm(args, train_text, dev_text, vocab_index):
    model = NeuralLanguageModel(
        vocab_size=len(vocab_index),
        d_model=128,
        d_internal=256,
        num_layers=2,
        vocab_index=vocab_index,  # Pass vocab_index here
        max_seq_len=20
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Using the custom parameters method
    loss_fn = nn.NLLLoss()

    # Prepare the dataset
    train_data = [train_text[i:i + 20] for i in range(len(train_text) - 20)]
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # Convert batch of sequences to indices
            input_sequences = [torch.LongTensor([vocab_index.index_of(c) for c in seq]) for seq in batch]
            input_tensor = torch.stack(input_sequences)

            # Shift inputs to predict next characters
            target_tensor = input_tensor[:, 1:]
            input_tensor = input_tensor[:, :-1]

            # Run model forward pass
            log_probs = model(input_tensor)

            # Compute loss
            loss = loss_fn(log_probs.reshape(-1, len(vocab_index)), target_tensor.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

    model.eval()  # Set model to evaluation mode
    return model