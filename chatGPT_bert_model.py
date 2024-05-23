#1. Extracting Text from a PDF--------------------------------------------------------------------------------------------
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

pdf_path = 'path/to/your/pdf_file.pdf'
corpus_text = extract_text_from_pdf(pdf_path)

#2. Tokenizing the Text--------------------------------------------------------------------------------------------------

from collections import Counter
import numpy as np
import torch

class SimpleTokenizer:
    def __init__(self, vocab_size=30522):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = []
        self.build_vocab()

    def build_vocab(self):
        words = corpus_text.split()
        vocab = Counter(words).most_common(self.vocab_size - 2)
        self.idx2word = ['[PAD]', '[UNK]'] + [word for word, _ in vocab]
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}

    def tokenize(self, text):
        return [self.word2idx.get(word, 1) for word in text.split()]

    def detokenize(self, token_ids):
        return ' '.join([self.idx2word[token] for token in token_ids])

tokenizer = SimpleTokenizer()

#3. Defining the BERT Model from Scratch-----------------------------------------------------------

import torch.nn as nn
import math

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(BertSelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(2)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, dropout_prob):
        super(BertLayer, self).__init__()
        self.attention = BertSelfAttention(hidden_size, num_attention_heads, dropout_prob)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layer_norm(layer_output + attention_output)
        return layer_output

class BertModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, max_position_embeddings, dropout_prob):
        super(BertModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layers = nn.ModuleList([BertLayer(hidden_size, intermediate_size, num_attention_heads, dropout_prob) for _ in range(num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, attention_mask):
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        word_embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        return hidden_states
    
#4. Masked Language Modeling (Object-Oriented)------------------------------------------------------------


class MaskedLanguageModeling:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.vocab_size = len(tokenizer.word2idx)

    def mask_tokens(self, inputs):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [[1 if token in [0, 1] else 0 for token in seq] for seq in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.word2idx.get('[MASK]', 3)  # Mask token ID (default to 3)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels
#5. Training Loop------------------------------------------------------------------------------------

from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer.tokenize(self.texts[idx])
        tokens = tokens[:self.max_length]
        tokens += [0] * (self.max_length - len(tokens))  # Padding
        return torch.tensor(tokens)

# Create dataset and dataloader
texts = corpus_text.split('\n')  # Assume each line is a separate text
dataset = TextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize MaskedLanguageModeling
mlm = MaskedLanguageModeling(tokenizer)

# Define model, optimizer, and loss function
vocab_size = len(tokenizer.word2idx)
model = BertModel(vocab_size=vocab_size, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512, dropout_prob=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(3):  # 3 epochs for demonstration
    for step, batch in enumerate(dataloader):
        inputs, labels = mlm.mask_tokens(batch)
        attention_mask = (inputs != 0).float()  # 0 is the pad token ID

        outputs = model(inputs, attention_mask)
        outputs = outputs.view(-1, vocab_size)
        labels = labels.view(-1)
        
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

# Save the pre-trained model
torch.save(model.state_dict(), './bert_pretrained.pth')









