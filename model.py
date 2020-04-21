import torch

class IDraggn(torch.nn.Module):
    def __init__(self, vocab_size, out_size, drop_prob):
        super(IDraggn, self).__init__()
        self.embed_layer = torch.nn.Embedding(vocab_size, 25)
        self.gru_layers = torch.nn.GRU(input_size=25, hidden_size=32, num_layers=2)
        self.fc = torch.nn.Linear(32, out_size)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(p=drop_prob)

    def forward(self, input):
        emb = self.embed_layer(input)
        _, hid1 = self.gru_layers(self.drop(emb))
        hidden = self.fc(hid1[-1])
        out = self.relu(hidden)
        # We don't need to softmax the output because 
        # PyTorch includes softmax in cross-entropy loss!

        return out