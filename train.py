import torch
from model import IDraggn

# These constants here are just dummy values for the moment.
# If one actually needed to implement the whole pipeline of training 
# this model on a dataset, one could change these accordingly or pass them 
# as arguments to main
INPUT_VOCAB_SIZE = 5
NUM_CLASSES = 32

# Sample input data from a 5 word vocab
data = torch.LongTensor([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], \
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

true_calls = torch.LongTensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
true_binds = torch.LongTensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

if __name__ == "__main__":
    model_call = IDraggn(INPUT_VOCAB_SIZE, NUM_CLASSES, 0.1)
    model_bind = IDraggn(INPUT_VOCAB_SIZE, NUM_CLASSES, 0.1)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_call = torch.optim.Adam(model_call.parameters(), lr=0.01)
    optimizer_bind = torch.optim.Adam(model_bind.parameters(), lr=0.01)

    for epoch in range(500):
        calls = model_call(data)
        binds = model_bind(data)

        loss_call = criterion(calls, true_calls)
        loss_bind = criterion(binds, true_binds)
        
        optimizer_call.zero_grad()
        optimizer_bind.zero_grad()
        
        loss_call.backward()
        loss_bind.backward()

        optimizer_call.step()
        optimizer_bind.step()


