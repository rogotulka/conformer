import torch
import torch.nn as nn
from conformer import Conformer

batch_size, sequence_length, dim = 3, 50, 80

cuda = torch.cuda.is_available()  
device = torch.device('cuda' if cuda else 'cpu')

criterion = nn.CTCLoss().to(device)

input = torch.randn(50,3, 10).log_softmax(2).detach().requires_grad_().to(device)
input_lengths = torch.LongTensor([50, 50, 50]).to(device)

model = Conformer().to(device)
output, output_lengths = model(input, input_lengths)

targets = torch.randint(low=1, high=10, size=(3, 9), dtype=torch.long)
target_lengths = torch.randint(low=6, high=9, size=(3,), dtype=torch.long)

ctc_loss = nn.CTCLoss()

loss = criterion(output, targets, output_lengths, target_lengths)
print(loss)