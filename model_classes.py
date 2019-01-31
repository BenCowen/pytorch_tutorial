

import torch
import torch.nn as nn

class playNet(nn.Module):
  """
  Homemade network.
  """
  def __init__(self, in_size, hidden_size, num_classes, bias=True):
    super(playNet, self).__init__()
    # This part of the code is always run when you create
    #  a new instance of this class. Give it attributes
    #  that you want it to have.
    self.A1    = nn.Linear(in_size, hidden_size, bias=bias)
    self.sigma = nn.Softshrink()
    self.A2    = nn.Linear(hidden_size, num_classes, bias=bias)

  # The forward function is where you define the links between attributes.
  # ie you define the directed graph/ network here.
  def forward(self, inputs):
    x1 = self.A1(inputs)
    x2 = self.sigma(x1)
    x3 = self.A2(x2)
    return x3


class playNet_recurrent(nn.Module):
  """
  Homemade RNN that illustrates flexibility of forward method.
    (and that cool nets don't necessarily perform better)
  """
  def __init__(self, in_size, hidden_size, num_classes, nloops=10, bias=True):
    super(playNet_recurrent, self).__init__()
    # This part of the code is always run when you create
    #  a new instance of this class. Give it attributes
    #  that you want it to have.
    self.input_layer     = nn.Linear(in_size, hidden_size, bias=bias)
    self.recurrent_layer = nn.Linear(hidden_size, hidden_size, bias=False)
    self.final_layer     = nn.Linear(hidden_size, num_classes, bias=bias)

    self.sigma  = nn.Softshrink()
    self.nloops = nloops

  # The forward function is where you define the links between attributes.
  # ie you define the directed graph/ network here.
  def forward(self, inputs):
    x = self.input_layer(inputs)

    # You can put all kinds of combinations of attributes and inputs.
    # Complex does not equal better.
    for loop in range(self.nloops):
      x  = self.sigma(x)
      x  = self.recurrent_layer(x)

    x  = self.sigma(x)
    output = self.final_layer(x)
    return output


##########################################################################################
class LeNet5(nn.Module):
    '''
    LeNet-5, based on
    https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
    '''
    def __init__(self, num_input_channels=3, num_classes=10, window_size=32, bias=True):
        super(LeNet5, self).__init__()
        self.bias = bias
        self.window_size = window_size
        self.features = nn.Sequential(
            nn.Conv2d(num_input_channels, 6, 5, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(6, 16, 5, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * int((int((window_size-4)/2)-4)/2)**2, 120, bias=bias),
            nn.ReLU(),
            nn.Linear(120, 84, bias=bias),
            nn.ReLU(),
            nn.Linear(84, num_classes, bias=bias),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

##########################################################################################
## This computes model accuracy
def test(model, data_loader, criterion=nn.CrossEntropyLoss(), label=''):
    '''
    Compute model accuracy.
    '''
    model.eval()
    device = next(model.parameters()).device

    # When using validation set the "len" fcn doesn't take into
    # account that you're not using the whole dataset.
    if hasattr(data_loader, 'numSamples'):
      N = data_loader.numSamples
    else:
      N = len(data_loader.dataset)

    # Now loop thru the data WITHOUT saving gradients.
    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            test_loss += criterion(output, target).item()
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

    accuracy = float(correct)/N
    test_loss /= len(data_loader) # loss function already averages over batch size??
    if label:
        print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            label, test_loss, correct, N, 100. * accuracy ))
    return accuracy









