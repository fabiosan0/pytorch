import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


#Define classifier
class Regressor(nn.Module):
    
    def __init__(self):
        # Inherient from parent pytorch class
        super().__init__()
        #perform operations in sequence
        #use sigmoid activation function
        self.model = nn.Sequential(
          nn.Linear(4, 3),
          nn.Sigmoid(),

          nn.Linear(3, 2),
          nn.Sigmoid(),

          nn.Linear(2, 1),
      )
        self.forward_pass_counter=0
        self.running_loss=[]
        #Loss function is MSE for regression problem
        self.loss_function = nn.MSELoss()
        
        #Use stochastic gradient descent as optimizer 
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
    def forward(self, inputs):
      return self.model(inputs)

        
    def train(self, inputs, targets):
    # calculate output of n/w
      outputs = self.forward(inputs)

      # initialize the gradients to zero; otherwise it will accumulate on each pass
      self.optimiser.zero_grad()
      
      # generate loss
      loss = self.loss_function(outputs, targets)


      #backward pass
      loss.backward()

      #update the weight
      self.optimiser.step()

      self.forward_pass_counter+=1
      
      #why do we need to keep running loss?
      self.running_loss.append(loss.item())

      if (self.forward_pass_counter % 10==0):
        print('the forward pass counter is ', self.forward_pass_counter)
        print('the loss ', loss.item())

    def plot_progress(self):
      df = pd.DataFrame(self.running_loss, columns=['loss'])
      df.plot( grid=True)
      pass



     
    
      pass

#Read sample file
df=pd.read_csv('AAPL.csv')
print(df.columns)
print(df.head())


#remove null values
df['Next_day']=df['Close'].shift(-1)
df=df.dropna()

# convert to numpy
x_numpy= df[['Open', 'High',
       'Low', 'Close', ]].to_numpy()
y_numpy=df['Next_day'].to_numpy()
print(x_numpy[0])
print(y_numpy[0])

# convert to tensor
from torch.utils.data import Dataset, TensorDataset, DataLoader

x_tensor = torch.from_numpy(x_numpy).float()
y_tensor = torch.from_numpy(y_numpy).float()
train_dataset=TensorDataset(x_tensor,y_tensor)
print(train_dataset[0])

# create the training data with batch size of 5
train_loader = DataLoader(dataset=train_dataset, batch_size=5)

#call our classifier
regressor=Regressor()

for i, data in enumerate(train_loader, 0):
  inputs, labels=data
  print (inputs)
  print(inputs.size())
  print(labels)
  print(labels.size())
  break

for i, data in enumerate(train_loader, 0):
  inputs, labels=data
  regressor.train(inputs,torch.unsqueeze(labels,1))

  regressor.plot_progress()