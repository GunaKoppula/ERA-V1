# ERA V1 Session 6

## Part 1 - Backpropagation

- Forward propagation

First, we compute the output of the neural network by propagating the input data through the network's layers. Each layer has a set of weights to the input and passes the result through an activation function and then calculate the loss.

```ruby
h1 = w1*i1 + w2*i2		
h2 = w3*i1 + w4*i2		
a_h1 = σ(h1) = 1/(1 + exp(-h1))		
a_h2 = σ(h2)		
o1 = w5*a_h1 + w6*a_h2		
o2 = w7*a_h1 + w8*a_h2		
a_o1 = σ(o1)		
a_o2 = σ(o2)	
E_total = E1 + E2		
E1 = ½ * (t1 - a_o1)²		
E2 = ½ * (t2 - a_o2)²		
```

- Backward propagation

```ruby
∂E_total/∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1					
∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2					
∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1					
∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2					

∂E_total/∂w1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1
∂E_total/∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2												
∂E_total/∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1												
∂E_total/∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2												
```

### Loss curve with change in learning rate

#### learning rate - 0.1
![learning_rate_0 1](https://github.com/GunaKoppula/ERA-V1/assets/61241928/296b92fb-fd09-4809-a633-322da07b2b40)

#### learning rate - 0.2
![learning_rate_0 2](https://github.com/GunaKoppula/ERA-V1/assets/61241928/8824eafe-c346-441d-aee5-fff69a93a22a)

#### learning rate - 0.5
![learning_rate_0 5](https://github.com/GunaKoppula/ERA-V1/assets/61241928/5b41e484-4c76-4b0d-b86e-5b9ff05bd8ed)

#### learning rate - 0.8
![learning_rate_0 8](https://github.com/GunaKoppula/ERA-V1/assets/61241928/f1f1acc3-e875-432b-bc4b-f2109064db3e)

#### learning rate - 1.0
![learning_rate_1](https://github.com/GunaKoppula/ERA-V1/assets/61241928/c089962d-6e11-40b8-a557-414460d63375)

#### learning rate - 2.0
![learning_rate_2](https://github.com/GunaKoppula/ERA-V1/assets/61241928/16772f44-3825-4076-8ac2-20598f267d79)





## Part 2

## Create and Train a Neural Network in Python

### Usage
### S6.ipynb

- First we have to import all the neccessary libraries.

```ruby
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
```
- Next we build a simple Neural Network.
For this, we define a **class Net()** and pass **nn.Module** as the parameter.

```ruby
class Net(nn.Module):
```

- Create two functions inside the class to get our model ready. First is the **init()** and the second is the **forward()**.
- We need to instantiate the class for training the dataset. When we instantiate the class, the forward() function will get executed.

```ruby
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.batch_1 = nn.BatchNorm2d(8)
        self.drop_1 = nn.Dropout(0.25)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.batch_2 = nn.BatchNorm2d(16)
        self.drop_2 = nn.Dropout(0.25)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.batch_3 = nn.BatchNorm2d(16)
        self.drop_3 = nn.Dropout(0.2)
        
        self.conv4 = nn.Conv2d(16,32, 3, padding=1)
        self.batch_4 = nn.BatchNorm2d(32)
        self.drop_4 = nn.Dropout(0.2)
        
        
        self.conv5 = nn.Conv2d(32,16,3, padding=1)
        self.batch_5 = nn.BatchNorm2d(16)
        self.drop_5 = nn.Dropout(0.2)
        
        self.conv6 = nn.Conv2d(16, 10, 1)
        self.batch_5 = nn.BatchNorm2d(16)
        self.drop_5 = nn.Dropout(0.2)        
        
        self.conv7 = nn.Conv2d(10,10,7)

    def forward(self, x):
        x = self.pool1(self.drop_1((F.relu(self.batch_1(self.conv1(x))))))                     
        x = self.pool2(self.drop_2(F.relu(self.batch_2(self.conv2(x)))))      
        x = self.drop_3(F.relu(self.batch_3((self.conv3(x)))))        
        x = self.drop_4(F.relu(self.batch_4((self.conv4(x)))))        
        x = self.drop_5(F.relu(self.batch_5((self.conv5(x)))))        
        x = F.relu(self.conv6(x))
                
        x = self.conv7(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x)
 ```


- Next we created two functions **train()** and **test()**
- train() funtion computes the prediction, traininng accuracy and loss

```ruby
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')
```

- And test() function calculates the loss and accuracy of the model

```ruby
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```


- **Training and Testing trigger**
-
```ruby
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

for epoch in range(0, 20):
    print(f"EPOCH: {epoch+1}")
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
```

I used total 20 epoch
```
EPOCH: 20
loss=0.007352722343057394 batch_id=468: 100%|██████████| 469/469 [01:15<00:00,  6.23it/s] 

Test set: Average loss: 0.0262, Accuracy: 9920/10000 (99.200%)
```


## Model Summary
![summary](https://github.com/GunaKoppula/ERA-V1-Session-6/assets/61241928/105eaa7c-9dbd-4b3f-9ff4-8e61714deb43)


