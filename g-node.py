# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:20:39 2024

@author: Luisa
"""

import time
import torch
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint

 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  
from torch_geometric.data import Data  

#from torch_geometric.datasets import Planetoid  # Dataset Cora
#from torch_geometric.transforms import NormalizeFeatures
#dataset = Planetoid(root='new_data2/Cora', name='Cora', transform=NormalizeFeatures())




x = torch.tensor([[3, 2], [4, 3], [5, 1]], dtype=torch.float)
true_y = torch.tensor([[3.2, 1.7], [3.8, 2.4], [2.3, 1.5]], dtype=torch.float)
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

data = Data(x=x, edge_index=edge_index)

y = torch.tensor([0, 1, 0], dtype=torch.float)
data.y = y


    
    
class ODEFunc(torch.nn.Module):
    def __init__(self, num_node_features):
        super(ODEFunc, self).__init__()
        self.conv1 = GCNConv(num_node_features, 5)
        self.conv2 = GCNConv(5, num_node_features)


    def forward(self, t, x):
        edge_index = self.edge_index  
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
                
        return x 
    
    
    
class Gnode(torch.nn.Module):
    def __init__(self, num_node_features):
        super(Gnode, self).__init__()
        self.ode_func = ODEFunc(num_node_features)
        #self.conv_out = GCNConv(num_node_features, 2) 
        

    def forward(self, data):
        x0, edge_index = data.x, data.edge_index  
        self.ode_func.edge_index = edge_index     

        t = torch.linspace(0, 1, steps=2)  
        x= odeint(self.ode_func, x0, t)   
        
        x = x[-1]
        out = x #self.conv_out(x, edge_index)
        
        return out 
    
    
        

model = Gnode(2)          
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
end = time.time()

for itr in range(1, 100):
    pred_y = model(data)              
    loss = torch.mean(torch.abs(pred_y - true_y))  
    loss.backward()
    optimizer.step()
    if itr % 20== 0:
        with torch.no_grad():
            pred_y = model(data)
            loss = torch.mean(torch.abs(pred_y - true_y))
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item())) 
