import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class GraphConvolution(nn.Module):
    def __init__(self,in_features,out_features):
        super(GraphConvolution,self).__init__()
        self.linear = nn.Linear(in_features,out_features)
    
    def forward(self,x,adj):
        x = torch.matmul(adj,x)
        x = self.linear(x)
        return x 
    
class GCN(nn.Module):
    def __init__(self,n_features,n_classes):
        super(GCN,self).__init__()
        self.conv1 = GraphConvolution(n_features,16)
        self.conv2 = GraphConvolution(16,n_classes)
    
    def forward(self,x,adj):
        x = self.conv1(x,adj)
        x = F.relu(x)
        x = F.dropout(x,0.5,training=self.training)
        x = self.conv2(x,adj)
        return x
    
def generate_synthetic_data(num_nodes=50, num_features=2):
    G = nx.random_geometric_graph(num_nodes, radius=0.2)
    
    # Generate binary adjacency matrix
    adj = nx.adjacency_matrix(G).toarray()
    adj = torch.tensor(adj, dtype=torch.float32)
    
    # Generate random features for each node
    features = np.random.rand(num_nodes, num_features)
    features = torch.tensor(features, dtype=torch.float32)

    labels = np.random.randint(0, 2, num_nodes)
    labels = torch.tensor(labels, dtype=torch.long)

    return G, adj, features, labels

    
def train_model(model,features,adj,labels,epochs=100):
    optimizer = optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits= model(features,adj)
        loss = criterion(logits,labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epochs: {epoch} , loss: {loss.item()}")

def visualize_graph(G,labels,pred_labels,title="Graph Convolutional Network"):
    pos = nx.get_node_attributes(G,"pos")
    plt.figure(figsize=(8,6))
    nx.draw(G,pos,node_size=300,node_color=pred_labels,cmap=plt.cm.RdYlBu,with_labels=True)
    plt.title(title)
    plt.show()



        
        


G,features,adj,labels =  generate_synthetic_data()

print("Shape of adj:", adj.shape)
print("Shape of features:", features.shape)


gcn = GCN(n_features=2,n_classes=2)
train_model(gcn , features,adj,labels)

gcn.eval()
with torch.no_grad():
    logits = gcn(features,adj)
    pred_labels = torch.argmax(logits,dim=1).numpy()

visualize_graph(G,labels,pred_labels )