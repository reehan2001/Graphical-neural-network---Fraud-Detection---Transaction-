import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv




class ModelBuilder:
    class Net(torch.nn.Module):
        def __init__(self, num_node_features):
             super(ModelBuilder.Net,self).__init__()
             self.conv1 = GCNConv(num_node_features,32)
             self.conv2 = GCNConv(32,16)
             self.conv3 = GCNConv(16,2)

        def forward(self,data):
            x,edge_index = data.x,data.edge_index
            x = self.conv1(x,edge_index)
            x = F.relu()
            x = F.dropout(x,training=self.training)
            x = self.conv2(x,edge_index)
            x = F.relu(x)
            x = self.conv3(x,edge_index)
            return  F.log_softmax(x,dim=1)
        
    def __init__(self,data):
        self.data = data

    def build_model(self):
        device = torch.device('Cuda is avaiable' if torch.cuda.is_available() else 'Cuda is not ')
        model = ModelBuilder.Net(self.data.num_node_features).to(device)
        self.data = self.data.to(device)
        num_classes = 2
        class_counts = [sum(self.data.y == i) for i in range(num_classes) ]
        class_weight = [1.0 / (c + 1e-10) for c in class_counts]
        weights = torch.tensor(class_weight,dtype=torch.float).to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=5e-4)
        criterion = torch.nn.NLLLoss(weight=weights)
        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            out = model(self.data)
            loss = criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()
        return model