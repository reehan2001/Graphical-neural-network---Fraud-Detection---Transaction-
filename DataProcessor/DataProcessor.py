import torch
from torch_geometric.data import Data


class DataProcessor:
    def __init__(self,fraud_tractions,high_amount_tranctions,records):
        self.fraud_tractions = fraud_tractions
        self.high_amount_tranctions = high_amount_tranctions
        self.records = records

    def process_data(self):
        fraud_transaction_ids = [transaction_id for transaction_id , _ in self.fraud_tractions]
        high_amount_tranctions_ids = [transaction_id for transaction_id , _ in self.high_amount_tranctions]
        all_transaction_ids = list(set(fraud_transaction_ids + high_amount_tranctions_ids))
        transaction_id_to_index = {transaction_id: index for index,transaction_id in enumerate(all_transaction_ids)}
        faurd_transaction_indices = [transaction_id_to_index[tid] for tid in fraud_transaction_ids]
        high_amount_tranctions_indices = [transaction_id_to_index[tid] for tid in high_amount_tranctions_ids]
        edge_index = torch.tensor([
            faurd_transaction_indices + high_amount_tranctions_indices,
            [0] * len(faurd_transaction_indices) + [1] * len(high_amount_tranctions_indices)
        ],dtype=torch.long)
        x = torch.tensor([
            [amount] for _,_, amount in self.records
        ],dtype=torch.float)
        num_nodes = len(all_transaction_ids)
        y = torch.zeros(num_nodes,dtype=torch.long)
        for idx in faurd_transaction_indices:
            y[idx] = 1
        fraud_indices = [i for i , label in enumerate(y) if label == 1]
        oversample_index = fraud_indices * (num_nodes // len(fraud_indices) - 1)
        x = torch.cat([x,x[oversample_index]],dim=0)
        y = torch.cat([y,y[oversample_index]],dim=0)
        num_nodes = len(y)
        data = Data(x=x , edge_index=edge_index,y=y)
        train_mask = torch.zeros(num_nodes,dtype=torch.bool)
        train_mask[:int(0.8 * num_nodes)] = 1
        data.train_mask = train_mask
        test_mask = torch.zeros(num_nodes,dtype=torch.bool)
        test_mask[int(0.8 * num_nodes):] = 1
        data.test_mask = test_mask
        return data

        

        
