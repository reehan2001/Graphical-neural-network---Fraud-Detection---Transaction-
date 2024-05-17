
from sklearn.metrics import precision_score,recall_score,f1_score

class ModelEvaluator:
    def __init__(self,model,data):
        self.model = model
        self.data = data

    def evaluate(self):
        _, pred = self.model(self.data).max(dim=1)
        test_mask = self.data.test_mask

        true_labels = self.data.y[test_mask].cpu().numpy()
        pred_labels = pred[test_mask].cpu().numpy()

        if sum(pred_labels) == 0:
            print("No Positive Sample predicted , setting precision and recall to zero")
            precision , recall , f1 = 0,0,0
        else:
            precision = precision_score(true_labels,pred_labels)
            recall = recall_score(true_labels,pred_labels)
            f1 = f1_score(true_labels,pred_labels)

        print("precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1 - Score : {:.4f}".format(f1))

