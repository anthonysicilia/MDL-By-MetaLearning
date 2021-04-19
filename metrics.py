import torch

class Critic:

    def __init__(self):
        pass
    
    def list_template(self):
        return {
            'ACC': [],
            'PPV': [],
            'TPR': [],
            'FPR': [],
            'DSC': [],
            'BS': []
        }
    
    def __call__(self, scores, y):

        bs = ((y - scores.sigmoid())**2).mean()
        yhat = (scores.sigmoid() > .5).long()

        # -> we cannot assume y, yhat are on cpu to use sklearn 
        # -> below is a torch approach that assumes 2 classes
        # - > implementation adapted from 
        # gist.github.com/the-bass/cae9f3976866776dea17a5049013258d

        confusion_vector = yhat.float() / y.float()
        # Element-wise division of the 2 tensors returns a new 
        # tensor which holds a unique value for each case:
        #   1     :prediction and truth are 1 (True Positive)
        #   inf   :prediction is 1 and truth is 0 (False Positive)
        #   nan   :prediction and truth are 0 (True Negative)
        #   0     :prediction is 0 and truth is 1 (False Negative)

        TP = torch.sum(confusion_vector == 1).item()
        FP = torch.sum(torch.isinf(confusion_vector)).item()
        TN = torch.sum(torch.isnan(confusion_vector)).item()
        FN = torch.sum(confusion_vector == 0).item()
        # adds some easy checks to avoid 0 denoms
        return {
            'ACC': (TP+TN)/(TP+FP+FN+TN),
            'PPV': TP/(TP+FP) if TP != 0 else 0,
            'TPR': TP/(TP+FN) if TP != 0 else 0,
            'FPR': FP/(FP+TN) if FP != 0 else 0,
            'DSC': 2*TP/(2*TP+FP+FN) if TP != 0 else 0,
            'BS': bs.item()
        }
