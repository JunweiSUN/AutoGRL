from sklearn.metrics import accuracy_score, f1_score

class Evaluator:
    def __init__(self, metric='acc'):
        self.metric = metric

    def __call__(self, y_true, y_pred):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()

        if self.metric == 'acc':
            return accuracy_score(y_true, y_pred)
        elif self.metric == 'f1':
            return f1_score(y_true, y_pred, average='micro')
        else:
            raise NotImplementedError('Not supported metric')