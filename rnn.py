import numpy as np


class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 0.01):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        #weights init
        self.wxh = np.random.randn(hidden_size,input_size)* 0.2
        self.whh = np.random.randn(hidden_size,hidden_size)* 0.2
        self.wyh = np.random.randn(output_size, hidden_size) * 0.2
        
        self.bh  = np.zeros((hidden_size,1))
        self.by = np.zeros((output_size,1))


    def forward_prop(self, inputs):
        h = {-1: np.zeros((self.hidden_size,1))}
        y_preds = {}

        for t in range(len(inputs)):
            xt = np.array([[inputs[t]]])

            h[t] = np.tanh(self.wxh @ xt + self.whh @ h[t-1] + self.bh)
            y_preds[t] = self.wyh @ h[t] + self.by

        return h, y_preds

    def backward_prop(self, inputs, targets, h, y_preds):
        dwxh, dwhh, dwyh = np.zeros_like(self.wxh), np.zeros_like(self.whh), np.zeros_like(self.wyh)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(h[0])

        for t in reversed(range(len(inputs))):
            xt = np.array([[inputs[t]]])
            dy = y_preds[t] - targets[t]
            dwyh += dy @ h[t].T
            dby += dy

            # combining output and future gradient
            dh = self.wyh.T @ dy + dh_next
            # dl/dz
            dtanh = (1-h[t] ** 2) * dh
            # bptt in action
            dwxh += dtanh @ xt.T
            dwhh += dtanh @ h[t-1].T
            dbh += dtanh 
            dh_next = self.whh.T @ dtanh

        #clipping value to prevent exploding and vanishing gradinet
        for dparam in [dwxh, dwhh, dwyh, dbh, dby]:
            np.clip(dparam, -1, 1, out=dparam)
        
        # Update Weights
        self.wxh -= self.learning_rate * dwxh
        self.whh -= self.learning_rate * dwhh
        self.wyh -= self.learning_rate * dwyh
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby


