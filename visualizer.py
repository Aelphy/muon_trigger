import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

plt.style.use('ggplot')

class Visualizer():
    def __init__(self, network):
        self.watches = {
            'auc' : self.auc,
            'loss' : self.loss,
        }

        self.network = network
        self.losses = []

    def watch(self, iterator, watch_list=['auc', 'loss']):
        '''
            iterator should support direct iteration, like:
                for xi, xr in iterator:
                    pass
        '''
        for batch in iterator:
            batch_size = batch[0].shape[0]

            x, y = batch

            if 'auc' in watch_list:
                # AUC
                self.targets = y
                self.answers = self.network.forward_pass(x)
                self.watches['auc']()

            if 'loss' in watch_list:
                # loss
                self.losses.append(self.network.eval_network(y, self.network.forward_pass(x)))
                self.watches['loss']()

    def auc(self):
        fpr, tpr, thr = roc_curve(np.hstack(self.targets),
                                  np.vstack(self.answers))

        plt.plot(fpr, tpr)
        plt.title('AUC = {0}'.format(roc_auc_score(np.hstack(self.targets), np.vstack(self.answers))))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('fpr')
        plt.ylabel('tpr')

        self.targets = []
        self.answers = []

        plt.show()

    def loss(self):
        plt.plot(self.losses)
        plt.title('Loss, iteration {0}'.format(len(self.losses) + 1))
        plt.xlabel('num iterations')
        plt.ylabel('loss')
        plt.show()
