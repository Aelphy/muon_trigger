import matplotlib.pyplot as plt
import numpy as np

class Visualizer():
    def __init__(self,
                 num_epochs,
                 num_cascades,
                 watches,
                 metrics=['obj', 'recall', 'precision', 'accuracy', 'loss', 'sub_loss', 'complexity']):
        self.metrics = metrics
        self.watches = watches
        
        self.train_metrics = []
        for i in range(len(metrics)):
            self.train_metrics.append([])
        
        self.val_metrics = []
        for i in range(len(metrics)):
            self.val_metrics.append([])
        
        self.fig1, ax1 = plt.subplots(1, 1)
        self.fig2, ax2 = plt.subplots(1, 1)
        ax1.set_xlabel('Num Epoch')
        ax1.set_ylabel('Metric Value')
        ax1.set_xlabel('Num Epoch')
        ax1.set_ylabel('Metric Value')
        ax2.set_xlabel('Num Epoch')
        ax2.set_ylabel('Metric Value')
        self.lines1 = []
        self.lines2 = []
        
        if 'total_complexity' in watches or 'complexity_parts' in watches:
            self.fig3, ax3 = plt.subplots(1, 1)
            self.fig4, ax4 = plt.subplots(1, 1)
            ax3.set_xlabel('Num Epoch')
            ax3.set_ylabel('Train Complexity')
            ax4.set_xlabel('Num Epoch')
            ax4.set_ylabel('Val Complexity')
            self.lines3 = []
            self.lines4 = []
    
        self.template_space = np.zeros(num_epochs)
        
        for w in watches:
            i = metrics.index(w)
            
            if w == 'total_complexity':
                line3, = ax3.plot(np.arange(num_epochs), self.template_space, label='Train ' + metrics[i])
                line4, = ax4.plot(np.arange(num_epochs), self.template_space, label='Val ' + metrics[i])

                self.lines3.append(line3)
                self.lines4.append(line4)
            elif w == 'complexity_parts':
                for j in range(num_cascades):
                    line3, = ax3.plot(np.arange(num_epochs), self.template_space, label='Train ' + metrics[i] + str(j + 1))
                    line4, = ax4.plot(np.arange(num_epochs), self.template_space, label='Val ' + metrics[i] + str(j + 1))

                    self.lines3.append(line3)
                    self.lines4.append(line4)
            else:
                line1, = ax1.plot(np.arange(num_epochs), self.template_space, label='Train ' + metrics[i])
                line2, = ax2.plot(np.arange(num_epochs), self.template_space, label='Val ' + metrics[i])

                self.lines1.append(line1)
                self.lines2.append(line2)
    
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        ax1.set_ylim(0, 1.01)
        ax2.set_ylim(0, 1.01)

        self.fig1.show()
        self.fig2.show()
        
        if 'total_complexity' in watches or 'complexity_parts' in watches:
            ax3.legend(loc='best')
            ax4.legend(loc='best')
            ax3.set_ylim(0, 1000)
            ax4.set_ylim(0, 1000)

            self.fig3.show()
            self.fig4.show()
        
    def watch(self, train_metrics, val_metrics):
        for j, v in enumerate(self.watches):
            i = self.metrics.index(v)
                       
            if v == 'total_complexity':
                template_y3 = self.template_space.copy()
                template_y4 = self.template_space.copy()
                
                template_y3[:len(train_metrics[i])] = train_metrics[i]
                template_y4[:len(val_metrics[i])] = val_metrics[i]

                self.lines3[j].set_ydata(template_y3)
                self.lines4[j].set_ydata(template_y4)
            elif v == 'complexity_parts':
                for j in range(num_cascades):
                    template_y3 = self.template_space.copy()
                    template_y4 = self.template_space.copy()

                    template_y3[:len(train_metrics[i])] = train_metrics[i][j]
                    template_y4[:len(val_metrics[i])] = val_metrics[i][j]

                    self.lines3[j].set_ydata(template_y3)
                    self.lines4[j].set_ydata(template_y4)
            else:
                template_y1 = self.template_space.copy()
                template_y2 = self.template_space.copy()
                
                template_y1[:len(train_metrics[i])] = train_metrics[i]
                template_y2[:len(val_metrics[i])] = val_metrics[i]

                self.lines1[j].set_ydata(template_y1)
                self.lines2[j].set_ydata(template_y2)
        
        self.fig1.canvas.draw()       
        self.fig2.canvas.draw()
        
        if 'total_complexity' in self.watches or 'complexity_parts' in self.watches:
            self.fig3.canvas.draw()       
            self.fig4.canvas.draw()
        
    def finish(self):
        plt.close(self.fig1)
        plt.close(self.fig2)
        
        if 'total_complexity' in self.watches or 'complexity_parts' in self.watches:
            plt.close(self.fig3)
            plt.close(self.fig4)
        