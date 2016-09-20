class Viewer():
    def __init__(self,
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
        
        fig1, ax1 = plt.subplots(1, 1)
        fig2, ax2 = plt.subplots(1, 1)
        ax1.set_xlabel('Num Epoch')
        ax1.set_ylabel('Metric Value')
        ax1.set_xlabel('Num Epoch')
        ax1.set_ylabel('Metric Value')
        ax2.set_xlabel('Num Epoch')
        ax2.set_ylabel('Metric Value')
        lines1 = []
        lines2 = []
        
        if 'complexity' in watches:
            fig3, ax3 = plt.subplots(1, 1)
            fig4, ax4 = plt.subplots(1, 1)
            ax3.set_xlabel('Num Epoch')
            ax3.set_ylabel('Train Complexity')
            ax4.set_xlabel('Num Epoch')
            ax4.set_ylabel('Val Complexity')
            lines3 = []
            lines4 = []
    
        self.template_space = np.zeros(num_epochs)
        
        for w in watches:
            i = metrics.index(w)
            
            if w == 'complexity':
                line3, = ax3.plot(np.arange(num_epochs), template_space, label='Train ' + metrics[i])
                line4, = ax4.plot(np.arange(num_epochs), template_space, label='Val ' + metrics[i])

                lines3.append(line3)
                lines4.append(line4)
            else:
                line1, = ax1.plot(np.arange(num_epochs), template_space, label='Train ' + metrics[i])
                line2, = ax2.plot(np.arange(num_epochs), template_space, label='Val ' + metrics[i])

                lines1.append(line1)
                lines2.append(line2)
    
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        ax1.set_ylim(0, 1.01)
        ax2.set_ylim(0, 1.01)

        fig1.show()
        fig2.show()
        
        if 'complexity' in watches:
            ax3.legend(loc='best')
            ax4.legend(loc='best')
            ax3.set_ylim(0, 1000)
            ax4.set_ylim(0, 1000)

            fig3.show()
            fig4.show()
        
    def watch(self, train_metrics, val_metrics):
        for j, v in enumerate(self.watches):
            i = metrics.index(v)
                       
            if v == 'complexity':
                pass
            else:
                template_y1 = self.template_space.copy()
                template_y2 = self.template_space.copy()
                
                template_y1[:len(train_metrics[i])] = train_metrics[i]
                template_y2[:len(val_metrics[i])] = val_metrics[i]

                lines1[j].set_ydata(template_y1)
                lines2[j].set_ydata(template_y2)
        
        fig1.canvas.draw()       
        fig2.canvas.draw()
        
        if 'complexity' in self.watches:
            fig3.canvas.draw()       
            fig4.canvas.draw()
        
    def finish(self):
        plt.close(fig1)
        plt.close(fig2)
        
        if 'complexity' in self.watches:
            plt.close(fig3)
            plt.close(fig4)
        