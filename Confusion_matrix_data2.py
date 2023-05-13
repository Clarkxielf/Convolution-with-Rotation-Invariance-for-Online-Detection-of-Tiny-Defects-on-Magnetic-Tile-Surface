import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

# label_dict = {0:'MT_Blowhole', 1:'MT_Break', 2:'MT_Crack', 3:'MT_Fray', 4:'MT_Free', 5:'MT_Uneven'}
label_dict = {0:'MT_Blowhole', 1:'MT_Crack', 2:'MT_Free'}

class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("The model accuracy is ", acc)

        # calculate kappa
        p0 = 0
        pe = 0
        for i in range(self.num_classes):
            p0 += self.matrix[i, i]/self.matrix.sum()
            pe += self.matrix[i, :].sum()*self.matrix[:, i].sum()/pow(self.matrix.sum(), 2)
        K = (p0-pe)/(1-pe)

        print("The model kappa is ", K)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity", 'F1-score']
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3)
            Recall = round(TP / (TP + FN), 3)
            Specificity = round(TN / (TN + FP), 3)
            F1_sorce = round((2*Precision*Recall)/(Precision+Recall), 3)
            table.add_row([self.labels[i], Precision, Recall, Specificity, F1_sorce])
        print(table)

    def plot(self, save_path):
        matrix = self.matrix
        print('ConfusionMatrix:\n', matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=30)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels, rotation=30)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.show()