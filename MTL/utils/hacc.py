import numpy as np


class Hacc():
    def __init__(self, splits=10, topk=10):
        self.hardness = []
        self.correct_prediction = []
        self.splits = splits
        self.topk = topk

    def add_data(self, hardness, correct_prediction):
        self.hardness.append(hardness)
        self.correct_prediction.append(correct_prediction)

    def get_splits_hacc(self):
        hardness = np.concatenate(np.array(self.hardness))
        correct = np.concatenate(np.array(self.correct_prediction))
        nq_per_split = int(len(hardness) / self.splits)
        hardness_sorted = hardness.argsort()
        sum_of_inverse_acc = 0
        for i in range(self.splits):
            start = nq_per_split * i
            end = start + nq_per_split
            idx = hardness_sorted[start:end][::-1]
            current_correct_score = correct[idx]
            current_acc = np.mean(current_correct_score)
            sum_of_inverse_acc += (1 / current_acc)
        harmonic_acc = self.splits / sum_of_inverse_acc
        return harmonic_acc * 100

    def get_topk_hacc(self):
        hardness = np.concatenate(np.array(self.hardness))
        correct = np.concatenate(np.array(self.correct_prediction))
        top_k = int(len(hardness) / self.topk)
        hard_idx = hardness.argsort()[-top_k:][::-1]
        easy_idx = hardness.argsort()[:top_k][::-1]
        hard_acc = np.mean(correct[hard_idx])
        easy_acc = np.mean(correct[easy_idx])
        harmonic_acc = 2 * (hard_acc * easy_acc) / (hard_acc + easy_acc)
        return harmonic_acc * 100

    def get_topk_hard_acc(self):
        hardness = np.concatenate(np.array(self.hardness))
        correct = np.concatenate(np.array(self.correct_prediction))
        top_k = int(len(hardness) / self.topk)
        hard_idx = hardness.argsort()[-top_k:][::-1]
        hard_acc = np.mean(correct[hard_idx])
        return hard_acc

    def get_acc_in_range(self, start, end):
        hardness = np.concatenate(np.array(self.hardness))
        correct = np.concatenate(np.array(self.correct_prediction))
        idx = np.logical_and((hardness>=start), (hardness < end))
        if (idx.sum() == 0):
            return 0
        
        acc = correct[idx].sum() / idx.sum()
        return acc

    def get_plot(self, splits):
        hardness = np.concatenate(np.array(self.hardness))
        correct = np.concatenate(np.array(self.correct_prediction))
        nq_per_split = int(len(hardness) / splits)
        hardness_sorted = hardness.argsort()
        bins = np.arange(0, 1, 1 / splits)
        accs = []
        for i in range(splits):
            start = nq_per_split * i
            end = start + nq_per_split
            idx = hardness_sorted[start:end][::-1]
            current_correct_score = correct[idx]
            current_acc = np.mean(current_correct_score)
            accs.append(current_acc)
        return bins, np.array(accs)