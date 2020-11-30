import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
from data.datamgr import TransformLoader
from PIL import Image
import numpy as np
import torch.nn.functional as F
import utils
from abc import abstractmethod

class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way = True, image_size=224):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = -1 #(change depends on input) 
        self.feature    = model_func()
        self.feat_dim   = self.feature.final_feat_dim
        self.change_way = change_way  #some methods allow different_way classification during training and test
        # !!!!! Change if not using ResNet
        self.image_size = image_size
        self.trans_loader = TransformLoader(self.image_size)
        self.transform = self.trans_loader.get_composed_transform(aug=False)

    @abstractmethod
    def set_forward(self,x,is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self,x):
        out  = self.feature.forward(x)
        return out

    def parse_feature(self,x,is_feature):
        x    = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            if self.feature is not None:
                z_all       = self.feature.forward(x)
            else:
                z_all = self.pretrain.get_features(x)
            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
            self.z_all = z_all
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query

    def parse_images(self, image_paths):
        support_size = self.n_way * self.n_support
        query_size = self.n_way * self.n_query
        support_imgs = torch.zeros(support_size, 3, self.image_size, self.image_size)
        query_imgs = torch.zeros(query_size, 3, self.image_size, self.image_size)
        s_idx = 0
        q_idx = 0
        for cl in image_paths:
            n = len(cl)
            for i in range(n):
                img = Image.open(cl[i]).convert('RGB')
                img = self.transform(img)
                if i < self.n_support:
                    support_imgs[s_idx] = img
                    s_idx += 1
                else:
                    query_imgs[q_idx] = img
                    q_idx += 1
        return support_imgs.cuda(), query_imgs.cuda()

    def correct(self, x, metric="acc"):       
        scores = self.set_forward(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)

        if metric == "dacc":
            diff_score = np.array(self._evaluate_hardness_logodd(self.z_all))
            return ((topk_ind[:,0] == y_query) * diff_score).sum(), diff_score.sum()
        return float(top1_correct), len(y_query)

    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        return x_normalized

    def _evaluate_hardness_logodd(self, z_all):
        z_all = z_all.cuda()
        pretrain = self.pretrain
        num_classes = self.pretrain.num_classes
        n_shot = self.n_support
        n_way = self.n_way
        n_query = self.n_query
        probs = torch.zeros(z_all.shape[0], z_all.shape[1], num_classes).cuda()
        relu = torch.nn.ReLU()
        softmax = torch.nn.Softmax(dim=1)
        for i in range(n_way):
            probs[i] = pretrain.classify(z_all[i], normalize_prob=False)[:, : num_classes]
            probs[i] = relu(probs[i])
        # 1. Obtain 5*64 weights
        w = probs[:, :n_shot, :].mean(dim=1)
        w = self.normalize(w)
        # 2. Obtain query in 75 * 64 format
        query = probs[:, n_shot:, :]  # 5*15*64
        query = query.contiguous().view(n_way * n_query, -1)  # 75 * 64
        query = self.normalize(query)
        y_query = np.repeat(range(n_way), n_query)
        w = w.unsqueeze(0).expand(query.shape[0], -1, -1)
        query = query.unsqueeze(1).expand(-1, n_way, -1)
        logits = (w * query).sum(dim=2)  # 75 * 5
        query_probs = softmax(logits)
        hardness = []
        # correct = (query_probs.argmax(dim=1).cpu().numpy() == y_query).sum()
        # self.tmp_correct = correct
        for i in range(query_probs.shape[0]):
            p = query_probs[i][y_query[i]]
            log_odd = torch.log((1 - p) / p)
            hardness.append(log_odd.cpu().numpy())
        hardness = np.array(hardness)
        if hardness.min() < 0:
            hardness -= hardness.min()
        return hardness

    def calc_correct(self, scores):
        y_query = np.repeat(range( self.n_way ), self.n_query )
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer ):
        print_freq = 10

        avg_loss=0
        avg_acc = 0
        for i, (x,_ ) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss( x )
            loss.backward()
            optimizer.step()
            # avg_loss = avg_loss+loss.data[0]
            avg_loss = avg_loss+loss.data.item()
            correct_this, count_this = self.calc_correct(self.current_scores)
            avg_acc += correct_this / count_this * 100
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}, Acc {:f}'.format(epoch, i, len(train_loader), avg_loss / float(i + 1), avg_acc / float(i + 1)))

    def test_loop(self, test_loader, record = None, metric="acc"):
        correct =0
        count = 0
        print_freq = 10
        acc_all = []
        avg_acc = 0
        iter_num = len(test_loader)
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            correct_this, count_this = self.correct(x, metric=metric)
            acc_all.append(correct_this/ count_this*100  )
            avg_acc += correct_this / count_this * 100
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}, Acc {:f}'.format(0, i, len(test_loader), 0, avg_acc / float(i + 1)))
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        return acc_mean

    def set_forward_adaptation(self, x, is_feature = True): #further adaptation, default is fixing feature and train a new softmax clasifier
        assert is_feature == True, 'Feature is fixed in further adaptation'
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())

        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        
        batch_size = 4
        support_size = self.n_way* self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores
