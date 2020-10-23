from models.resnet_mtl import ResNetMtl
import torch
import torch.nn as nn
import os.path as osp
import tqdm
from dataloader.dataset_loader import DatasetLoader as Dataset
from torch.utils.data import DataLoader
from utils.misc import count_acc, ensure_path
from torch.nn.utils.weight_norm import WeightNorm
import numpy as np
import torch.nn.functional as F
from models.IFSL_modules import FeatureProcessor

class PretrainNet():
    def __init__(self, args):
        self.args = args
        log_base_dir = '/data2/yuezhongqi/Model/mtl/logs/'
        self.encoder = ResNetMtl(mtl=False)
        if self.args.init_weights is not None:
            pretrained_dict = torch.load(self.args.init_weights)['params']
        else:
            pre_base_dir = osp.join(log_base_dir, 'pre')
            pre_save_path1 = '_'.join([args.dataset, args.model_type])
            pre_save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(args.pre_gamma) + '_step' + \
                str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch)
            pre_save_path = pre_base_dir + '/' + pre_save_path1 + '_' + pre_save_path2
            pretrained_dict = torch.load(osp.join(pre_save_path, 'max_acc.pth'))['params']
        model_dict = self.encoder.state_dict()
        model_dict.update(pretrained_dict)
        self.encoder.load_state_dict(model_dict)
        self.encoder = self.encoder.cuda()
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def load_classifier(self, n_splits, epoch=22, num_classes=64):
        self.save_dir = osp.join("logs/clfs", self.args.dataset, str(n_splits))
        outfile = osp.join(self.save_dir, "%d.tar" % (epoch))
        in_dim = int(640 / n_splits)
        out_dim = num_classes
        self.clfs = nn.ModuleList([nn.Linear(in_dim, out_dim) for i in range(n_splits)])
        self.clfs = self.clfs.cuda()
        saved_states = torch.load(outfile)
        state_dict = self.clfs.state_dict()
        state_dict.update(saved_states)
        self.clfs.load_state_dict(state_dict)
        for param in self.clfs.parameters():
            param.requires_grad = False

    def train_classifier(self, n_splits, num_classes=64):
        self.save_dir = osp.join("logs/clfs", self.args.dataset, str(n_splits))
        ensure_path(self.save_dir)

        in_dim = int(640 / n_splits)
        out_dim = num_classes
        self.clfs = nn.ModuleList([nn.Linear(in_dim, out_dim) for i in range(n_splits)])
        self.clfs = self.clfs.cuda()
        optimizer = torch.optim.Adam(self.clfs.parameters())
        loss_fn = nn.CrossEntropyLoss()

        # Load pretrain set
        num_workers = 8
        if self.args.debug:
            num_workers = 0
        self.trainset = Dataset('train', self.args, train_aug=True)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=self.args.pre_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        
        for epoch in range(1, self.args.pre_max_epoch + 1):
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                optimizer.zero_grad()
                data, _ = [_.cuda() for _ in batch]
                label = batch[1]
                label = label.type(torch.cuda.LongTensor)
                with torch.no_grad():
                    logits = self.encoder(data)
                avg_loss = 0
                avg_acc = 0
                for j in range(n_splits):
                    start = in_dim * j
                    stop = start + in_dim
                    scores = self.clfs[j](logits[:, start:stop])
                    loss = loss_fn(scores, label)
                    loss.backward(retain_graph=True)
                    acc = count_acc(scores, label)
                    avg_loss += loss.item() / n_splits
                    avg_acc += acc / n_splits
                optimizer.step()
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, avg_loss, acc))
            if epoch % 2 == 0:
                outfile = osp.join(self.save_dir, "%d.tar" % (epoch))
                torch.save(self.clfs.state_dict(), outfile)
    
    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        return x_normalized

    def get_base_means(self, num_classes=64, is_cosine_feature=False):
        self.means_save_dir = osp.join("logs/means", "%s_%s.npy" % (self.args.dataset, str(is_cosine_feature)))
        if osp.isfile(self.means_save_dir):
            means = np.load(self.means_save_dir)
        else:
            means = self.save_base_means(num_classes, is_cosine_feature)
        return means

    def save_base_means(self, num_classes=64, is_cosine_feature=False):
        ensure_path("logs/means")
        self.means_save_dir = osp.join("logs/means", "%s_%s.npy" % (self.args.dataset, str(is_cosine_feature)))
        # Load pretrain set
        num_workers = 8
        if self.args.debug:
            num_workers = 0
        self.trainset = Dataset('train', self.args, train_aug=False)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=self.args.pre_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        means = torch.zeros(num_classes, 640).cuda()
        counts = torch.zeros(num_classes).cuda()
        for epoch in range(1):
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                data, _ = [_.cuda() for _ in batch]
                label = batch[1]
                with torch.no_grad():
                    data = self.encoder(data)
                if is_cosine_feature:
                    data = self.normalize(data)
                for j in range(data.shape[0]):
                    means[label[j]] += data[j]
                    counts[label[j]] += 1
        counts = counts.unsqueeze(1).expand_as(means)
        means = means / counts
        means_np = means.cpu().detach().numpy()
        np.save(self.means_save_dir, means_np)
        return means_np


class distLinear(nn.Module):
    def __init__(self, indim, outdim, class_wise_learnable_norm=True):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias=False)
        self.class_wise_learnable_norm = class_wise_learnable_norm   #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0)  #split the weight update component to direction and norm      

        if outdim <= 200:
            self.scale_factor = 2  # a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github 
        else:
            self.scale_factor = 10  #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor * (cos_dist) 

        return scores


class MultiLinearClassifier(nn.Module):
    def __init__(self, n_clf, feat_dim, n_way, sum_log=True, permute=False, shapes=None, loss_type="softmax"):
        super(MultiLinearClassifier, self).__init__()
        self.n_clf = n_clf
        self.feat_dim = feat_dim
        self.n_way = n_way
        self.sum_log = sum_log
        self.softmax = nn.Softmax(dim=2)
        self.permute = permute
        self.shapes = shapes
        if self.permute:
            self.clfs = nn.ModuleList([self.create_clf(loss_type, shapes[i], n_way).cuda() for i in range(n_clf)])
        else:
            self.clfs = nn.ModuleList([self.create_clf(loss_type, feat_dim, n_way).cuda() for i in range(n_clf)])

    def create_clf(self, loss_type, in_dim, out_dim):
        if loss_type == "softmax":
            return nn.Linear(in_dim, out_dim)
        elif loss_type == "dist":
            return distLinear(in_dim, out_dim, True)

    def forward(self, X):
        # X is n_clf * N * feat_dim
        if self.permute:
            N = X[0].shape[0]
        else:
            N = X.shape[1]
        resp = torch.zeros(self.n_clf, N, self.n_way).cuda()
        for i in range(self.n_clf):
            resp[i] = self.clfs[i](X[i])
        proba = self.softmax(resp)
        if self.sum_log:
            log_proba = torch.log(proba)
            sum_log_proba = log_proba.mean(dim=0)
            scores = sum_log_proba
        else:
            mean_proba = proba.mean(dim=0)
            log_proba = torch.log(mean_proba)
            scores = log_proba
        return scores


class MultiBiLinearClassifier(nn.Module):
    def __init__(self, n_clf, x_feat_dim, d_feat_dim, n_way, sum_log=True, loss_type="softmax", logit_fusion="linear_sum"):
        super(MultiBiLinearClassifier, self).__init__()
        self.n_clf = n_clf
        self.x_feat_dim = x_feat_dim
        self.d_feat_dim = d_feat_dim
        self.n_way = n_way
        self.sum_log = sum_log
        self.softmax = nn.Softmax(dim=2)
        self.logit_fusion = logit_fusion
        self.x_clfs = nn.ModuleList([self.create_clf(loss_type, x_feat_dim, n_way).cuda() for i in range(n_clf)])
        self.d_clfs = nn.ModuleList([self.create_clf(loss_type, d_feat_dim, n_way).cuda() for i in range(n_clf)])

    def fuse_logits(self, p1, p2):
        sigmoid = torch.nn.Sigmoid()
        if self.logit_fusion == "linear_sum":
            return p1 + p2
        elif self.logit_fusion == "product":
            return torch.log(sigmoid(p1) * sigmoid(p2))
        elif self.logit_fusion == "sum":
            return torch.log(sigmoid(p1 + p2))
        elif self.logit_fusion == "harmonic":
            p = sigmoid(p1) * sigmoid(p2)
            return torch.log(p / (1 + p))

    def create_clf(self, loss_type, in_dim, out_dim):
        if loss_type == "softmax":
            return nn.Linear(in_dim, out_dim)
        elif loss_type == "dist":
            return distLinear(in_dim, out_dim, True)

    def forward(self, X, D, counterfactual=False):
        # X is n_clf * N * feat_dim
        N = X.shape[1]
        resp = torch.zeros(self.n_clf, N, self.n_way).cuda()
        for i in range(self.n_clf):
            d_logit = self.d_clfs[i](D[i])
            if counterfactual:
                x_logit = torch.ones_like(d_logit).cuda()
            else:
                x_logit = self.x_clfs[i](X[i])
            resp[i] = self.fuse_logits(x_logit, d_logit)
        proba = self.softmax(resp)
        if self.sum_log:
            log_proba = torch.log(proba)
            sum_log_proba = log_proba.mean(dim=0)
            scores = sum_log_proba
        else:
            mean_proba = proba.mean(dim=0)
            log_proba = torch.log(mean_proba)
            scores = log_proba
        return scores


class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.args.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.args.way))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars

    def initialize(self):
        torch.nn.init.kaiming_normal_(self.fc1_w)
        torch.nn.init.constant_(self.fc1_b, 0)


class DeconfoundedLearner():
    def __init__(self, pretrain, classifier="bi", logit_fusion="product", fusion="concat", n_splits=10, n_way=5,
                 sum_log=True, lr=0.005, weight_decay=0.001, d_feature="pd", n_steps=100, batch_size=4,
                 ori_embedding_for_pd=True, is_cosine_feature=False, use_counterfactual=False,
                 x_zero=False, preprocess_before_split="none", preprocess_after_split="none",
                 normalize_before_center=False, normalize_d=False, normalize_ed=False, use_x_only=False):
        self.pretrain = pretrain
        self.classifier = classifier
        self.logit_fusion = logit_fusion
        self.fusion = fusion
        self.n_splits = n_splits
        self.n_way = n_way
        self.sum_log = sum_log
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.d_feature = d_feature
        self.ori_embedding_for_pd = ori_embedding_for_pd
        self.num_classes = pretrain.num_classes
        num_classes = self.num_classes
        self.is_cosine_feature = is_cosine_feature
        self.use_counterfactual = use_counterfactual
        self.x_zero = x_zero
        self.use_x_only = use_x_only
        '''
        if n_splits > 1:
            pretrain.load_classifier(n_splits, 62, num_classes)
        else:
            pretrain.load_classifier(n_splits, 32, num_classes)
        '''
        self.pretrain_features = pretrain.get_pretrained_class_mean(is_cosine_feature)
        self.pretrain_features = torch.from_numpy(self.pretrain_features).cuda().float()[:num_classes]
        
        self.feature_processor = FeatureProcessor(self.pretrain, self.n_splits, self.is_cosine_feature, self.d_feature, self.num_classes,
                                                  preprocess_after_split=preprocess_after_split, preprocess_before_split=preprocess_before_split,
                                                  normalize_before_center=normalize_before_center, normalize_d=normalize_d, normalize_ed=normalize_ed)

        total_feat_dim = self.pretrain.feat_dim
        x_feat_dim = int(total_feat_dim / n_splits)
        if self.d_feature == "pd":
            d_feat_dim = num_classes
        elif self.d_feature == "ed":
            d_feat_dim = x_feat_dim
        self.x_feat_dim = x_feat_dim
        self.d_feat_dim = d_feat_dim

        if n_splits >= 1:
            if self.classifier == "bi":
                self.clf = MultiBiLinearClassifier(n_splits, x_feat_dim, d_feat_dim, n_way, sum_log, "softmax", logit_fusion).cuda()
            elif self.classifier == "single":
                if fusion == "concat" and not self.use_x_only:
                    feat_dim = x_feat_dim + d_feat_dim
                else:
                    feat_dim = x_feat_dim
                self.clf = MultiLinearClassifier(n_splits, feat_dim, n_way, sum_log, False, None, "softmax").cuda()
        else:
            if self.classifier == "bi":
                self.x_clf = BaseLearner(self.pretrain.args, x_feat_dim).cuda()
                self.d_clf = BaseLearner(self.pretrain.args, d_feat_dim).cuda()
            elif self.classifier == "single":
                if fusion == "concat":
                    feat_dim = x_feat_dim + d_feat_dim
                else:
                    feat_dim = x_feat_dim
                self.clf = BaseLearner(self.pretrain.args, feat_dim).cuda()
        self.nll = nn.NLLLoss().cuda()

    def calc_pd(self, x, clf_idx):
        proba = self.pretrain.classify(x)
        return proba
    
    def get_pd_features(self, x):
        feat_dim = self.x_feat_dim
        fpd = torch.zeros(self.n_splits, x.shape[0], self.num_classes).cuda()
        pd = self.calc_pd(x, 0)
        for i in range(self.n_splits):
            start = i * feat_dim
            stop = start + feat_dim
            fpd[i] = pd
        return fpd

    def get_ed_features(self, x):
        feat_dim = self.d_feat_dim
        ed = torch.zeros(self.n_splits, x.shape[0], feat_dim).cuda()
        pd = self.calc_pd(x, 0)
        for i in range(self.n_splits):
            start = i * feat_dim
            stop = start + feat_dim
            ed[i] = torch.mm(pd, self.pretrain_features)[:, start:stop]
        return ed

    def get_split_features(self, x):
        # Sequentially cut into n_splits parts
        split_dim = self.x_feat_dim
        split_features = torch.zeros(self.n_splits, x.shape[0], split_dim).cuda()
        for i in range(self.n_splits):
            start_idx = split_dim * i
            end_idx = split_dim * i + split_dim
            split_features[i] = x[:, start_idx:end_idx]
        return split_features

    def fuse_feature(self, a, b, dim=2):
        if self.fusion == "concat":
            return torch.cat((a, b), dim=dim)
        elif self.fusion == "+":
            return a + b
        elif self.fusion == "-":
            return a - b

    def fuse_logits(self, p1, p2):
        sigmoid = torch.nn.Sigmoid()
        if self.logit_fusion == "linear_sum":
            return p1 + p2
        elif self.logit_fusion == "product":
            return torch.log(sigmoid(p1) * sigmoid(p2))
        elif self.logit_fusion == "sum":
            return torch.log(sigmoid(p1 + p2))
        elif self.logit_fusion == "harmonic":
            p = sigmoid(p1) * sigmoid(p2)
            return torch.log(p / (1 + p))

    def backward_loss_and_step(self, loss, optimizer):
        grad = torch.autograd.grad(loss, list(self.clf.parameters()))
        for i, param in enumerate(list(self.clf.parameters())):
            param.grad = grad[i]
        optimizer.step()
        '''
        grad = torch.autograd.grad(loss, list(self.clf.parameters()))
        fast_weights = list(map(lambda p: p[1] - self.lr * p[0], zip(grad, list(self.clf.parameters()))))
        for i, param in enumerate(self.clf.parameters()):
            param.data = fast_weights[i]  # Not working
        '''

    def fit(self, support, query, support_labels, support_embedding, query_embedding):
        if self.n_splits >= 1:
            return self.fit_multi_splits(support, query, support_labels, support_embedding, query_embedding)
        else:
            return self.fit_no_split(support, query, support_labels, support_embedding, query_embedding)
    
    def fit_no_split(self, support, query, support_labels, support_embedding, query_embedding):
        if self.ori_embedding_for_pd:
            support_ori_embedding = self.pretrain.get_features(support)
            query_ori_embedding = self.pretrain.get_features(query)
        if self.d_feature == "pd":
            if self.ori_embedding_for_pd:
                support_d = self.get_pd_features(support_ori_embedding)[0]
                query_d = self.get_pd_features(query_ori_embedding)[0]
            else:
                support_d = self.get_pd_features(support_embedding)[0]
                query_d = self.get_pd_features(query_embedding)[0]
        elif self.d_feature == "ed":
            if self.ori_embedding_for_pd:
                support_d = self.get_ed_features(support_ori_embedding)[0]
                query_d = self.get_ed_features(query_ori_embedding)[0]
            else:
                support_d = self.get_ed_features(support_embedding)[0]
                query_d = self.get_ed_features(query_embedding)[0]
        self.fast_weight_x = None
        self.fast_weight_d = None
        for _ in range(self.n_steps):
            self.no_split_update(support_embedding, support_d, support_labels)
        return self.calc_no_split_logit(query_embedding, query_d, self.fast_weight_x, self.fast_weight_d)

    def no_split_update(self, x, d, label):
        logits = self.calc_no_split_logit(x, d, self.fast_weight_x, self.fast_weight_d)
        loss = F.cross_entropy(logits, label)
        # print(loss.item())
        if self.classifier == "bi":
            x_grad = torch.autograd.grad(loss, self.x_clf.parameters(), retain_graph=True)
            d_grad = torch.autograd.grad(loss, self.d_clf.parameters())
            if self.fast_weight_x is None:
                self.fast_weight_x = list(map(lambda p: p[1] - self.lr * p[0], zip(x_grad, self.x_clf.parameters())))
                self.fast_weight_d = list(map(lambda p: p[1] - self.lr * p[0], zip(d_grad, self.d_clf.parameters())))
            else:
                self.fast_weight_x = list(map(lambda p: p[1] - self.lr * p[0], zip(x_grad, self.fast_weight_x)))
                self.fast_weight_d = list(map(lambda p: p[1] - self.lr * p[0], zip(d_grad, self.fast_weight_d)))
        elif self.classifier == "single":
            grad = torch.autograd.grad(loss, self.clf.parameters())
            if self.fast_weight_x is None:
                self.fast_weight_x = list(map(lambda p: p[1] - self.lr * p[0], zip(grad, self.clf.parameters())))
            else:
                self.fast_weight_x = list(map(lambda p: p[1] - self.lr * p[0], zip(grad, self.fast_weight_x)))

    def calc_no_split_logit(self, x, d, fast_weight_x=None, fast_weight_d=None):
        if self.classifier == "bi":
            x_logits = self.x_clf(x, fast_weight_x)
            d_logits = self.d_clf(d, fast_weight_d)
            return self.fuse_logits(x_logits, d_logits)
        elif self.classifier == "single":
            fused_feat = self.fuse_feature(x, d, dim=1)
            logits = self.clf(fused_feat, fast_weight_x)
            return logits

    def fit_multi_splits(self, support, query, support_labels, support_embedding, query_embedding):
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        support_size = support_embedding.shape[0]

        if self.ori_embedding_for_pd:
            support_ori_embedding = self.pretrain.get_features(support)
            query_ori_embedding = self.pretrain.get_features(query)
        else:
            support_ori_embedding = None
            query_ori_embedding = None
        split_support, support_d, split_query, query_d = self.feature_processor.get_features(support_embedding,
                            query_embedding, support_ori_embedding, query_ori_embedding)
        '''
        split_support = self.get_split_features(support_embedding)
        split_query = self.get_split_features(query_embedding)
        if self.d_feature == "pd":
            if self.ori_embedding_for_pd:
                support_d = self.get_pd_features(support_ori_embedding)
                query_d = self.get_pd_features(query_ori_embedding)
            else:
                support_d = self.get_pd_features(support_embedding)
                query_d = self.get_pd_features(query_embedding)
        elif self.d_feature == "ed":
            if self.ori_embedding_for_pd:
                support_d = self.get_ed_features(support_ori_embedding)
                query_d = self.get_ed_features(query_ori_embedding)
            else:
                support_d = self.get_ed_features(support_embedding)
                query_d = self.get_ed_features(query_embedding)
        '''

        if self.classifier == "single" and not self.use_x_only:
            fused_support = self.fuse_feature(split_support, support_d)
            fused_query = self.fuse_feature(split_query, query_d)
        else:
            fused_support = split_support
            fused_query = split_query

        for epoch in range(self.n_steps):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, self.batch_size):
                selected_id = torch.from_numpy(rand_id[i: min(i + self.batch_size, support_size)]).cuda()
                y_batch = support_labels[selected_id]
                if self.classifier == "bi":
                    logits = self.clf(split_support[:, selected_id, :], support_d[:, selected_id, :])
                elif self.classifier == "single":
                    logits = self.clf(fused_support[:, selected_id, :])
                loss = self.nll(logits, y_batch)
                # print(loss.item())
                self.backward_loss_and_step(loss, optimizer)

        for param in list(self.clf.parameters()):
            param.grad = None

        if self.classifier == "bi":
            logits = self.clf(split_query, query_d)
        elif self.classifier == "single":
            logits = self.clf(fused_query)
            if self.use_counterfactual:
                if self.x_zero:
                    c_split_query_x = torch.zeros(split_query.shape).cuda()
                else:
                    c_split_query_x = split_support.mean(dim=1).unsqueeze(1).expand(split_query.shape)
                c_fused_query = self.fuse_feature(c_split_query_x, query_d)
                c_scores = self.clf(c_fused_query)
                logits = logits - c_scores
        return logits

    def predict(self, support_labels, support_embedding, query_embedding):
        support_ori_embedding = None
        query_ori_embedding = None
        split_support, support_d, split_query, query_d = self.feature_processor.get_features(support_embedding,
                            query_embedding, support_ori_embedding, query_ori_embedding)
        if self.classifier == "single" and not self.use_x_only:
            fused_support = self.fuse_feature(split_support, support_d)
            fused_query = self.fuse_feature(split_query, query_d)
        else:
            fused_support = split_support
            fused_query = split_query
        logits = self.clf(fused_query)
        return logits