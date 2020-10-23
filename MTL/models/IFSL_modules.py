import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight 
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None  #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class IFSLBaseLearner(nn.Module):
    def __init__(self, feat_dim, n_way, update_step, approx=True, lr=0.01):
        super(IFSLBaseLearner, self).__init__()
        self.feat_dim = feat_dim
        self.n_way = n_way
        self.update_step = update_step
        self.approx = approx
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = Linear_fw(self.feat_dim, self.n_way)
        self.classifier.bias.data.fill_(0)

    def forward(self, x):
        scores = self.classifier.forward(x)
        return scores

    def fit(self, support, labels):
        fast_parameters = list(self.parameters())
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()
        for step in range(self.update_step):
            scores = self.forward(support)
            loss = self.loss_fn(scores, labels)
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True, allow_unused=True)
            if self.approx:
                for g in grad:
                    if g is not None:
                        g = g.detach()
            fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                if weight.fast is None:
                    if grad[k] is None:
                        weight.fast = None
                    else:
                        weight.fast = weight - self.lr * grad[k]
                else:
                    if grad[k] is None:
                        weight.fast = weight.fast
                    else:
                        weight.fast = weight.fast - self.lr * grad[k]
                fast_parameters.append(weight.fast)

    def predict(self, query):
        return self.forward(query)


class FeatureProcessor():
    def __init__(self, pretrain, n_splits, is_cosine_feature=False, d_feature="ed", num_classes=64,
                 preprocess_after_split="none", preprocess_before_split="none", normalize_before_center=False,
                 normalize_d=False, normalize_ed=False):
        super(FeatureProcessor, self).__init__()
        self.pretrain = pretrain
        self.feat_dim = self.pretrain.feat_dim
        self.n_splits = n_splits
        self.num_classes = 64
        self.is_cosine_feature = is_cosine_feature
        self.d_feature = d_feature
        self.preprocess_after_split = preprocess_after_split
        self.preprocess_before_split = preprocess_before_split
        self.normalize_before_center = normalize_before_center
        self.normalize_d = normalize_d
        self.normalize_ed = normalize_ed
        # print(self.preprocess_before_split, self.preprocess_after_split, self.normalize_before_center,
              # self.normalize_d, self.normalize_ed)

        pretrain_features = self.pretrain.get_pretrained_class_mean(normalize=is_cosine_feature)
        self.pretrain_features = torch.from_numpy(pretrain_features).float().cuda()[:num_classes]
        if normalize_d:
            self.pretrain_features = self.normalize(self.pretrain_features)
        self.pretrain_features_mean = self.pretrain_features.mean(dim=0)
        # if self.n_splits > 1:
            # self.pretrain.load_d_specific_classifiers(n_splits)

    def get_split_features(self, x, preprocess=False, center=None, preprocess_method="l2n"):
        # Sequentially cut into n_splits parts
        split_dim = int(self.feat_dim / self.n_splits)
        split_features = torch.zeros(self.n_splits, x.shape[0], split_dim).cuda()
        for i in range(self.n_splits):
            start_idx = split_dim * i
            end_idx = split_dim * i + split_dim
            if preprocess_method == "l2n":
                split_features[i] = self.normalize(x[:, start_idx:end_idx])
            elif preprocess_method == "none":
                split_features[i] = x[:, start_idx:end_idx]
            elif preprocess_method == "cl2n":
                split_features[i] = self.normalize(x[:, start_idx:end_idx] - center[:, start_idx:end_idx])
            '''
            if preprocess:
                if preprocess_method != "dl2n":
                    split_features[i] = self.nn_preprocess(split_features[i], center[:, start_idx:end_idx], preprocessing=preprocess_method)
                else:
                    if self.normalize_before_center:
                        split_features[i] = self.normalize(split_features[i])
                    centered_data = split_features[i] - center[i]
                    split_features[i] = self.normalize(centered_data)
            '''
        return split_features

    def nn_preprocess(self, data, center=None, preprocessing="l2n"):
        if preprocessing == "none":
            return data
        elif preprocessing == "l2n":
            return self.normalize(data)
        elif preprocessing == "cl2n":
            if self.normalize_before_center:
                data = self.normalize(data)
            centered_data = data - center
            return self.normalize(centered_data)

    def calc_pd(self, x, clf_idx):
        proba = self.pretrain.classify(x)
        return proba

    def normalize(self, x, dim=1):
        x_norm = torch.norm(x, p=2, dim=dim).unsqueeze(dim).expand_as(x)
        x_normalized = x.div(x_norm)
        return x_normalized

    def get_d_feature(self, x, x_ori):
        feat_dim = int(self.feat_dim / self.n_splits)
        if self.d_feature == "ed":
            d_feat_dim = int(self.feat_dim / self.n_splits)
        else:
            d_feat_dim = self.num_classes
        d_feature = torch.zeros(self.n_splits, x.shape[0], d_feat_dim).cuda()
        if x_ori is None:
            pd = self.calc_pd(x, 0)
        else:
            pd = self.calc_pd(x_ori, 0)
        for i in range(self.n_splits):
            start = i * feat_dim
            stop = start + feat_dim
            if self.d_feature == "pd":
                d_feature[i] = pd
            else:
                d_feature[i] = torch.mm(pd, self.pretrain_features)[:, start:stop]
        return d_feature

    def get_features(self, support, query, support_ori, query_ori):
        support_d = self.get_d_feature(support, support_ori)
        query_d = self.get_d_feature(query, query_ori)
        if self.normalize_ed:
            support_d = self.normalize(support_d, dim=2)
            query_d = self.normalize(query_d, dim=2)
        support_size = support.shape[0]
        query_size = query.shape[0]
        pmean_support = self.pretrain_features_mean.expand((support_size, self.feat_dim))
        pmean_query = self.pretrain_features_mean.expand((query_size, self.feat_dim))
        support = self.nn_preprocess(support, pmean_support, preprocessing=self.preprocess_before_split)
        query = self.nn_preprocess(query, pmean_query, preprocessing=self.preprocess_before_split)
        split_support = self.get_split_features(support, preprocess=True, center=pmean_support,
                                                preprocess_method=self.preprocess_after_split)
        split_query = self.get_split_features(query, preprocess=True, center=pmean_query,
                                              preprocess_method=self.preprocess_after_split)
        return split_support, support_d, split_query, query_d