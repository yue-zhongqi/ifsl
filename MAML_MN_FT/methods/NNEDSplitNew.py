import backbone
import torch
from torch.autograd import Variable
import numpy as np
from methods.meta_template import MetaTemplate
from io_utils import print_with_carriage_return, end_carriage_return_print


class NNEDSplitNew(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, n_query, pretrain, n_steps=200, batch_size=4, num_classes=64,
                 reforward=False, classifier="bi", measure="linear", normalize_d=False, normalize_ed=False,
                 split="seq", n_splits=16, preprocess_after_split="l2n", preprocess_before_split="l2n",
                 is_cosine_feature=False, normalize_before_center=False, x_zero=False, process_cx=False,
                 image_size=224, use_counterfactual=False, proba_fusion="linear_sum", fusion="concat"):
        super(NNEDSplitNew, self).__init__(model_func, n_way, n_support, image_size=image_size)
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_query = n_query
        self.pretrain = pretrain
        self.reforward = reforward
        self.classifier = classifier
        self.measure = measure
        self.is_cosine_feature = is_cosine_feature
        self.normalize_before_center = normalize_before_center
        self.num_classes = num_classes
        pretrain_features = pretrain.get_pretrained_class_mean(normalize=is_cosine_feature)
        self.pretrain_features = torch.from_numpy(pretrain_features).float().cuda()[:self.num_classes]
        if normalize_d:
            self.pretrain_features = self.normalize(self.pretrain_features)
        self.pretrain_features_mean = self.pretrain_features.mean(dim=0)
        # self.pretrain.load_d_specific_classifiers(n_splits)

        self.use_counterfactual = use_counterfactual
        self.proba_fusion = proba_fusion
        self.fusion = fusion
        self.x_zero = x_zero
        self.process_cx = process_cx

        # Split
        self.split = split
        self.n_splits = n_splits
        self.preprocess_after_split = preprocess_after_split
        self.preprocess_before_split = preprocess_before_split
        self.normalize_ed = normalize_ed

    def set_forward(self, x, is_feature=True):
        return self.set_forward_adaptation(x, is_feature)  # Baseline always do adaptation

    def calc_pd(self, x, clf_idx):
        '''
        softmax = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            logits = self.pretrain.clfs[clf_idx](x)
            logits = logits
            proba = softmax(logits)
        '''
        with torch.no_grad():
            proba = self.pretrain.classify(x)
        return proba[:, :self.num_classes]

    def calc_ed(self, x):
        feat_dim = int(self.feat_dim / self.n_splits)
        ed = torch.zeros(self.n_splits, x.shape[0], feat_dim).cuda()
        for i in range(self.n_splits):
            start = i * feat_dim
            stop = start + feat_dim
            # pd = self.calc_pd(x[:, start:stop], i)
            pd = self.calc_pd(x, i)
            ed[i] = torch.mm(pd, self.pretrain_features)[:, start:stop]
        return ed
    
    def temp(self, x):
        feat_dim = int(self.feat_dim / self.n_splits)
        ed = torch.zeros(self.n_splits, x.shape[1], feat_dim).cuda()
        for i in range(self.n_splits):
            start = i * feat_dim
            stop = start + feat_dim
            ed[i] = x[i][:, start:stop]
        return ed

    def normalize(self, x, dim=1):
        x_norm = torch.norm(x, p=2, dim=dim).unsqueeze(dim).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        return x_normalized

    def fuse_proba(self, p1, p2):
        sigmoid = torch.nn.Sigmoid()
        if self.proba_fusion == "linear_sum":
            return p1 + p2
        elif self.proba_fusion == "product":
            return torch.log(sigmoid(p1) * sigmoid(p2))
        elif self.proba_fusion == "sum":
            return torch.log(sigmoid(p1 + p2))
        elif self.proba_fusion == "harmonic":
            p = sigmoid(p1) * sigmoid(p2)
            return torch.log(p / (1 + p))

    def fuse_features(self, x1, x2):
        if self.fusion == "concat":
            return torch.cat((x1, x2), dim=2)
        elif self.fusion == "+":
            return x1 + x2
        elif self.fusion == "-":
            return x1 - x2

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

    def get_split_features(self, x, preprocess=False, center=None, preprocess_method="l2n"):
        if self.split == "seq":
            # Sequentially cut into n_splits parts
            split_dim = int(self.feat_dim / self.n_splits)
            split_features = torch.zeros(self.n_splits, x.shape[0], split_dim).cuda()
            for i in range(self.n_splits):
                start_idx = split_dim * i
                end_idx = split_dim * i + split_dim
                split_features[i] = x[:, start_idx:end_idx]
                if preprocess:
                    if preprocess_method != "dl2n":
                        split_features[i] = self.nn_preprocess(split_features[i], center[:, start_idx:end_idx], preprocessing=preprocess_method)
                    else:
                        if self.normalize_before_center:
                            split_features[i] = self.normalize(split_features[i])
                        centered_data = split_features[i] - center[i]
                        split_features[i] = self.normalize(centered_data)
            return split_features

    def set_forward_adaptation(self, x, image_paths=None, is_feature=True):
        # Feature
        assert is_feature == True, 'Baseline only support testing with feature'
        support, query = self.parse_feature(x, is_feature)

        support = support.contiguous().view(self.n_way * self.n_support, -1 )
        query = query.contiguous().view(self.n_way * self.n_query, -1 )

        if self.reforward:
            support_imgs, query_imgs = self.parse_images(image_paths)
            support = self.pretrain.get_features(support_imgs)
            query = self.pretrain.get_features(query_imgs)

        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support ))
        y_support = Variable(y_support.cuda())

        support_size = self.n_way * self.n_support
        query_size = self.n_way * self.n_query

        support_ed = self.calc_ed(support)
        query_ed = self.calc_ed(query)
        
        if self.normalize_ed:
            support_ed = self.normalize(support_ed, dim=2)
            query_ed = self.normalize(query_ed, dim=2)

        pmean_support = self.pretrain_features_mean.expand((support_size, self.feat_dim))
        pmean_query = self.pretrain_features_mean.expand((query_size, self.feat_dim))
        self.support_center = pmean_support
        self.query_center = pmean_query
        if self.preprocess_before_split == "dl2n" or self.preprocess_after_split == "dl2n":
            self.support_center = support_ed
            self.query_center = query_ed

        if self.classifier == "bi":
            # Never dl2n before split
            support = self.nn_preprocess(support, pmean_support, preprocessing=self.preprocess_before_split)
            query = self.nn_preprocess(query, pmean_query, preprocessing=self.preprocess_before_split)

            split_support_x = self.get_split_features(support, preprocess=True, center=self.support_center, preprocess_method=self.preprocess_after_split)
            split_query_x = self.get_split_features(query, preprocess=True, center=self.query_center, preprocess_method=self.preprocess_after_split)

            nn_clf = backbone.MultiNNClassifier(self.n_way, self.n_splits, measure=self.measure)
            nn_clf.fit(split_support_x, y_support.cpu().numpy())
            x_scores = nn_clf.predict(split_query_x)

            ed_clf = backbone.MultiNNClassifier(self.n_way, self.n_splits, measure=self.measure, temp=1)
            ed_clf.fit(support_ed, y_support.cpu().numpy())
            ed_scores = ed_clf.predict(query_ed)

            scores = self.fuse_proba(x_scores, ed_scores)
            if self.use_counterfactual:
                c_x_scores = torch.ones(x_scores.shape).cuda() / self.n_way
                c_scores = self.fuse_proba(c_x_scores, ed_scores)
                scores = scores - c_scores
            return scores
        elif self.classifier == "single":
            support = self.nn_preprocess(support, pmean_support, preprocessing=self.preprocess_before_split)
            query = self.nn_preprocess(query, pmean_query, preprocessing=self.preprocess_before_split)

            split_support_x = self.get_split_features(support, preprocess=True, center=self.support_center, preprocess_method=self.preprocess_after_split)
            split_query_x = self.get_split_features(query, preprocess=True, center=self.query_center, preprocess_method=self.preprocess_after_split)

            support_features = self.fuse_features(split_support_x, support_ed)
            query_features = self.fuse_features(split_query_x, query_ed)

            clf = backbone.MultiNNClassifier(self.n_way, self.n_splits, measure=self.measure, temp=1)
            clf.fit(support_features, y_support.cpu().numpy())
            scores = clf.predict(query_features)
            if self.use_counterfactual:
                if not self.process_cx:
                    if self.x_zero:
                        c_x = torch.zeros(split_query_x.shape).cuda()
                    else:
                        c_x = split_support_x.mean(dim=1).unsqueeze(1).expand(split_query_x.shape)
                else:
                    if self.x_zero:
                        c_x = torch.zeros(query.shape).cuda()
                    else:
                        c_x = support.mean(dim=0).unsqueeze(0).expand(query.shape)
                    c_x = self.get_split_features(c_x, preprocess=True, center=self.query_center, preprocess_method=self.preprocess_after_split)
                c_query_feature = self.fuse_features(c_x, query_ed)
                c_scores = clf.predict(c_query_feature)
                scores = scores - c_scores
            return scores

    def set_forward_loss(self, x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')
        
 