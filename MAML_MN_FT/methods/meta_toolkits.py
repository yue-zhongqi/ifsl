import torch
import torch.nn as nn
import backbone
from torch.autograd import Variable
import numpy as np
import math


class MatchingNetModule(nn.Module):
    def __init__(self, feat_dim):
        super(MatchingNetModule, self).__init__()
        self.feat_dim = feat_dim
        self.FCE = FullyContextualEmbedding(self.feat_dim).cuda()
        self.G_encoder = nn.LSTM(self.feat_dim, self.feat_dim, 1, batch_first=True, bidirectional=True).cuda()

    def forward(self, support, query):
        G_encoder = self.G_encoder
        FCE = self.FCE
        out_G = G_encoder(support.unsqueeze(0))[0]
        out_G = out_G.squeeze(0)
        G = support + out_G[:, :support.size(1)] + out_G[:, support.size(1):]
        F = FCE(query, G)
        return G, F

    def cuda(self):
        super(MatchingNetModule, self).cuda()
        self.FCE = self.FCE.cuda()
        self.G_encoder = self.G_encoder.cuda()
        return self


class FullyContextualEmbedding(nn.Module):
    def __init__(self, feat_dim):
        super(FullyContextualEmbedding, self).__init__()
        self.lstmcell = nn.LSTMCell(feat_dim * 2, feat_dim)
        self.softmax = nn.Softmax()
        self.c_0 = Variable(torch.zeros(1, feat_dim))
        self.feat_dim = feat_dim

    def forward(self, f, G):
        h = f
        c = self.c_0.expand_as(f)
        G_T = G.transpose(0, 1)
        K = G.size(0)  # Tuna to be comfirmed
        for k in range(K):
            logit_a = h.mm(G_T)
            a = self.softmax(logit_a)
            r = a.mm(G)
            x = torch.cat((f, r), 1)
            h, c = self.lstmcell(x, (h, c))
            h = h + f
        return h

    def cuda(self):
        super(FullyContextualEmbedding, self).cuda()
        self.c_0 = self.c_0.cuda()
        self.lstmcell = self.lstmcell.cuda()
        return self


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
        print(self.preprocess_before_split, self.preprocess_after_split, self.normalize_before_center,
              self.normalize_d, self.normalize_ed)

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
        with torch.no_grad():
            proba = self.pretrain.classify(x)
        return proba

    def normalize(self, x, dim=1):
        x_norm = torch.norm(x, p=2, dim=dim).unsqueeze(dim).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        return x_normalized

    def get_d_feature(self, x):
        feat_dim = int(self.feat_dim / self.n_splits)
        if self.d_feature == "ed":
            d_feat_dim = int(self.feat_dim / self.n_splits)
        else:
            d_feat_dim = self.num_classes
        d_feature = torch.zeros(self.n_splits, x.shape[0], d_feat_dim).cuda()
        for i in range(self.n_splits):
            start = i * feat_dim
            stop = start + feat_dim
            pd = self.calc_pd(x, i)
            if self.d_feature == "pd":
                d_feature[i] = pd
            else:
                d_feature[i] = torch.mm(pd, self.pretrain_features)[:, start:stop]
        return d_feature

    def get_features(self, support, query):
        support_d = self.get_d_feature(support)
        query_d = self.get_d_feature(query)
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


class MAMLBlock(nn.Module):
    def __init__(self, feat_dim, n_way, update_step, approx=True, lr=0.01):
        super(MAMLBlock, self).__init__()
        self.in_dim = feat_dim
        self.hidden_dim = int(self.in_dim / 2)
        # self.hidden_dim = self.in_dim
        #self.out_dim = int(self.hidden_dim / 2)
        self.out_dim = self.hidden_dim * 2
        self.n_way = n_way
        self.update_step = update_step
        self.approx = approx
        self.lr = lr

        self.loss_fn = nn.CrossEntropyLoss()
        self.feature = nn.Sequential(backbone.Linear_fw(self.in_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     backbone.Linear_fw(self.hidden_dim, self.out_dim))
        self.classifier = backbone.Linear_fw(self.out_dim, self.n_way)
        self.classifier.bias.data.fill_(0)

    def forward(self, x):
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
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


class LEOBlock(nn.Module):
    def __init__(self, feat_dim, latent_dim, n_way, drop_rate):
        super(LEOBlock, self).__init__()
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.n_way = n_way
        self.encoder = nn.Linear(self.feat_dim, self.latent_dim, bias=False)
        relation_dim = 2 * self.latent_dim
        self.relation_dim = relation_dim
        self.relation_net = nn.Sequential(nn.Linear(relation_dim, relation_dim, bias=False),
                                          nn.Linear(relation_dim, relation_dim, bias=False),
                                          nn.Linear(relation_dim, relation_dim, bias=False),
                                          )
        self.decoder = nn.Linear(self.latent_dim, self.feat_dim * 2, bias=False)
        self.latent_lr = nn.Parameter(torch.ones(1))  # initialize as 1
        self.param_lr = nn.Parameter(torch.ones(1) / 1000)  # initialize as 0.001
        self.latent_steps = 5
        self.param_steps = 5
        self.drop_rate = drop_rate
        self.loss = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

    def prepare_for_rel_net(self, embeddings):
        support_size = embeddings.shape[0]
        interleave_emb = torch.repeat_interleave(embeddings, support_size, dim=0)  # 000001111122222...
        tiled_embs = embeddings.repeat(support_size, 1)  # 0123401234...
        cat_embs = torch.cat((interleave_emb, tiled_embs), dim=1)
        return cat_embs

    def get_sampled_weights(self, weight_stats):
        w = torch.zeros(self.n_way, self.feat_dim, dtype=torch.float64).cuda()  # 5*640
        for i in range(self.n_way):
            mu = weight_stats[i, :self.feat_dim]
            Sigma = weight_stats[i, self.feat_dim:]
            m = torch.distributions.MultivariateNormal(mu, scale_tril=torch.diag(Sigma))
            w[i] = m.rsample()
        return w

    def finetune_z(self, support, initial_z, initial_w, labels):
        support_size = support.shape[0]
        w = initial_w
        z = initial_z
        for _ in range(self.latent_steps):
            if self.training:
                support_dropped = self.dropout(support)
            else:
                support_dropped = support
            support_expand = support_dropped.unsqueeze(1).expand(-1, self.n_way, -1)
            w_expand = w.unsqueeze(0).expand(support_size, -1, -1)
            logits = (w_expand * support_expand).sum(dim=2)
            loss = self.loss(logits, labels)
            grad = torch.autograd.grad(loss, z)
            z = z - self.latent_lr * grad[0]
            w = self.forward_decoder(z)
        return z, w

    def finetune_w(self, support, initial_w, labels):
        support_size = support.shape[0]
        w = initial_w
        for _ in range(self.latent_steps):
            if self.training:
                support_dropped = self.dropout(support)
            else:
                support_dropped = support
            support_expand = support_dropped.unsqueeze(1).expand(-1, self.n_way, -1)
            w_expand = w.unsqueeze(0).expand(support_size, -1, -1)
            logits = (w_expand * support_expand).sum(dim=2)
            loss = self.loss(logits, labels)
            grad = torch.autograd.grad(loss, w)
            w = w - self.param_lr * grad[0]
        return w

    def forward_relation_net(self, embeddings):
        support_size = embeddings.shape[0]
        cat_embs = self.prepare_for_rel_net(embeddings)
        relation_output = self.relation_net(cat_embs)   # 625*128 for 5 shots
        outputs = relation_output.view(self.n_way, -1, support_size, self.relation_dim)  # 5way*5shot*25*128
        outputs = outputs.mean(dim=2)  # 5way * 5 shot * 128
        return outputs

    def average_codes_per_class(self, relation_net_outputs):
        # relation_net_outputs is 5 way * 5 shot * 128
        codes = relation_net_outputs.mean(dim=1)
        codes = codes.unsqueeze(1).expand_as(relation_net_outputs)
        return codes

    def possibly_sample(self, distribution_params, stddev_offset=0.):
        dim = int(distribution_params.shape[2] / 2)
        means = distribution_params[:, :, :dim]
        unnormalized_stddev = distribution_params[:, :, dim:]
        stddev = torch.exp(unnormalized_stddev)
        stddev = stddev - Variable(torch.tensor(1. - stddev_offset).cuda().double().expand(stddev.size()))
        # stddev = torch.max(stddev, Variable(torch.tensor(1e-10).cuda().double()))
        # x[torch.lt(x,0.5)]=0.5
        # stddev[torch.lt(stddev, 1e-10)] = 1e-10
        # stddev_new = stddev.clone()
        # stddev_new = stddev.clamp(1e-10, 100)
        stddev = self.relu(stddev) + 1e-10
        distributions = torch.distributions.Normal(loc=means, scale=stddev)
        return distributions.rsample(), distributions

    def forward_decoder(self, latents):
        # latents is N * K * 64
        weights_dist_params = self.decoder(latents)
        fan_in = self.feat_dim
        fan_out = self.n_way
        stddev_offset = math.sqrt(2. / (fan_out + fan_in))
        classifier_weights, _ = self.possibly_sample(weights_dist_params, stddev_offset)
        classifier_weights = classifier_weights.mean(dim=1)  # N * 640
        return classifier_weights

    def calc_kl_penalty(self, latent_samples, latent_distributions):
        mean = torch.zeros_like(latent_samples).cuda().double()
        std = torch.ones_like(latent_samples).cuda().double()
        prior = torch.distributions.Normal(loc=mean, scale=std)
        kl = (latent_distributions.log_prob(latent_samples) - prior.log_prob(latent_samples)).mean()
        self.kl_penalty = kl

    def calc_encoder_penalty(self, z_f, z):
        self.encoder_penalty = torch.norm(z_f.detach() - z, p=2) ** 2

    def calc_l2_penalty(self):
        l2_penalty = 0
        params = list(self.encoder.parameters()) \
            + list(self.relation_net.parameters()) \
            + list(self.decoder.parameters())
        for param in params:
            l2_penalty += (torch.norm(param, p=2) ** 2)
        self.l2_penalty = l2_penalty

    def calc_orthogonality_penalty(self):
        # weight of shape 1280 * 64
        w2 = torch.mm(self.decoder.weight.t(), self.decoder.weight)
        wn = torch.norm(self.decoder.weight, dim=0, keepdim=True, p=2) + 1e-32
        corr_matrix = w2 / torch.mm(wn.t(), wn)
        matrix_size = corr_matrix.shape[0]
        identity = torch.eye(matrix_size).cuda().double()
        orthogonality_penalty = ((corr_matrix - identity) * (corr_matrix - identity)).mean()
        self.orthogonality_penalty = orthogonality_penalty

    def fit(self, support, labels):
        torch.autograd.set_detect_anomaly(True)
        support = support.double()
        embeddings = self.encoder(support)
        relation_net_outputs = self.forward_relation_net(embeddings)
        latent_dist_params = self.average_codes_per_class(relation_net_outputs)
        latents, latent_distributions = self.possibly_sample(latent_dist_params)
        self.calc_kl_penalty(latents, latent_distributions)
        weights = self.forward_decoder(latents)
        z_f, w = self.finetune_z(support, latents, weights, labels)
        w = self.finetune_w(support, w, labels)
        self.calc_encoder_penalty(z_f, latents)
        self.calc_l2_penalty()
        self.calc_orthogonality_penalty()
        return w

    def predict(self, query, w):
        query_size = query.shape[0]
        query_expand = query.unsqueeze(1).expand(-1, self.n_way, -1)
        w_expand = w.unsqueeze(0).expand(query_size, -1, -1)
        logits = (w_expand * query_expand).sum(dim=2)
        return logits

