# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from numpy import linalg as LA
import os


# Basic ResNet model
def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class NNClassifier():
    def __init__(self, n_way):
        self.n_way = n_way

    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        return x_normalized

    def preprocess(self, data):
        '''
        if self.preprocessing == "none":
            return data
        elif self.preprocessing == "l2n":
            return self.normalize(data)
        '''
        return data  # Do reprocessing outside

    def dist(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def kl_divergence(self, k1, k2):
        k1_safe = k1 + 0.0001
        k2_safe = k2 + 0.0001
        t = k1_safe * torch.log(k1_safe / k2_safe)
        return torch.sum(t, dim=2)

    def fit(self, support, support_labels, support_weights=None):
        self.support = support
        self.labels = support_labels
        self.feat_dim = support.shape[1]
        processed_support = support
        self.centroids = torch.zeros(self.n_way, self.feat_dim).cuda()
        for i in range(self.n_way):
            class_support = processed_support[support_labels == i]
            if support_weights is None:
                self.centroids[i] = class_support.mean(dim=0)
            else:
                class_support_weights = support_weights[support_labels == i]
                softmax = nn.Softmax(dim=0)
                class_support_weights = softmax(class_support_weights)
                class_support_weights = class_support_weights.unsqueeze(1).expand_as(class_support)
                weighted_support = class_support * class_support_weights
                self.centroids[i] = weighted_support.sum(dim=0)
    
    def predict(self, query):
        query_size = query.shape[0]
        processed_query = self.preprocess(query).cpu().numpy()
        scores = torch.zeros(query_size, self.n_way).cuda()
        for i in range(query_size):
            for j in range(self.n_way):
                d = self.dist(self.centroids[j], processed_query[i])
                scores[i][j] = np.exp(-d * d)
        softmax = torch.nn.Softmax(dim=1)
        scores = softmax(scores)
        return scores
    
    def predict_alt(self, query, measure="euclidean", norm_scores=False, temp=1.0):
        query_size = query.shape[0]
        scores = torch.zeros(query_size, self.n_way).cuda()
        processed_query = query.unsqueeze(1).expand(query_size, self.n_way, self.feat_dim)
        # processed_query = processed_query.cpu().numpy()
        centroids = self.centroids
        centroids = centroids.unsqueeze(0).expand(query_size, self.n_way, self.feat_dim)
        if measure == "euclidean":
            dist = torch.norm(processed_query - centroids, p=2, dim=2)
            scores = torch.exp(-dist * dist)
        elif measure == "cosine":
            inner_product = (processed_query * centroids).sum(dim=2)
            n2 = torch.norm(centroids, p=2, dim=2)
            n1 = torch.norm(processed_query, p=2, dim=2)
            dist = inner_product / n1 / n2
            scores = dist
        elif measure == "kl":
            # This has problem now
            dist = self.kl_divergence(processed_query, centroids) / 4.0
        elif measure == "linear":
            dist = torch.norm(processed_query - centroids, p=2, dim=2)
            scores = -dist * dist / temp
            if norm_scores:
                scores = scores / torch.norm(scores, p=2, dim=1).unsqueeze(1).expand(-1, self.n_way)
        softmax = torch.nn.Softmax(dim=1)
        scores = softmax(scores)
        return scores


class MultiNNBiClassifier():
    def __init__(self, n_way, n_classifiers, measure="linear", fusion="linear_sum", temp=1.0):
        self.n_way = n_way
        self.n_classifiers = n_classifiers
        self.x_clfs = [NNClassifier(n_way) for i in range(n_classifiers)]
        self.d_clfs = [NNClassifier(n_way) for i in range(n_classifiers)]
        self.measure = measure
        self.temp = temp
        self.proba_fusion = fusion

    def fit(self, support_x, support_d, support_labels, support_weights=None):
        for i in range(self.n_classifiers):
            if support_weights is None:
                self.x_clfs[i].fit(support_x[i], support_labels)
                self.d_clfs[i].fit(support_d[i], support_labels)
            else:
                self.x_clfs[i].fit(support_x[i], support_labels, support_weights=support_weights[:, i])
                self.d_clfs[i].fit(support_d[i], support_labels, support_weights=support_weights[:, i])

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

    def predict(self, query_x, query_d, weights=None, counterfactual=False):
        if isinstance(query_x, list):
            query_size = query_x[0].shape[0]
        else:
            query_size = query_x.shape[1]
        scores = torch.zeros(self.n_classifiers, query_size, self.n_way).cuda()
        for i in range(self.n_classifiers):
            d_scores = self.d_clfs[i].predict_alt(query_d[i], self.measure, temp=self.temp)
            if not counterfactual:
                x_scores = self.x_clfs[i].predict_alt(query_x[i], self.measure, temp=self.temp)
            else:
                x_scores = torch.ones(d_scores.shape).cuda() / self.n_way
            scores[i] = self.fuse_proba(x_scores, d_scores)
        if weights is None:
            combined_scores = scores.mean(dim=0)
        else:
            scores = scores.permute(1, 0, 2)
            weights = weights.unsqueeze(2).expand(-1, -1, self.n_way)
            combined_scores = (weights * scores).sum(dim=1)
        return combined_scores


class MultiNNClassifier():
    def __init__(self, n_way, n_classifiers, measure="euclidean", temp=1.0):
        self.n_way = n_way
        self.n_classifiers = n_classifiers
        self.clfs = [NNClassifier(n_way) for i in range(n_classifiers)]
        self.measure = measure
        self.temp = temp

    '''
    support of shape (n_classifiers, N, feature_dim)
    '''
    def fit(self, support, support_labels, support_weights=None):
        for i in range(self.n_classifiers):
            if support_weights is None:
                self.clfs[i].fit(support[i], support_labels)
            else:
                self.clfs[i].fit(support[i], support_labels, support_weights=support_weights[:, i])

    '''
    query of shape (n_classifiers, N, feature_dim)
    optionally provide weights of shape (n_classifiers) for a weighted average
    '''
    def predict(self, query, weights=None):
        if isinstance(query, list):
            query_size = query[0].shape[0]
        else:
            query_size = query.shape[1]
        scores = torch.zeros(self.n_classifiers, query_size, self.n_way).cuda()
        for i in range(self.n_classifiers):
            classifier_scores = self.clfs[i].predict_alt(query[i], self.measure, temp=self.temp)
            scores[i] = classifier_scores
        self.scores = scores
        if weights is None:
            combined_scores = scores.mean(dim=0)
        else:
            # weights = weights.unsqueeze(1).expand(-1, query_size).unsqueeze(2).expand(-1, -1, self.n_way)
            scores = scores.permute(1, 0, 2)
            weights = weights.unsqueeze(2).expand(-1, -1, self.n_way)
            combined_scores = (weights * scores).sum(dim=1)
        return combined_scores


class BidrectionalLSTM(nn.Module):
    def __init__(self, size: int, layers: int):
        """Bidirectional LSTM used to generate fully conditional embeddings (FCE) of the support set as described
        in the Matching Networks paper.

        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            layers: Number of LSTM layers
        """
        super(BidrectionalLSTM, self).__init__()
        self.num_layers = layers
        self.batch_size = 1
        # Force input size and hidden size to be the same in order to implement
        # the skip connection as described in Appendix A.1 and A.2 of Matching Networks
        self.lstm = nn.LSTM(input_size=size,
                            num_layers=layers,
                            hidden_size=size,
                            bidirectional=True)

    def forward(self, inputs):
        # Give None as initial state and Pytorch LSTM creates initial hidden states
        output, (hn, cn) = self.lstm(inputs, None)

        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]

        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i) as written in Appendix A.2
        # AKA A skip connection between inputs and outputs is used
        output = forward_output + backward_output + inputs
        return output, hn, cn


class AttentionLSTM(nn.Module):
    def __init__(self, size: int, unrolling_steps: int):
        """Attentional LSTM used to generate fully conditional embeddings (FCE) of the query set as described
        in the Matching Networks paper.

        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            unrolling_steps: Number of steps of attention over the support set to compute. Analogous to number of
                layers in a regular LSTM
        """
        super(AttentionLSTM, self).__init__()
        self.unrolling_steps = unrolling_steps
        self.lstm_cell = nn.LSTMCell(input_size=size,
                                     hidden_size=size)

    def forward(self, support, queries):
        # Get embedding dimension, d
        if support.shape[-1] != queries.shape[-1]:
            raise(ValueError("Support and query set have different embedding dimension!"))

        batch_size = queries.shape[0]
        embedding_dim = queries.shape[1]

        h_hat = torch.zeros_like(queries).cuda().double()
        c = torch.zeros(batch_size, embedding_dim).cuda().double()

        for k in range(self.unrolling_steps):
            # Calculate hidden state cf. equation (4) of appendix A.2
            h = h_hat + queries

            # Calculate softmax attentions between hidden states and support set embeddings
            # cf. equation (6) of appendix A.2
            attentions = torch.mm(h, support.t())
            attentions = attentions.softmax(dim=1)

            # Calculate readouts from support set embeddings cf. equation (5)
            readout = torch.mm(attentions, support)

            # Run LSTM cell cf. equation (3)
            # h_hat, c = self.lstm_cell(queries, (torch.cat([h, readout], dim=1), c))
            h_hat, c = self.lstm_cell(queries, (h + readout, c))
        h = h_hat + queries
        return h


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


class ResNetKernelClusterAgent():
    def __init__(self, pretrain, n_clusters, pca_dim, cluster_method="kmeans"):
        self.pretrain = pretrain
        self.n_clusters = n_clusters
        self.cluster_method = cluster_method
        self.pca_dim = pca_dim

    def fit(self):
        # !!!! Note this 30 is a hard coded value that may need to change if not using ResNet10
        weights = list(self.pretrain.model.parameters())[30]
        # weights = weights.permute(1, 0, 2, 3)
        # kernel_features = torch.flatten(weights, 1, 3)
        kernel_features = torch.mean(weights, dim=1)
        kernel_features = torch.flatten(kernel_features, 1, 2)
        kernel_features_np = kernel_features.cpu().detach().numpy()
        transformed = kernel_features_np
        # Cluster: in this case, PCA then KMeans
        # pca = PCA(n_components=self.pca_dim)
        # transformed = pca.fit_transform(kernel_features_np)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(transformed)
        return kmeans.labels_


class ResNetParamClusterModel():
    def __init__(self, pretrain, n_clusters, cluster_method="kmeans"):
        self.pretrain = pretrain
        self.cluster_method = cluster_method
        self.n_clusters = n_clusters

    def cluster(self, features, n_clusters):
        if self.cluster_method == "kmeans":
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(features)
            return kmeans.labels_

    def get_weight_features(self, weights):
        kernel_features = torch.mean(weights, dim=1)
        kernel_features = torch.flatten(kernel_features, 1, 2)
        kernel_features_np = kernel_features.cpu().detach().numpy()
        return kernel_features_np

    def fit(self):
        c1_weights = list(self.pretrain.model.parameters())[27]
        c2_weights = list(self.pretrain.model.parameters())[30]
        c1_features = self.get_weight_features(c1_weights)
        c2_features = self.get_weight_features(c2_weights)
        self.c1_labels = self.cluster(c1_features, self.n_clusters)
        self.c2_labels = self.cluster(c2_features, self.n_clusters)

    def conv_forward(self, inputs, labels, n_clusters, original_conv):
        output = original_conv(inputs)
        desired_output = output.unsqueeze(0).expand(n_clusters, -1, -1, -1, -1)
        # N * n_channels * size * size
        new_output = torch.zeros(desired_output.shape).cuda()
        for i in range(n_clusters):
            cluster_output = output[:, labels == i, :, :]
            n_channels = output.shape[1]
            cluster_channels = cluster_output.shape[1]
            n_repeat = int(n_channels / cluster_channels)
            # Tile up the entire n_channels
            tiled_channels = n_repeat * cluster_channels
            remaining_channels = n_channels - tiled_channels
            tiled_cluster_output = cluster_output.repeat(1, n_repeat, 1, 1)
            #new_output[i, :, :tiled_channels, :, :] = tiled_cluster_output
            #new_output[i, :, tiled_channels:, :, :] = cluster_output[:, :remaining_channels, :, :]
            # Set specific channels to cluster output value
            new_output[i, :, labels == i, :, :] = cluster_output
        return new_output

    def forward(self, imgs):
        # imgs are N * 3 * 224 * 224
        model = self.pretrain.model.feature.trunk
        # 1. Get output before trunk 7
        trunk6_out = model[:7](imgs)
        # 2. Get output of C1
        '''
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out
        '''
        t7 = model[7]
        c1_outputs = self.conv_forward(trunk6_out, self.c1_labels, self.n_clusters, t7.C1)
        output_features = []
        shapes = []
        for i in range(self.n_clusters):
            out = c1_outputs[i]
            out = t7.BN1(out)
            out = t7.relu1(out)
            out[:, self.c1_labels!=i, :, :] = 0
            c2_outputs = self.conv_forward(out, self.c2_labels, self.n_clusters, t7.C2)
            for j in range(self.n_clusters):
                out = c2_outputs[j]
                out = t7.BN2(out)
                short_out = t7.BNshortcut(t7.shortcut(trunk6_out))
                out = out + short_out
                out = t7.relu2(out)
                out = model[8:](out).detach()
                out = out[:, self.c2_labels == j]
                output_features.append(out)
                shapes.append(out.shape[1])
        return output_features, shapes

    def forward2(self, imgs):
        # Use only 1 conv block to cluster
        # imgs are N * 3 * 224 * 224
        model = self.pretrain.model.feature.trunk
        # 1. Get output before trunk 7
        trunk6_out = model[:7](imgs)
        # 2. Get output of C1
        t7 = model[7]
        out = t7.C1(trunk6_out)
        out = t7.BN1(out)
        out = t7.relu1(out)
        c2_outputs = self.conv_forward(out, self.c2_labels, self.n_clusters, t7.C2)
        output_features = []
        shapes = []
        for i in range(self.n_clusters):
            out = c2_outputs[i]
            out = t7.BN2(out)
            short_out = t7.BNshortcut(t7.shortcut(trunk6_out))
            out = out + short_out
            out = t7.relu2(out)
            out = model[8:](out).detach()
            out = out[:, self.c2_labels == i]
            output_features.append(out)
            shapes.append(out.shape[1])
        return output_features, shapes

    def forward3(self, imgs):
        model = self.pretrain.model.feature.trunk
        trunk6_out = model[:7](imgs)
        # 2. Get output of C1
        t7 = model[7]
        

class BasisTransformer():
    def __init__(self, pretrain, recluster=False, cluster_method="kmeans", mode='project', kernel='rbf'):
        self.pretrain = pretrain
        self.recluster = recluster
        self.mode = mode
        self.kernel = kernel
        self.cluster_method = cluster_method

    def fit(self, n_clusters, feat_dim, pca_dim=50):
        self.feat_dim = feat_dim
        self.n_clusters = n_clusters
        # Get features
        features, labels = self.pretrain.get_pretrain_dataset('base')
        # Perform PCA dimension reduction before k-means
        if pca_dim > 0:
            pca_model = PCA(n_components=pca_dim)
            features_reduced = pca_model.fit_transform(features)
        else:
            features_reduced = features
        
        # K Means clustering
        if self.recluster:
            if self.cluster_method == "kmeans":
                new_labels_file = "kmeans/new_labels_%s_%s_%s_%s.npy" % (str(self.n_clusters), self.pretrain.method,
                                                                         self.pretrain.model_name, self.pretrain.dataset)
                if os.path.isfile(new_labels_file):
                    new_labels = np.load(new_labels_file)
                else:
                    kmeans_model = KMeans(n_clusters=n_clusters, random_state=0).fit(features_reduced)
                    self.kmeans_model = kmeans_model
                    new_labels = kmeans_model.labels_
                    np.save(new_labels_file, new_labels)
            elif self.cluster_method == "hdbscan":
                a = 1
        else:
            new_labels = labels

        # Fit basis transformation function
        self.basis_transform_models = []
        for i in range(n_clusters):
            cluster_features = features[new_labels == i]
            if self.mode == 'project':
                model = PCA(n_components=feat_dim)
                model.fit(cluster_features)
            elif self.mode == 'kernel':
                model = KernelTransformer(feat_dim, self.kernel)
                model.fit(cluster_features)
            self.basis_transform_models.append(model)

    def transform(self, X):
        # X is N * original_feat_dim
        N = X.shape[0]
        transformed = np.zeros((self.n_clusters, N, self.feat_dim))
        for i in range(self.n_clusters):
            transformed[i] = self.basis_transform_models[i].transform(X)
        return transformed


class KernelTransformer():
    def __init__(self, feat_dim, kernel):
        self.feat_dim = feat_dim
        self.kernel = kernel

    def fit(self, features):
        N = features.shape[0]
        # Randomly sample feat_dim points from features
        rand_id = np.random.permutation(N)
        selected_id = rand_id[:self.feat_dim]
        self.centroids = features[selected_id]

    def transform(self, X):
        N = X.shape[0]
        transformed = np.zeros((N, self.feat_dim))
        for i in range(N):
            for j in range(self.feat_dim):
                transformed[i][j] = self.kernel_f(X[i], self.centroids[j])
        return transformed

    def kernel_f(self, x1, x2):
        if self.kernel == "rbf":
            diff = x1 - x2
            norm = LA.norm(diff, ord=2)
            t = -0.5 * norm * norm / 10
            k = np.exp(t)
        elif self.kernel == "linear":
            k = np.sum(x1 * x2)
        return k


class ChannelwiseClassifier(nn.Module):
    def __init__(self, feat_dim, n_way, weight, bias=False):
        super(ChannelwiseClassifier, self).__init__()
        self.n_way = n_way
        self.feat_dim = feat_dim
        self.bias = bias
        self.W = nn.Parameter(torch.Tensor(n_way, feat_dim))
        if self.bias:
            self.B = nn.Parameter(torch.Tensor(n_way, feat_dim))
        # self.reset_parameters()
        with torch.no_grad():
            self.W.copy_(weight)
    
    def reset_parameters(self):
        # init.kaiming_uniform_(self.W, a=math.sqrt(5))
        init.uniform_(self.W, -0.75, 0.75)
        # init.normal_(self.W)
        if self.bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.B, -bound, bound)

    def forward(self, X):
        n = X.shape[0]
        # W_expanded = self.W.unsqueeze(0).expand(n, -1, -1)
        X_expanded = X.unsqueeze(1).expand(-1, self.n_way, -1)
        scores = torch.zeros(n, self.n_way, self.feat_dim).cuda()
        for i in range(n):
            if self.bias:
                scores[i] = self.W * X_expanded[i] + self.B
            else:
                dist = (self.W - X_expanded[i]).pow(2)
                scores[i] = -dist
        '''
        if self.bias:
            B_expanded = self.B.unsqueeze(0).expand(n, -1, -1)
            scores = W_expanded * X_expanded + B_expanded
        else:
            scores = W_expanded * X_expanded
        '''
        scores = scores
        softmax = nn.Softmax(dim=1)
        softmax_scores = softmax(scores)
        probability = torch.mean(softmax_scores, dim=2).view(n, self.n_way)
        return probability, softmax_scores


class UnbiasedClassifier(nn.Module):
    '''
    Input:
        architecture
            softmax
        feature_fusion
            concat
        has_x_branch_classifier
            If True, will have a separate x classifier and will produce self.branch_clf_resp after forward
        logit_fusion
            Requried if has_x_branch_classifier; Otherwise ignored
            product, sum, harmonic
    '''
    def __init__(self, n_way, x_feature_dim, z_feature_dim, d_feature_dim=0,
                 has_d_branch=False, has_x_branch_classifier=False, architecture="softmax",
                 feature_fusion="concat", logit_fusion="product"):
        super(UnbiasedClassifier, self).__init__()
        self.n_way = n_way
        self.architecture = architecture
        self.feature_fusion = feature_fusion
        self.logit_fusion = logit_fusion
        self.has_x_branch_classifier = has_x_branch_classifier
        self.has_d_branch = has_d_branch
        self.x_feature_dim = x_feature_dim
        self.z_feature_dim = z_feature_dim
        self.d_feature_dim = d_feature_dim
        self.main_feature_dim = x_feature_dim + z_feature_dim
        if self.has_d_branch:
            self.main_feature_dim += d_feature_dim
        if self.has_x_branch_classifier:
            self.branch_clf = self.create_clf(x_feature_dim)
            self.logit_fusion_fn = self.create_logit_fusion_fn()
        self.main_clf = self.create_clf(self.main_feature_dim)

    def create_clf(self, feat_dim):
        if self.architecture == "softmax":
            clf = nn.Linear(feat_dim, self.n_way).cuda()
        elif self.architecture == "dist":
            clf = distLinear(feat_dim, self.n_way, True).cuda()
        return clf
    
    def create_logit_fusion_fn(self):
        if self.logit_fusion == "product":
            fn = ProductGate().cuda()
        if self.logit_fusion == "harmonic":
            fn = HarmonicGate().cuda()
        if self.logit_fusion == "sum":
            fn = SumGate().cuda()
        return fn

    def get_fused_feature(self, feature_array):
        if self.feature_fusion == "concat":
            return torch.cat(feature_array, 1)

    def forward(self, X, Z, D=None):
        batch_size = X.shape[0]
        if self.has_d_branch:
            fused_feature = self.get_fused_feature((X, Z, D))
        else:
            fused_feature = self.get_fused_feature((X, Z))
        main_clf_resp = self.main_clf(fused_feature)

        if self.has_x_branch_classifier:
            branch_feat = X
            branch_clf_resp = self.branch_clf(branch_feat)
            cat_resp = torch.cat((main_clf_resp.view(batch_size, self.n_way, 1), branch_clf_resp.view(batch_size, self.n_way, 1)), dim=2)
            combined_resp = self.logit_fusion_fn(cat_resp).view(batch_size, self.n_way)
            self.branch_clf_resp = branch_clf_resp
            return combined_resp
        else:
            return main_clf_resp


class XDBiClassifier(nn.Module):
    def __init__(self, n_way, x_feature_dim, d_feature_dim, architecture="softmax", 
                 fusion="product", d_clf_is_linear=True, sigmoid_d_resp=False):
        super(XDBiClassifier, self).__init__()
        self.n_way = n_way
        self.architecture = architecture
        self.x_feature_dim = x_feature_dim
        self.d_feature_dim = d_feature_dim
        self.x_clf = self.create_clf(self.x_feature_dim)
        if d_clf_is_linear:
            self.d_clf = nn.Linear(self.d_feature_dim, self.n_way).cuda()
        else:
            self.d_clf = self.create_clf(self.d_feature_dim)
        self.logit_fusion = fusion
        self.fusion_fn = self.create_logit_fusion_fn()
        self.sigmoid = nn.Sigmoid().cuda()
        self.sigmoid_d_resp = sigmoid_d_resp

    def create_clf(self, feat_dim):
        if self.architecture == "softmax":
            clf = nn.Linear(feat_dim, self.n_way).cuda()
        elif self.architecture == "dist":
            clf = distLinear(feat_dim, self.n_way, True).cuda()
        return clf

    def create_logit_fusion_fn(self):
        if self.logit_fusion == "product":
            fn = ProductGate().cuda()
        if self.logit_fusion == "harmonic":
            fn = HarmonicGate().cuda()
        if self.logit_fusion == "sum":
            fn = SumGate().cuda()
        if self.logit_fusion == "linear_sum":
            fn = nn.Linear(2, 1, bias=False).cuda()
        return fn

    def cat_for_logit_fusion(self, A, B):
        batch_size = A.shape[0]
        cat_resp = torch.cat((A.view(batch_size, self.n_way, 1), B.view(batch_size, self.n_way, 1)), dim=2)
        return cat_resp

    def forward(self, X, D):
        actual_batch_size = X.shape[0]
        x_resp = self.x_clf(X)
        d_resp = self.d_clf(D)
        self.d_resp = d_resp
        self.x_resp = x_resp
        if self.logit_fusion != "linear_sum":
            cat_resp = self.cat_for_logit_fusion(x_resp, d_resp)
            scores = self.fusion_fn(cat_resp).view(actual_batch_size, self.n_way)
        else:
            # change d_resp to -1 to 1
            if self.sigmoid_d_resp:
                d_resp = (self.sigmoid(d_resp) - 0.5) * 2
            scores = x_resp + d_resp
        return scores


class XDClassifier(nn.Module):
    '''
    Input:
        architecture
            softmax
        feature_fusion
            concat
        has_x_branch_classifier
            If True, will have a separate x classifier and will produce self.branch_clf_resp after forward
        logit_fusion
            Requried if has_x_branch_classifier; Otherwise ignored
            product, sum, harmonic
    '''
    def __init__(self, n_way, x_feature_dim, d_feature_dim, architecture="softmax",
                 feature_fusion="concat", transform_d=False, hidden_nodes=50, use_d=True):
        super(XDClassifier, self).__init__()
        self.n_way = n_way
        self.architecture = architecture
        self.feature_fusion = feature_fusion
        self.x_feature_dim = x_feature_dim
        self.d_feature_dim = d_feature_dim
        self.feat_dim = self.get_feature_dim()
        self.transform_d = transform_d
        self.use_d = use_d
        if self.transform_d:
            self.hidden_nodes = hidden_nodes
            self.transform_linear = nn.Linear(self.feat_dim, self.hidden_nodes).cuda()
            self.transform_activation = nn.LeakyReLU().cuda()
            self.clf = self.create_clf(self.hidden_nodes)
        else:
            if use_d:
                self.clf = self.create_clf(self.feat_dim)
            else:
                self.clf = self.create_clf(self.x_feature_dim)

    def get_feature_dim(self):
        if self.feature_fusion == "concat":
            return self.x_feature_dim + self.d_feature_dim
        elif self.feature_fusion == "sum":
            assert self.x_feature_dim == self.d_feature_dim
            return self.x_feature_dim
        elif self.feature_fusion == "gate":
            assert self.x_feature_dim == self.d_feature_dim
            return self.x_feature_dim
        elif self.feature_fusion == "-":
            return self.x_feature_dim
        elif self.feature_fusion == "+":
            return self.x_feature_dim

    def create_clf(self, feat_dim):
        if self.architecture == "softmax":
            clf = nn.Linear(feat_dim, self.n_way).cuda()
        elif self.architecture == "dist":
            clf = distLinear(feat_dim, self.n_way, True).cuda()
        return clf

    def get_fused_feature(self, feature_array):
        if self.feature_fusion == "concat":
            return torch.cat(feature_array, 1)
        elif self.feature_fusion == "-":
            return feature_array[0] - feature_array[1]
        elif self.feature_fusion == "+":
            return feature_array[0] + feature_array[1]

    def forward(self, X, D):
        if self.transform_d:
            fused_feature = self.get_fused_feature((X, D))
            hidden_resp = self.transform_linear(fused_feature)
            activated_resp = self.transform_activation(hidden_resp)
            clf_resp = self.clf(activated_resp)
        else:
            if self.use_d:
                fused_feature = self.get_fused_feature((X, D))
                clf_resp = self.clf(fused_feature)
            else:
                clf_resp = self.clf(X)
        return clf_resp


class ProductGate(nn.Module):
    def __init__(self):
        super(ProductGate, self).__init__()

    def forward(self, x):
        permuted = x.permute(1, 0, 2)
        sigmoid = nn.Sigmoid().cuda()
        sig_results = sigmoid(permuted)
        product = torch.mul(sig_results[:, :, 0], sig_results[:, :, 1])
        log = torch.log(product).permute(1, 0)
        return log

class HarmonicGate(nn.Module):
    def __init__(self):
        super(HarmonicGate, self).__init__()

    def forward(self, x):
        permuted = x.permute(1, 0, 2)
        sigmoid = nn.Sigmoid().cuda()
        sig_results = sigmoid(permuted)
        product = torch.mul(sig_results[:, :, 0], sig_results[:, :, 1])
        val = product / (1 + product)
        log = torch.log(val).permute(1, 0)
        return log

class SumGate(nn.Module):
    def __init__(self):
        super(SumGate, self).__init__()

    def forward(self, x):
        permuted = x.permute(1, 0, 2)
        sigmoid = nn.Sigmoid().cuda()
        sig_results = sigmoid(permuted[:, :, 0] + permuted[:, :, 1])
        log = torch.log(sig_results).permute(1, 0)
        return log

class distLinear(nn.Module):
    def __init__(self, indim, outdim, class_wise_learnable_norm=True):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = class_wise_learnable_norm  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <= 200:
            self.scale_factor = 2  # a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github 
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)


class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight 
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out

class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out
            
class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight 
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
            #batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out

# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, pool = True, padding = 1):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        if self.maml:
            self.C      = Conv2d_fw(indim, outdim, 3, padding = padding)
            self.BN     = BatchNorm2d_fw(outdim)
        else:
            self.C      = nn.Conv2d(indim, outdim, 3, padding= padding)
            self.BN     = nn.BatchNorm2d(outdim)
        self.relu   = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)


    def forward(self,x):
        out = self.trunk(x)
        return out

# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = BatchNorm2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out



# Bottleneck block
class BottleneckBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim/4)
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = BatchNorm2d_fw(bottleneckdim)
            self.C2 = Conv2d_fw(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = BatchNorm2d_fw(bottleneckdim)
            self.C3 = Conv2d_fw(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = nn.BatchNorm2d(bottleneckdim)
            self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = nn.BatchNorm2d(bottleneckdim)
            self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res


        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)


    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out


class ConvNet(nn.Module):
    def __init__(self, depth, flatten = True):
        super(ConvNet,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self,x):
        out = self.trunk(x)
        return out

class ConvNetNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth):
        super(ConvNetNopool,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1  ) #only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64,19,19]

    def forward(self,x):
        out = self.trunk(x)
        return out

class ConvNetS(nn.Module): #For omniglot, only 1 input channel, output dim is 64
    def __init__(self, depth, flatten = True):
        super(ConvNetS,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 64

    def forward(self,x):
        out = x[:,0:1,:,:] #only use the first dimension
        out = self.trunk(out)
        return out

class ConvNetSNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling. For omniglot, only 1 input channel, output dim is [64,5,5]
    def __init__(self, depth):
        super(ConvNetSNopool,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1  ) #only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64,5,5]

    def forward(self,x):
        out = x[:,0:1,:,:] #only use the first dimension
        out = self.trunk(out)
        return out

class ResNet(nn.Module):
    maml = False #Default
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten = True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)


        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)
        return out

def Conv4():
    return ConvNet(4)

def Conv6():
    return ConvNet(6)

def Conv4NP():
    return ConvNetNopool(4)

def Conv6NP():
    return ConvNetNopool(6)

def Conv4S():
    return ConvNetS(4)

def Conv4SNP():
    return ConvNetSNopool(4)

def ResNet10( flatten = True):
    return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten)

def ResNet18( flatten = True):
    return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten)

def ResNet34( flatten = True):
    return ResNet(SimpleBlock, [3,4,6,3],[64,128,256,512], flatten)

def ResNet50( flatten = True):
    return ResNet(BottleneckBlock, [3,4,6,3], [256,512,1024,2048], flatten)

def ResNet101( flatten = True):
    return ResNet(BottleneckBlock, [3,4,23,3],[256,512,1024,2048], flatten)
