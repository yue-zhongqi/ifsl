import torch.nn as nn
import torch
import torch.nn.functional as F


def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


class ProtoNet(nn.Module):

    def __init__(self, feature_net, args=None):
        super().__init__()
        self.encoder = feature_net
        if args is None:
            self.train_info = [30, 1, 15, 'euclidean']
            # self.val_info = [5, 1, 15]
        else:
            self.train_info = [args.meta_train_way, args.meta_train_shot, args.meta_train_query, args.meta_train_metric]
            # self.val_info = [args.meta_val_way, args.meta_val_shot, args.meta_val_query]

    def forward(self, data, _=False):
        if not self.training:
            return self.encoder(data, True)
        way, shot, query, metric_name = self.train_info
        proto, _ = self.encoder(data, True)
        shot_proto, query_proto = proto[:shot * way], proto[shot * way:]
        shot_proto = shot_proto.reshape(way, shot, -1).mean(1)
        logits = -get_metric(metric_name)(shot_proto, query_proto)

        return logits
