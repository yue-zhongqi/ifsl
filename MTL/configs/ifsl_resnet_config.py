class Params():
    def __init__(self):
        self.is_param = True

# python main.py --config=mini_5_resnet_d --gpu=
def mini_5_resnet_d():
    param = Params()
    param.shot = 5
    param.test_iter = 2000
    param.test = True
    param.debug = False
    param.dataset = "miniImagenet"
    param.method = "simpleshot"
    param.model = "ResNet10"
    param.deconfound = True
    param.init_weights = "/data2/yuezhongqi/Model/simple_shot/miniImagenet/resnet10/model_best.pth.tar"
    param.meta_label = "exp31"
    
    ifsl_params = {
        "classifier": "single",
        "logit_fusion": "product",
        "fusion": "concat",
        "n_splits": 8,
        "sum_log": True,
        "lr": 0.005,
        "weight_decay": 0.001,
        "d_feature": "ed",
        "n_steps": 100,
        "ori_embedding_for_pd": False,
        "preprocess_before_split": "cl2n",
        "preprocess_after_split": "l2n",
        "is_cosine_feature": True,
        "normalize_before_center": True,
        "normalize_d": False,
        "normalize_ed": False
    }
    param.ifsl_params = ifsl_params

    param.learner = "DLearner"
    param.dacc = False
    return param

# python main.py --config=mini_1_resnet_d --gpu=
def mini_1_resnet_d():
    param = Params()
    param.shot = 1
    param.test_iter = 2000
    param.test = True
    param.debug = False
    param.dataset = "miniImagenet"
    param.method = "simpleshot"
    param.model = "ResNet10"
    param.deconfound = True
    param.init_weights = "/data2/yuezhongqi/Model/simple_shot/miniImagenet/resnet10/model_best.pth.tar"
    param.meta_label = "exp32"
    ifsl_params = {
        "classifier": "single",
        "logit_fusion": "product",
        "fusion": "concat",
        "n_splits": 8,
        "sum_log": True,
        "lr": 0.005,
        "weight_decay": 0.001,
        "d_feature": "ed",
        "n_steps": 100,
        "ori_embedding_for_pd": False,
        "preprocess_before_split": "cl2n",
        "preprocess_after_split": "l2n",
        "is_cosine_feature": True,
        "normalize_before_center": True,
        "normalize_d": False,
        "normalize_ed": False
    }
    param.ifsl_params = ifsl_params

    param.learner = "DLearner"
    param.dacc = False
    return param


# python main.py --config=tiered_5_resnet_d --gpu=
def tiered_5_resnet_d():
    param = Params()
    param.shot = 5
    param.test_iter = 2000
    param.test = True
    param.debug = False
    param.dataset = "tiered"
    param.method = "simpleshot"
    param.model = "ResNet10"
    param.deconfound = True
    param.init_weights = "/data2/yuezhongqi/Model/simple_shot/tiered/resnet10/model_best.pth.tar"
    param.meta_label = "tiered_base"
    
    ifsl_params = {
        "classifier": "single",
        "logit_fusion": "product",
        "fusion": "concat",
        "n_splits": 8,
        "sum_log": True,
        "lr": 0.005,
        "weight_decay": 0.001,
        "d_feature": "ed",
        "n_steps": 100,
        "ori_embedding_for_pd": False,
        "preprocess_before_split": "cl2n",
        "preprocess_after_split": "l2n",
        "is_cosine_feature": True,
        "normalize_before_center": True,
        "normalize_d": False,
        "normalize_ed": False
    }
    param.ifsl_params = ifsl_params

    param.learner = "DLearner"
    param.dacc = False
    return param

# python main.py --config=tiered_1_resnet_d --gpu=
def tiered_1_resnet_d():
    param = Params()
    param.shot = 1
    param.test_iter = 2000
    param.test = True
    param.debug = False
    param.dataset = "tiered"
    param.method = "simpleshot"
    param.model = "ResNet10"
    param.deconfound = True
    param.init_weights = "/data2/yuezhongqi/Model/simple_shot/tiered/resnet10/model_best.pth.tar"
    param.meta_label = "exp32"
    ifsl_params = {
        "classifier": "single",
        "logit_fusion": "product",
        "fusion": "concat",
        "n_splits": 8,
        "sum_log": True,
        "lr": 0.005,
        "weight_decay": 0.001,
        "d_feature": "ed",
        "n_steps": 100,
        "ori_embedding_for_pd": False,
        "preprocess_before_split": "cl2n",
        "preprocess_after_split": "l2n",
        "is_cosine_feature": True,
        "normalize_before_center": True,
        "normalize_d": False,
        "normalize_ed": False
    }
    param.ifsl_params = ifsl_params

    param.learner = "DLearner"
    param.dacc = False
    return param