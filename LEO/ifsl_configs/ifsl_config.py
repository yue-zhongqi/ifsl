class Config():
    def __init__(self):
        self.is_config = True

def mini_5_resnet_ifsl():
    config = Config()
    config.shot = 5
    config.test = True
    config.debug = False
    config.dataset = "miniImagenet"
    config.method = "simpleshot"
    config.model = "ResNet10"
    config.deconfound = True
    config.meta_label = "ifsl"
    # IFSL parameters
    config.n_splits = 8
    config.fusion = "+"
    config.classifier = "single"
    config.num_classes = 64
    config.logit_fusion = "product"
    config.use_x_only = False
    config.preprocess_before_split = "cl2n"
    config.preprocess_after_split = "l2n"
    config.is_cosine_feature = True
    config.normalize_before_center = True
    config.normalize_d = False
    config.normalize_ed = False
    # config.outer_lr = 1.51024e-4
    return config

def mini_1_resnet_ifsl():
    config = Config()
    config.shot = 1
    config.test = True
    config.debug = False
    config.dataset = "miniImagenet"
    config.method = "simpleshot"
    config.model = "ResNet10"
    config.deconfound = True
    config.meta_label = "ifsl"
    # IFSL parameters
    config.n_splits = 8
    config.fusion = "+"
    config.classifier = "single"
    config.num_classes = 64
    config.logit_fusion = "product"
    config.use_x_only = False
    config.preprocess_before_split = "cl2n"
    config.preprocess_after_split = "l2n"
    config.is_cosine_feature = True
    config.normalize_before_center = True
    config.normalize_d = False
    config.normalize_ed = False
    return config

def mini_5_wrn_ifsl():
    config = Config()
    config.shot = 5
    config.test = True
    config.debug = False
    config.dataset = "miniImagenet"
    config.method = "simpleshotwide"
    config.model = "wideres"
    config.deconfound = True
    config.meta_label = "ifsl_lr"
    # IFSL parameters
    config.n_splits = 8
    config.fusion = "+"
    config.classifier = "single"
    config.num_classes = 64
    config.logit_fusion = "product"
    config.use_x_only = False
    config.preprocess_before_split = "cl2n"
    config.preprocess_after_split = "l2n"
    config.is_cosine_feature = True
    config.normalize_before_center = True
    config.normalize_d = False
    config.normalize_ed = False
    config.outer_lr = 2.61024e-4
    return config

def mini_1_wrn_ifsl():
    config = Config()
    config.shot = 1
    config.test = True
    config.debug = False
    config.dataset = "miniImagenet"
    config.method = "simpleshotwide"
    config.model = "wideres"
    config.deconfound = True
    config.meta_label = "ifsl_lr"
    # IFSL parameters
    config.n_splits = 8
    config.fusion = "+"
    config.classifier = "single"
    config.num_classes = 64
    config.logit_fusion = "product"
    config.use_x_only = False
    config.preprocess_before_split = "cl2n"
    config.preprocess_after_split = "l2n"
    config.is_cosine_feature = True
    config.normalize_before_center = True
    config.normalize_d = False
    config.normalize_ed = False
    config.outer_lr = 1.51024e-4
    return config

def tiered_5_resnet_ifsl():
    config = Config()
    config.shot = 5
    config.test = True
    config.debug = False
    config.dataset = "tiered"
    config.method = "simpleshot"
    config.model = "ResNet10"
    config.deconfound = True
    config.meta_label = "split"
    # IFSL parameters
    config.n_splits = 8
    config.fusion = "+"
    config.classifier = "single"
    config.num_classes = 351
    config.logit_fusion = "product"
    config.use_x_only = False
    config.preprocess_before_split = "cl2n"
    config.preprocess_after_split = "l2n"
    config.is_cosine_feature = True
    config.normalize_before_center = True
    config.normalize_d = False
    config.normalize_ed = False
    config.outer_lr = 2.669053e-4
    return config

def tiered_1_resnet_ifsl():
    config = Config()
    config.shot = 1
    config.test = True
    config.debug = False
    config.dataset = "tiered"
    config.method = "simpleshot"
    config.model = "ResNet10"
    config.deconfound = True
    config.meta_label = "ifsl"
    # IFSL parameters
    config.n_splits = 8
    config.fusion = "+"
    config.classifier = "single"
    config.num_classes = 351
    config.logit_fusion = "product"
    config.use_x_only = False
    config.preprocess_before_split = "cl2n"
    config.preprocess_after_split = "l2n"
    config.is_cosine_feature = True
    config.normalize_before_center = True
    config.normalize_d = False
    config.normalize_ed = False
    config.outer_lr = 2.669053e-4
    return config

def tiered_5_wrn_ifsl():
    config = Config()
    config.shot = 5
    config.test = True
    config.debug = False
    config.dataset = "tiered"
    config.method = "simpleshotwide"
    config.model = "wideres"
    config.deconfound = True
    config.meta_label = "ifsl"
    # IFSL parameters
    config.n_splits = 8
    config.fusion = "+"
    config.classifier = "single"
    config.num_classes = 351
    config.logit_fusion = "product"
    config.use_x_only = False
    config.preprocess_before_split = "cl2n"
    config.preprocess_after_split = "l2n"
    config.is_cosine_feature = True
    config.normalize_before_center = True
    config.normalize_d = False
    config.normalize_ed = False
    return config

def tiered_1_wrn_ifsl():
    config = Config()
    config.shot = 1
    config.test = True
    config.debug = False
    config.dataset = "tiered"
    config.method = "simpleshotwide"
    config.model = "wideres"
    config.deconfound = True
    config.meta_label = "ifsl"
    # IFSL parameters
    config.n_splits = 8
    config.fusion = "+"
    config.classifier = "single"
    config.num_classes = 351
    config.logit_fusion = "product"
    config.use_x_only = False
    config.preprocess_before_split = "cl2n"
    config.preprocess_after_split = "l2n"
    config.is_cosine_feature = True
    config.normalize_before_center = True
    config.normalize_d = False
    config.normalize_ed = False
    return config