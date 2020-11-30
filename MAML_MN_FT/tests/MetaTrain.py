from methods.MethodTester import MethodTester
from methods.VanillaMAML import VanillaMAML
from methods.DMAML import DMAML
from methods.VanillaMatchingNet import VanillaMatchingNet
from methods.DMatchingNet import DMatchingNet


class MetaTrain(MethodTester):
    def __init__(self, params):
        super(MetaTrain, self).__init__(params)

    ############################################# MAML Baseline ###########################################
    '''
    python main.py --dataset miniImagenet --method metatrain --train_aug --test maml5_resnet
    '''
    def maml5_resnet(self):
        config = {}
        config["dataset"] = "miniImagenet"
        config["model"] = "ResNet10"
        config["n_shot"] = 5
        config["method"] = "simpleshot"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 30

        config["params"] = {
            "approx": False,
            "update_step": 20,
            "n_task": 4,
            "lr": 0.005
        }
        self.meta_train(config, VanillaMAML, "maml0", debug=False, use_test=False, require_pretrain=False)

    '''
    python main.py --dataset miniImagenet --method metatrain --train_aug --test maml5_wrn
    '''
    def maml5_wrn(self):
        config = {}
        config["dataset"] = "miniImagenet"
        config["model"] = "wideres"
        config["n_shot"] = 5
        config["method"] = "simpleshotwide"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 30

        config["params"] = {
            "approx": False,
            "update_step": 20,
            "n_task": 4,
            "lr": 0.005
        }
        self.meta_train(config, VanillaMAML, "maml0", debug=False, use_test=False, require_pretrain=False)

    '''
    python main.py --dataset miniImagenet --method metatrain --train_aug --test maml1_resnet
    '''
    def maml1_resnet(self):
        config = {}
        config["dataset"] = "miniImagenet"
        config["model"] = "ResNet10"
        config["n_shot"] = 1
        config["method"] = "simpleshot"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 30

        config["params"] = {
            "approx": False,
            "update_step": 20,
            "n_task": 4,
            "lr": 0.005
        }
        self.meta_train(config, VanillaMAML, "maml0", debug=False, use_test=False, require_pretrain=False)

    '''
    python main.py --dataset miniImagenet --method metatrain --train_aug --test maml1_wrn
    '''
    def maml1_wrn(self):
        config = {}
        config["dataset"] = "miniImagenet"
        config["model"] = "wideres"
        config["n_shot"] = 1
        config["method"] = "simpleshotwide"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 30

        config["params"] = {
            "approx": False,
            "update_step": 20,
            "n_task": 4,
            "lr": 0.005
        }
        self.meta_train(config, VanillaMAML, "maml0", debug=False, use_test=False, require_pretrain=False)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test maml5_resnet_tiered
    '''
    def maml5_resnet_tiered(self):
        config = {}
        config["dataset"] = "tiered"
        config["model"] = "ResNet10"
        config["n_shot"] = 5
        config["method"] = "simpleshot"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        config["params"] = {
            "approx": False,
            "update_step": 20,
            "n_task": 4,
            "lr": 0.01
        }
        self.meta_train(config, VanillaMAML, "maml0", debug=False, use_test=False, require_pretrain=False)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test maml5_wrn_tiered
    '''
    def maml5_wrn_tiered(self):
        config = {}
        config["dataset"] = "tiered"
        config["model"] = "wideres"
        config["n_shot"] = 5
        config["method"] = "simpleshotwide"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        config["params"] = {
            "approx": False,
            "update_step": 20,
            "n_task": 4,
            "lr": 0.01
        }
        self.meta_train(config, VanillaMAML, "maml0", debug=False, use_test=False, require_pretrain=False)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test maml1_resnet_tiered
    '''
    def maml1_resnet_tiered(self):
        config = {}
        config["dataset"] = "tiered"
        config["model"] = "ResNet10"
        config["n_shot"] = 1
        config["method"] = "simpleshot"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        config["params"] = {
            "approx": False,
            "update_step": 20,
            "n_task": 4,
            "lr": 0.01
        }
        self.meta_train(config, VanillaMAML, "maml0", debug=False, use_test=False, require_pretrain=False)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test maml1_wrn_tiered
    '''
    def maml1_wrn_tiered(self):
        config = {}
        config["dataset"] = "tiered"
        config["model"] = "wideres"
        config["n_shot"] = 1
        config["method"] = "simpleshotwide"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        config["params"] = {
            "approx": False,
            "update_step": 20,
            "n_task": 4,
            "lr": 0.01
        }
        self.meta_train(config, VanillaMAML, "maml0", debug=False, use_test=False, require_pretrain=False)
    

    ############################################# MAML IFSL #############################################
    '''
    python main.py --dataset miniImagenet --model na --method metatrain --train_aug --test maml5_ifsl_resnet
    '''
    def maml5_ifsl_resnet(self):
        config = {}
        config["dataset"] = "miniImagenet"
        config["model"] = "ResNet10"
        config["n_shot"] = 5
        config["method"] = "simpleshot"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        config["params"] = {
            "approx": False,
            "update_step": 20,
            "n_task": 4,
            "lr": 0.01,
            "n_splits": 8,
            "d_feature": "ed",
            "fusion": "concat",
            "num_classes": 64,
            "sum_log": True,
            "classifier": "single",
            "logit_fusion": "product",
            "use_counterfactual": False,
            "x_zero": False,
            "temp": 5,
            "use_x_only": False,
            "preprocess_after_split": "none",
            "preprocess_before_split": "none",
            "is_cosine_feature": True,
            "normalize_before_center": False,
            "normalize_d": False,
            "normalize_ed": False
        }
        self.meta_train(config, DMAML, "maml1", debug=False, use_test=False, require_pretrain=True)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test maml5_ifsl_wrn
    '''
    def maml5_ifsl_wrn(self):
        config = {}
        config["dataset"] = "miniImagenet"
        config["model"] = "wideres"
        config["n_shot"] = 5
        config["method"] = "simpleshotwide"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        config["params"] = {
            "approx": False,
            "update_step": 20,
            "n_task": 4,
            "lr": 0.01,
            "n_splits": 8,
            "d_feature": "ed",
            "fusion": "concat",
            "num_classes": 64,
            "sum_log": True,
            "classifier": "single",
            "logit_fusion": "product",
            "use_counterfactual": False,
            "x_zero": False,
            "temp": 5,
            "use_x_only": False,
            "preprocess_after_split": "none",
            "preprocess_before_split": "none",
            "is_cosine_feature": True,
            "normalize_before_center": False,
            "normalize_d": False,
            "normalize_ed": False
        }
        self.meta_train(config, DMAML, "maml1", debug=False, use_test=False, require_pretrain=True)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test maml1_ifsl_resnet
    '''
    def maml1_ifsl_resnet(self):
        config = {}
        config["dataset"] = "miniImagenet"
        config["model"] = "ResNet10"
        config["n_shot"] = 1
        config["method"] = "simpleshot"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        config["params"] = {
            "approx": False,
            "update_step": 20,
            "n_task": 4,
            "lr": 0.01,
            "n_splits": 8,
            "d_feature": "ed",
            "fusion": "concat",
            "num_classes": 64,
            "sum_log": True,
            "classifier": "single",
            "logit_fusion": "product",
            "use_counterfactual": False,
            "x_zero": False,
            "temp": 5,
            "use_x_only": False,
            "preprocess_after_split": "none",
            "preprocess_before_split": "none",
            "is_cosine_feature": True,
            "normalize_before_center": False,
            "normalize_d": False,
            "normalize_ed": False
        }
        self.meta_train(config, DMAML, "maml1", debug=False, use_test=False, require_pretrain=True)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test maml1_ifsl_wrn
    '''
    def maml1_ifsl_wrn(self):
        config = {}
        config["dataset"] = "miniImagenet"
        config["model"] = "wideres"
        config["n_shot"] = 1
        config["method"] = "simpleshotwide"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        config["params"] = {
            "approx": False,
            "update_step": 20,
            "n_task": 4,
            "lr": 0.01,
            "n_splits": 8,
            "d_feature": "ed",
            "fusion": "concat",
            "num_classes": 64,
            "sum_log": True,
            "classifier": "single",
            "logit_fusion": "product",
            "use_counterfactual": False,
            "x_zero": False,
            "temp": 5,
            "use_x_only": False,
            "preprocess_after_split": "none",
            "preprocess_before_split": "none",
            "is_cosine_feature": True,
            "normalize_before_center": False,
            "normalize_d": False,
            "normalize_ed": False
        }
        self.meta_train(config, DMAML, "maml1", debug=False, use_test=False, require_pretrain=True)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test maml5_ifsl_resnet_tiered
    '''
    def maml5_ifsl_resnet_tiered(self):
        config = {}
        config["dataset"] = "tiered"
        config["model"] = "ResNet10"
        config["n_shot"] = 5
        config["method"] = "simpleshot"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        config["params"] = {
            "approx": False,
            "update_step": 20,
            "n_task": 4,
            "lr": 0.01,
            "n_splits": 10,
            "is_cosine_feature": True,
            "d_feature": "pd",
            "fusion": "concat",
            "num_classes": 351,
            "sum_log": True,
            "classifier": "single",
            "logit_fusion": "product",
            "use_counterfactual": False,
            "x_zero": False,
            "temp": 5,
            "use_x_only": True,
            "preprocess_after_split": "none",
            "preprocess_before_split": "none",
            "normalize_before_center": False,
            "normalize_d": False,
            "normalize_ed": False
        }
        self.meta_train(config, DMAML, "maml1", debug=False, use_test=False, require_pretrain=True)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test maml5_ifsl_wrn_tiered
    '''
    def maml5_ifsl_wrn_tiered(self):
        config = {}
        config["dataset"] = "tiered"
        config["model"] = "wideres"
        config["n_shot"] = 5
        config["method"] = "simpleshotwide"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        config["params"] = {
            "approx": False,
            "update_step": 20,
            "n_task": 4,
            "lr": 0.01,
            "n_splits": 10,
            "is_cosine_feature": True,
            "d_feature": "pd",
            "fusion": "concat",
            "num_classes": 351,
            "sum_log": True,
            "classifier": "single",
            "logit_fusion": "product",
            "use_counterfactual": False,
            "x_zero": False,
            "temp": 5,
            "use_x_only": True,
            "preprocess_after_split": "none",
            "preprocess_before_split": "none",
            "normalize_before_center": False,
            "normalize_d": False,
            "normalize_ed": False
        }
        self.meta_train(config, DMAML, "wrnbasev", debug=False, use_test=False, require_pretrain=True)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test maml1_ifsl_resnet_tiered
    '''
    def maml1_ifsl_resnet_tiered(self):
        config = {}
        config["dataset"] = "tiered"
        config["model"] = "ResNet10"
        config["n_shot"] = 1
        config["method"] = "simpleshot"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        config["params"] = {
            "approx": False,
            "update_step": 20,
            "n_task": 4,
            "lr": 0.01,
            "n_splits": 10,
            "is_cosine_feature": True,
            "d_feature": "pd",
            "fusion": "concat",
            "num_classes": 351,
            "sum_log": True,
            "classifier": "single",
            "logit_fusion": "product",
            "use_counterfactual": False,
            "x_zero": False,
            "temp": 5,
            "use_x_only": True,
            "preprocess_after_split": "none",
            "preprocess_before_split": "none",
            "normalize_before_center": False,
            "normalize_d": False,
            "normalize_ed": False
        }
        self.meta_train(config, DMAML, "maml1", debug=False, use_test=False, require_pretrain=True)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test maml1_ifsl_wrn_tiered
    '''
    def maml1_ifsl_wrn_tiered(self):
        config = {}
        config["dataset"] = "tiered"
        config["model"] = "wideres"
        config["n_shot"] = 1
        config["method"] = "simpleshotwide"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        config["params"] = {
            "approx": False,
            "update_step": 20,
            "n_task": 4,
            "lr": 0.01,
            "n_splits": 10,
            "is_cosine_feature": True,
            "d_feature": "pd",
            "fusion": "concat",
            "num_classes": 351,
            "sum_log": True,
            "classifier": "single",
            "logit_fusion": "product",
            "use_counterfactual": False,
            "x_zero": False,
            "temp": 5,
            "use_x_only": True,
            "preprocess_after_split": "none",
            "preprocess_before_split": "none",
            "normalize_before_center": False,
            "normalize_d": False,
            "normalize_ed": False
        }
        self.meta_train(config, DMAML, "maml1", debug=False, use_test=False, require_pretrain=True)


    ####################################### Matching Network ##########################################
    '''
    python main.py --method metatrain --train_aug --test mn5_resnet
    '''
    def mn5_resnet(self):
        config = {}
        config["dataset"] = "miniImagenet"
        config["model"] = "ResNet10"
        config["n_shot"] = 5
        config["method"] = "simpleshot"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        self.meta_train(config, VanillaMatchingNet, "mn0", debug=False, use_test=False)

    '''
    python main.py --method metatrain --train_aug --test mn5_wrn
    '''
    def mn5_wrn(self):
        config = {}
        config["dataset"] = "miniImagenet"
        config["model"] = "wideres"
        config["n_shot"] = 5
        config["method"] = "simpleshotwide"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        self.meta_train(config, VanillaMatchingNet, "mn0", debug=False, use_test=False)

    '''
    python main.py --method metatrain --train_aug --test mn1_resnet
    '''
    def mn1_resnet(self):
        config = {}
        config["dataset"] = "miniImagenet"
        config["model"] = "ResNet10"
        config["n_shot"] = 1
        config["method"] = "simpleshot"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        self.meta_train(config, VanillaMatchingNet, "mn0", debug=False, use_test=False)

    '''
    python main.py --method metatrain --train_aug --test mn1_wrn
    '''
    def mn1_wrn(self):
        config = {}
        config["dataset"] = "miniImagenet"
        config["model"] = "wideres"
        config["n_shot"] = 1
        config["method"] = "simpleshotwide"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        self.meta_train(config, VanillaMatchingNet, "mn0", debug=False, use_test=False)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test mn5_resnet_tiered
    '''
    def mn5_resnet_tiered(self):
        config = {}
        config["dataset"] = "tiered"
        config["model"] = "ResNet10"
        config["n_shot"] = 5
        config["method"] = "simpleshot"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        self.meta_train(config, VanillaMatchingNet, "mn0", debug=False, use_test=False)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test mn5_wrn_tiered
    '''
    def mn5_wrn_tiered(self):
        config = {}
        config["dataset"] = "tiered"
        config["model"] = "wideres"
        config["n_shot"] = 5
        config["method"] = "simpleshotwide"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        self.meta_train(config, VanillaMatchingNet, "mn0", debug=False, use_test=False)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test mn1_resnet_tiered
    '''
    def mn1_resnet_tiered(self):
        config = {}
        config["dataset"] = "tiered"
        config["model"] = "ResNet10"
        config["n_shot"] = 1
        config["method"] = "simpleshot"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        self.meta_train(config, VanillaMatchingNet, "mn0", debug=False, use_test=False)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test mn1_wrn_tiered
    '''
    def mn1_wrn_tiered(self):
        config = {}
        config["dataset"] = "tiered"
        config["model"] = "wideres"
        config["n_shot"] = 1
        config["method"] = "simpleshotwide"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        self.meta_train(config, VanillaMatchingNet, "mn0", debug=False, use_test=False)

    ####################################### Matching Network + IFSL ###########################################
    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test mn5_ifsl_resnet
    '''
    def mn5_ifsl_resnet(self):
        config = {}
        config["dataset"] = "miniImagenet"
        config["model"] = "ResNet10"
        config["n_shot"] = 5
        config["method"] = "simpleshot"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100
        config["params"] = {
            "n_splits": 16,
            "d_feature": "ed",
            "fusion": "concat",
            "logit_fusion": "product",
            "sum_log": False,
            "num_classes": 64,
            "classifier": "single",
            "use_counterfactual": False,
            "x_zero": False,
            "temp": 100,
            "use_x_only": False,
            "preprocess_before_split": "none",
            "preprocess_after_split": "none",
            "is_cosine_feature": True,
            "normalize_before_center": False,
            "normalize_d": False,
            "normalize_ed": False
        }
        self.meta_train(config, DMatchingNet, "mn1", debug=False, use_test=False, require_pretrain=True)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test mn5_ifsl_wrn
    '''
    def mn5_ifsl_wrn(self):
        config = {}
        config["dataset"] = "miniImagenet"
        config["model"] = "wideres"
        config["n_shot"] = 5
        config["method"] = "simpleshotwide"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100
        config["params"] = {
            "n_splits": 16,
            "d_feature": "ed",
            "fusion": "concat",
            "logit_fusion": "product",
            "sum_log": False,
            "num_classes": 64,
            "classifier": "single",
            "use_counterfactual": False,
            "x_zero": False,
            "temp": 100,
            "use_x_only": False,
            "preprocess_before_split": "none",
            "preprocess_after_split": "none",
            "is_cosine_feature": True,
            "normalize_before_center": False,
            "normalize_d": False,
            "normalize_ed": False
        }
        self.meta_train(config, DMatchingNet, "mn1", debug=False, use_test=False, require_pretrain=True)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test mn1_ifsl_resnet
    '''
    def mn1_ifsl_resnet(self):
        config = {}
        config["dataset"] = "miniImagenet"
        config["model"] = "ResNet10"
        config["n_shot"] = 1
        config["method"] = "simpleshot"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100
        config["params"] = {
            "n_splits": 16,
            "d_feature": "ed",
            "fusion": "concat",
            "logit_fusion": "product",
            "sum_log": False,
            "num_classes": 64,
            "classifier": "single",
            "use_counterfactual": False,
            "x_zero": False,
            "temp": 100,
            "use_x_only": False,
            "preprocess_before_split": "none",
            "preprocess_after_split": "none",
            "is_cosine_feature": True,
            "normalize_before_center": False,
            "normalize_d": False,
            "normalize_ed": False
        }
        self.meta_train(config, DMatchingNet, "mn1", debug=False, use_test=False, require_pretrain=True)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test mn1_ifsl_wrn
    '''
    def mn1_ifsl_wrn(self):
        config = {}
        config["dataset"] = "miniImagenet"
        config["model"] = "wideres"
        config["n_shot"] = 1
        config["method"] = "simpleshotwide"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100
        config["params"] = {
            "n_splits": 16,
            "d_feature": "ed",
            "fusion": "concat",
            "logit_fusion": "product",
            "sum_log": False,
            "num_classes": 64,
            "classifier": "single",
            "use_counterfactual": False,
            "x_zero": False,
            "temp": 100,
            "use_x_only": False,
            "preprocess_before_split": "none",
            "preprocess_after_split": "none",
            "is_cosine_feature": True,
            "normalize_before_center": False,
            "normalize_d": False,
            "normalize_ed": False
        }
        self.meta_train(config, DMatchingNet, "mn1", debug=False, use_test=False, require_pretrain=True)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test mn5_ifsl_resnet_tiered
    '''
    def mn5_ifsl_resnet_tiered(self):
        config = {}
        config["dataset"] = "tiered"
        config["model"] = "ResNet10"
        config["n_shot"] = 5
        config["method"] = "simpleshot"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        config["params"] = {
            "n_splits": 8,
            "d_feature": "ed",
            "fusion": "concat",
            "logit_fusion": "product",
            "sum_log": False,
            "num_classes": 351,
            "classifier": "single",
            "use_counterfactual": False,
            "x_zero": False,
            "temp": 100,
            "use_x_only": False,
            "preprocess_before_split": "none",
            "preprocess_after_split": "none",
            "is_cosine_feature": True,
            "normalize_before_center": False,
            "normalize_d": False,
            "normalize_ed": False
        }
        self.meta_train(config, DMatchingNet, "mn1", debug=False, use_test=False, require_pretrain=True)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test mn5_ifsl_wrn_tiered
    '''
    def mn5_ifsl_wrn_tiered(self):
        config = {}
        config["dataset"] = "tiered"
        config["model"] = "wideres"
        config["n_shot"] = 5
        config["method"] = "simpleshotwide"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        config["params"] = {
            "n_splits": 16,
            "d_feature": "ed",
            "fusion": "concat",
            "logit_fusion": "product",
            "sum_log": False,
            "num_classes": 351,
            "classifier": "single",
            "use_counterfactual": False,
            "x_zero": False,
            "temp": 100,
            "use_x_only": False,
            "preprocess_before_split": "none",
            "preprocess_after_split": "none",
            "is_cosine_feature": True,
            "normalize_before_center": False,
            "normalize_d": False,
            "normalize_ed": False
        }
        self.meta_train(config, DMatchingNet, "mn1", debug=False, use_test=False, require_pretrain=True)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test mn1_ifsl_resnet_tiered
    '''
    def mn1_ifsl_resnet_tiered(self):
        config = {}
        config["dataset"] = "tiered"
        config["model"] = "ResNet10"
        config["n_shot"] = 1
        config["method"] = "simpleshot"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        config["params"] = {
            "n_splits": 8,
            "d_feature": "ed",
            "fusion": "concat",
            "logit_fusion": "product",
            "sum_log": False,
            "num_classes": 351,
            "classifier": "single",
            "use_counterfactual": False,
            "x_zero": False,
            "temp": 100,
            "use_x_only": False,
            "preprocess_before_split": "none",
            "preprocess_after_split": "none",
            "is_cosine_feature": True,
            "normalize_before_center": False,
            "normalize_d": False,
            "normalize_ed": False
        }
        self.meta_train(config, DMatchingNet, "mn1", debug=False, use_test=False, require_pretrain=True)

    '''
    python new_test.py --dataset miniImagenet --model na --method metatrain --train_aug --test mn1_ifsl_wrn_tiered
    '''
    def mn1_ifsl_wrn_tiered(self):
        config = {}
        config["dataset"] = "tiered"
        config["model"] = "wideres"
        config["n_shot"] = 1
        config["method"] = "simpleshotwide"
        config["train_episodes"] = 100
        config["val_episodes"] = 2000
        config["end_epoch"] = 100

        config["params"] = {
            "n_splits": 16,
            "d_feature": "ed",
            "fusion": "concat",
            "logit_fusion": "product",
            "sum_log": False,
            "num_classes": 351,
            "classifier": "single",
            "use_counterfactual": False,
            "x_zero": False,
            "temp": 100,
            "use_x_only": False,
            "preprocess_before_split": "none",
            "preprocess_after_split": "none",
            "is_cosine_feature": True,
            "normalize_before_center": False,
            "normalize_d": False,
            "normalize_ed": False
        }
        self.meta_train(config, DMatchingNet, "mn1", debug=False, use_test=False, require_pretrain=True)