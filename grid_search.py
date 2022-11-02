import argparse
import torch
import wandb
wandb.login()
import yaml
import os
import uuid

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.maple
import trainers.independentVL
import trainers.vpt

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    
    if args.text_weight:
        cfg.TRAINER.IVLP.TEXT_WEIGHT = args.text_weight
    
    if args.visual_weight:
        cfg.TRAINER.IVLP.VISUAL_WEIGHT = args.visual_weight

    if args.n_ins:
        cfg.DATALOADER.TRAIN_X.N_INS = args.n_ins
    
    if args.batch_size:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    
    if args.lr:
        cfg.OPTIM.LR = args.lr

    if args.epochs:
        cfg.OPTIM.MAX_EPOCH = args.epochs
    
    if args.num_shots:
        cfg.DATASET.NUM_SHOTS = args.num_shots
    
    if args.subsample_classes:
        cfg.DATASET.SUBSAMPLE_CLASSES = args.subsample_classes
    
    if args.visual_loss:
        cfg.TRAINER.IVLP.VISUAL_LOSS = args.visual_loss

    if args.n_ctx_vision:
        cfg.TRAINER.IVLP.N_CTX_VISION = args.n_ctx_vision
    
    if args.n_ctx_text:
        cfg.TRAINER.IVLP.N_CTX_TEXT = args.n_ctx_text
    
    if args.prompt_depth_vision:
        cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = args.prompt_depth_vision

    if args.prompt_depth_text:
        cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = args.prompt_depth_text


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9 # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.IVLP.TEXT_WEIGHT = 0.0
    cfg.TRAINER.IVLP.VISUAL_WEIGHT = 0.0
    cfg.TRAINER.IVLP.VISUAL_LOSS = "multi_similarity_loss"
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for only vision side prompting
    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1  # if set to 1, will represent shallow vision prompting only
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    # cfg.merge_from_list(args.opts)

    # cfg.freeze()

    return cfg


def main(args):
    with open('./grid_search.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    wandb.init(config=config, sync_tensorboard=True)
    os.environ["WANDB_DIR"] = os.path.abspath("/nobackup3/yiwei/wandb")

    # args.root = wandb.config.root
    # args.seed = wandb.config.seed
    # args.trainer = wandb.config.trainer
    # args.configfile = wandb.config.configfile 
    # args.datasetconfigfile = wandb.config.datasetconfigfile
    args.output_dir = os.path.join(args.output_dir, wandb.run.id)
    # args.textweight = wandb.config.textweight 
    # args.visualweight = wandb.config.visualweight 

    # args.opts = ["DATALOADER.N_INS", wandb.config.nins, "DATALOADER.BATCH_SIZE", wandb.config.batchsize,
    # "OPTIM.LR", wandb.config.lr, "OPTIM.MAX_EPOCH", wandb.config.epochs, "DATASET.NUM_SHOTS", 16,
    # "DATASET.SUBSAMPLE_CLASSES", "base", ]

    cfg = setup_cfg(args)

    # cfg.DATALOADER.N_INS = wandb.config.nins
    # cfg.DATALOADER.BATCH_SIZE = wandb.config.batchsize
    # cfg.OPTIM.LR = wandb.config.lr
    # cfg.OPTIM.MAX_EPOCH = wandb.config.epochs
    # cfg.DATASET.NUM_SHOTS = 16
    # cfg.DATASET.SUBSAMPLE_CLASSES = "base"

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    output_dir = cfg.OUTPUT_DIR
    cfg.OUTPUT_DIR = os.path.join(output_dir, str(cfg.SEED))
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)
    base_accuracy_1 = trainer.train()
    cfg.DATASET.SUBSAMPLE_CLASSES = "new"
    novel_accuracy_1 = trainer.test()

    cfg.SEED = cfg.SEED + 1
    set_random_seed(cfg.SEED)
    cfg.OUTPUT_DIR = os.path.join(output_dir, str(cfg.SEED))
    cfg.DATASET.SUBSAMPLE_CLASSES = "base"
    setup_logger(cfg.OUTPUT_DIR)
    trainer = build_trainer(cfg)
    base_accuracy_2 = trainer.train()
    cfg.DATASET.SUBSAMPLE_CLASSES = "new"
    novel_accuracy_2 = trainer.test()

    cfg.SEED = cfg.SEED + 1
    set_random_seed(cfg.SEED)
    cfg.OUTPUT_DIR = os.path.join(output_dir, str(cfg.SEED))
    cfg.DATASET.SUBSAMPLE_CLASSES = "base"
    setup_logger(cfg.OUTPUT_DIR)
    trainer = build_trainer(cfg)
    base_accuracy_3 = trainer.train()
    cfg.DATASET.SUBSAMPLE_CLASSES = "new"
    novel_accuracy_3 = trainer.test()

    mean_accuracy = 0.1666 * base_accuracy_1 + 0.1666 * novel_accuracy_1 + 0.1666 * base_accuracy_2 + 0.1666 * novel_accuracy_2 + 0.1666 * base_accuracy_3 + 0.1666 * novel_accuracy_3
    wandb.log({"mean_accuracy": mean_accuracy})
    # train_accuracy = 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "--text-weight", type=float, help="weight of text losses"
    )
    parser.add_argument(
        "--visual-weight", type=float, help="weight of visual losses"
    )
    parser.add_argument(
        "--batch-size", type=int, help="batch size"
    )
    # parser.add_argument(
    #     "--config-file", type=str, help="config file"
    # )
    # parser.add_argument(
    #     "--dataset-config-file", type=str, help="dataset config file"
    # )
    parser.add_argument(
        "--epochs", type=int, help="epochs"
    )
    parser.add_argument(
        "--lr", type=float, help="lr"
    )
    parser.add_argument(
        "--n-ins", type=int, help="n_ins"
    )
    parser.add_argument(
        "--num-shots", type=int, help="num_shots"
    )
    parser.add_argument(
        "--subsample-classes", type=str, help="subsample_classes"
    )
    parser.add_argument(
        "--visual-loss", type=str, help="visual_loss"
    )
    parser.add_argument(
        "--n-ctx-vision", type=int, help="n-ctx-vision"
    )
    parser.add_argument(
        "--n-ctx-text", type=int, help="n-ctx-text"
    )
    parser.add_argument(
        "--n-ctx-text", type=int, help="n-ctx-text"
    )
    parser.add_argument(
        "--prompt-depth-vision", type=int, help="prompt-depth-vision"
    )
    parser.add_argument(
        "--prompt-depth-text", type=int, help="prompt-depth-text"
    )
    # parser.add_argument(
    #     "--output-dir", type=str, help="output base dir"
    # )
    # parser.add_argument(
    #     "opts",
    #     default=None,
    #     nargs=argparse.REMAINDER,
    #     help="modify config options using the command-line",
    # )
    args = parser.parse_args()
    main(args)
