import os
import sys

from detectron2.config import get_cfg
from detectron2.engine import default_setup, launch
from detectron2.utils.logger import setup_logger

# --- add MaskDINO repo to Python path ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MASKDINO_DIR = os.path.join(THIS_DIR, "MaskDINO")
sys.path.append(MASKDINO_DIR)

# --- add the ops dir so MultiScaleDeformableAttention.pyd is importable ---
OPS_DIR = os.path.join(
    MASKDINO_DIR,
    "maskdino",
    "modeling",
    "pixel_decoder",
    "ops",
)
if OPS_DIR not in sys.path:
    sys.path.append(OPS_DIR)

from maskdino.config import add_maskdino_config
from datasets.register_drone_coco import register_drone_coco
from MaskDINO.train_net import Trainer


def setup_cfg(args):
    cfg = get_cfg()
    add_maskdino_config(cfg)

    # point to your custom yaml
    cfg.merge_from_file(
        os.path.join(
            MASKDINO_DIR,
            "configs",
            "coco",
            "instance-segmentation",
            "maskdino_R50_drone.yaml",
        )
    )
    cfg.merge_from_list(args.opts)

    # (optional) enforce these in case the yaml is edited later
    cfg.DATASETS.TRAIN = ("drone_train",)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 8

    cfg.OUTPUT_DIR = os.path.join(THIS_DIR, "output", "drone_maskdino")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


class DroneTrainer(Trainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        # you can add a COCO evaluator later when you have a val set
        return None


def main(args):
    setup_logger()

    # 1) register the dataset
    register_drone_coco(project_root=THIS_DIR)

    # 2) build config
    cfg = setup_cfg(args)
    default_setup(cfg, args)

    # 3) train
    trainer = DroneTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MaskDINO on drone dataset")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    # launch distributed (even if num_gpus=1)
    launch(
        main,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(args,),
    )
