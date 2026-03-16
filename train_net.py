# Root-level train_net wrapper — safe to modify without touching MaskDINO/ folder
# This file mirrors MaskDINO/train_net.py behavior but ensures TEST.IMS_PER_BATCH
# exists before config merging so CLI overrides for TEST.* won't raise KeyError.
# It is intended to be used by `scripts/launch_maskdino.py` if present in repo root.
try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

# Ensure `warnings` is available and silence known AMP/autocast deprecation spam
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast.*",
    category=FutureWarning,
)

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, CfgNode as CN
from detectron2.data import MetadataCatalog, build_detection_train_loader

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskDINO
from maskdino import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskdino_config,
    DetrDatasetMapper,
)
import random
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer,
)
import weakref
import glob


class PruneCheckpointsHook(hooks.HookBase):
    """Hook that prunes old checkpoints in the output directory.

    Keeps only the most-recent `keep` checkpoint files named `model_*.pth`.
    Runs every `check_period` iterations (default 10) on the main process.
    """
    def __init__(self, keep: int = 3, check_period: int = 10):
        self.keep = keep
        self.check_period = check_period

    def after_step(self):
        # Only run pruning on main process
        if not comm.is_main_process():
            return
        # Reduce overhead by checking periodically
        try:
            it = int(self.trainer.iter)
        except Exception:
            return
        if it % self.check_period != 0:
            return

        # Determine output dir
        output_dir = None
        try:
            output_dir = self.trainer.cfg.OUTPUT_DIR
        except Exception:
            pass
        if not output_dir:
            return

        pattern = os.path.join(output_dir, "model_*.pth")
        files = glob.glob(pattern)
        if len(files) <= self.keep:
            return

        # Sort by modification time (oldest first)
        files.sort(key=lambda f: os.path.getmtime(f))
        to_delete = files[: len(files) - self.keep]
        for f in to_delete:
            try:
                os.remove(f)
            except Exception:
                pass


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        kwargs = {
            'trainer': weakref.proxy(self),
        }
        self.checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, **kwargs)
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())
        self.checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, **kwargs)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder)
            )
        if evaluator_type == "coco":
            # Enable distributed evaluation so worker processes contribute
            # predictions and results are aggregated across the cluster.
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, distributed=True))
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON:
                # Aggregate panoptic evaluation across distributed workers
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder, distributed=True))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
            # InstanceSegEvaluator from MaskDINO may accept distributed flag in newer versions;
            # if it does, pass `distributed=True` to ensure proper aggregation.
            try:
                evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder, distributed=True))
            except TypeError:
                evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        if evaluator_type == "cityscapes_instance":
            assert torch.cuda.device_count() > comm.get_rank(), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert torch.cuda.device_count() > comm.get_rank(), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON:
                assert torch.cuda.device_count() > comm.get_rank(), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
                assert torch.cuda.device_count() > comm.get_rank(), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(f"no Evaluator for the dataset {dataset_name} with the type {evaluator_type}")
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_detr":
            mapper = DetrDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def build_hooks(self):
        # Use DefaultTrainer hooks and append a checkpoint-pruning hook
        hooks_list = DefaultTrainer.build_hooks(self)
        # If the user explicitly requested no evaluation via the `no:eval`
        # token, disable Detectron2's EvalHook. We check `cfg.NO_EVAL` so
        # evaluation is only disabled when explicitly requested.
        try:
            if getattr(self.cfg, "NO_EVAL", False):
                hooks_list = [h for h in hooks_list if h.__class__.__name__ != "EvalHook"]
        except Exception:
            pass
        # Append prune hook that keeps only the latest 3 checkpoints
        hooks_list.append(PruneCheckpointsHook(keep=3))
        return hooks_list

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED
        defaults = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if ("relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    # Ensure TEST keys exist so merging user configs with TEST.* doesn't raise
    # a YACS KeyError when the default cfg lacks those fields.
    if not hasattr(cfg, "TEST"):
        cfg.TEST = CN()
    if not hasattr(cfg.TEST, "IMS_PER_BATCH"):
        cfg.TEST.IMS_PER_BATCH = 1
    cfg.merge_from_file(args.config_file)
    # Filter out any standalone CLI flags that may have been injected into
    # `args.opts` (e.g., '--resume') which would break cfg.merge_from_list
    safe_opts = []
    i = 0
    no_eval_flag = False
    while i < len(args.opts):
        if args.opts[i] == '--resume':
            args.resume = True
            i += 1
            continue
        # Support a standalone token 'no_eval' to disable periodic evaluation
        if args.opts[i] == 'no_eval':
            no_eval_flag = True
            i += 1
            continue
        safe_opts.append(args.opts[i])
        i += 1
    cfg.merge_from_list(safe_opts)
    # If user requested 'no:eval', set a cfg flag so Trainer can
    # intentionally disable EvalHook without relying on TEST.EVAL_PERIOD.
    try:
        if no_eval_flag:
            cfg.NO_EVAL = True
            if not hasattr(cfg, 'TEST'):
                cfg.TEST = CN()
            cfg.TEST.EVAL_PERIOD = 0
    except Exception:
        pass
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    return cfg


def main(args):
    cfg = setup(args)
    print("Command cfg:", cfg)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    # print(cfg)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--EVAL_FLAG', type=int, default=1)
    args = parser.parse_args()
    port = random.randint(1000, 20000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port)
    print("Command Line Args:", args)
    print("pwd:", os.getcwd())
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
