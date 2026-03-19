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
import torch.distributed as dist
import weakref
import glob

from detectron2.checkpoint import DetectionCheckpointer

class AMPCheckpointer(DetectionCheckpointer):
    def __init__(self, model, save_dir="", *, trainer=None, optimizer=None, scheduler=None, scaler=None, **kwargs):
        # Accept an optional `trainer` kwarg for compatibility with caller sites
        # that pass a `trainer` proxy. We don't forward it to the parent.
        self.trainer = trainer
        super().__init__(model, save_dir, optimizer=optimizer, scheduler=scheduler)
        self.scaler = scaler

    def save(self, name, **kwargs):
        if self.scaler is not None:
            kwargs["scaler"] = self.scaler.state_dict()
        super().save(name, **kwargs)

    def load(self, path, *args, **kwargs):
        checkpoint = super().load(path, *args, **kwargs)
        if self.scaler is not None and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        return checkpoint
    
    
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
        self.scaler = torch.amp.GradScaler() if cfg.SOLVER.AMP.ENABLED else None

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        kwargs = {
            'trainer': weakref.proxy(self),
        }
        
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.checkpointer = AMPCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
            optimizer=optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
        )
        self.register_hooks(self.build_hooks())

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
        # Insert synchronization and duplicate-work checks around evaluation
        try:
            class PrePostEvalSyncHook(hooks.HookBase):
                """Ensure ranks synchronize before/after evaluation and perform
                lightweight checks to detect duplicate work (sampler type and RNG state).
                """
                def before_step(self):
                    try:
                        cfg = getattr(self.trainer, 'cfg', None)
                        eval_period = 0
                        if cfg is not None:
                            eval_period = int(getattr(cfg.TEST, 'EVAL_PERIOD', 0) or 0)
                        cur_iter = int(getattr(self.trainer, 'iter', 0))
                        # If the next iteration will trigger evaluation, sync now
                        if eval_period > 0 and ((cur_iter + 1) % eval_period) == 0:
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                            try:
                                comm.synchronize()
                            except Exception:
                                pass
                            # Duplicate-work checks: sampler type and RNG fingerprints
                            try:
                                world_size = comm.get_world_size()
                                if world_size > 1 and dist.is_initialized():
                                    # sampler name
                                    sampler_name = None
                                    try:
                                        dl = getattr(self.trainer._trainer, 'data_loader', None)
                                        if dl is not None and hasattr(dl, 'sampler'):
                                            sampler_name = dl.sampler.__class__.__name__
                                    except Exception:
                                        sampler_name = None

                                    # RNG fingerprints
                                    try:
                                        import numpy as _np
                                        py_state = random.getstate()
                                        py_digest = hash(str(py_state[1][:10]))
                                    except Exception:
                                        py_digest = None
                                    try:
                                        np_state = None
                                        if 'numpy' in globals():
                                            np_state = _np.random.get_state()
                                            np_digest = hash(str(np_state[1][:10]))
                                        else:
                                            np_digest = None
                                    except Exception:
                                        np_digest = None
                                    try:
                                        torch_digest = None
                                        tr = torch.get_rng_state()
                                        try:
                                            torch_digest = hash(tr.cpu().numpy().tobytes())
                                        except Exception:
                                            torch_digest = hash(str(tr))
                                    except Exception:
                                        torch_digest = None

                                    payload = {'sampler': sampler_name, 'py': py_digest, 'np': np_digest, 'torch': torch_digest}
                                    try:
                                        gathered = [None] * world_size
                                        dist.all_gather_object(gathered, payload)
                                        # analyze gathered for obvious duplicates
                                        samplers = [g.get('sampler') for g in gathered if g]
                                        if any(s is None for s in samplers) is False:
                                            # if not using DistributedSampler, warn
                                            if not all(s == 'DistributedSampler' for s in samplers):
                                                if comm.is_main_process():
                                                    logging.getLogger('detectron2').warning(
                                                        'Sampler names across ranks: %s. Consider using DistributedSampler to avoid duplicated work.' % samplers
                                                    )
                                        # check RNG duplicates
                                        pylist = [g.get('py') for g in gathered if g]
                                        torchlist = [g.get('torch') for g in gathered if g]
                                        if len(set(pylist)) < len(pylist) or len(set(torchlist)) < len(torchlist):
                                            if comm.is_main_process():
                                                logging.getLogger('detectron2').warning(
                                                    'Detected identical RNG fingerprints across ranks. This may cause duplicated data augmentations/work. Ensure each rank has a unique seed.'
                                                )
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                    except Exception:
                        pass

                def after_step(self):
                    try:
                        cfg = getattr(self.trainer, 'cfg', None)
                        eval_period = 0
                        if cfg is not None:
                            eval_period = int(getattr(cfg.TEST, 'EVAL_PERIOD', 0) or 0)
                        cur_iter = int(getattr(self.trainer, 'iter', 0))
                        # If we just ran an evaluation, sync again to ensure clean state
                        if eval_period > 0 and (cur_iter % eval_period) == 0:
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                            try:
                                comm.synchronize()
                            except Exception:
                                pass
                    except Exception:
                        pass

            # add hook near start so its before_step runs before EvalHook's after_step
            hooks_list.insert(0, PrePostEvalSyncHook())
            # Also insert explicit pre/post barrier hooks around any EvalHook
            # instances to force an explicit comm/ CUDA sync before and after
            # evaluation. This is a lightweight, non-invasive way to ensure
            # all ranks have reached the same point when EvalHook runs.
            try:
                class BarrierHook(hooks.HookBase):
                    def __init__(self, when: str = "pre"):
                        # when: 'pre' runs barrier in before_step, 'post' runs in after_step
                        self.when = when

                    def before_step(self):
                        if self.when != "pre":
                            return
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                        try:
                            comm.synchronize()
                        except Exception:
                            pass

                    def after_step(self):
                        if self.when != "post":
                            return
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                        try:
                            comm.synchronize()
                        except Exception:
                            pass

                # Rebuild hooks_list inserting barriers around EvalHook instances
                new_hooks = []
                for h in hooks_list:
                    try:
                        if h.__class__.__name__ == "EvalHook":
                            new_hooks.append(BarrierHook("pre"))
                            new_hooks.append(h)
                            new_hooks.append(BarrierHook("post"))
                            continue
                    except Exception:
                        pass
                    new_hooks.append(h)
                hooks_list = new_hooks
            except Exception:
                pass
        except Exception:
            pass
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

    def resume_or_load(self, resume=True):
        """Resume from checkpoint and ensure trainer start iteration is set from
        the checkpoint `iteration` key (or `last_checkpoint` file) when present.
        This addresses cases where Detectron2 loads weights/optimizer but
        doesn't restore the trainer iteration counter.
        """
        # Call DefaultTrainer behavior first (loads model/optim/scheduler)
        try:
            DefaultTrainer.resume_or_load(self, resume=resume)
        except Exception:
            try:
                super().resume_or_load(resume=resume)
            except Exception:
                pass

        # Now explicitly read the checkpoint file to set start iteration
        try:
            import os
            ckpt_path = None
            last_ck = os.path.join(self.cfg.OUTPUT_DIR, "last_checkpoint")
            if os.path.exists(last_ck):
                try:
                    with open(last_ck, 'r') as f:
                        content = f.read().strip()
                    if content:
                        if os.path.isabs(content):
                            ckpt_path = content
                        else:
                            ckpt_path = os.path.join(self.cfg.OUTPUT_DIR, content)
                except Exception:
                    ckpt_path = None

            # fallback to cfg.MODEL.WEIGHTS if explicit file exists
            if ckpt_path is None:
                try:
                    w = self.cfg.MODEL.WEIGHTS
                except Exception:
                    w = None
                if isinstance(w, str) and w and os.path.exists(w):
                    ckpt_path = w

            if ckpt_path and os.path.exists(ckpt_path):
                import torch, logging
                ckpt = torch.load(ckpt_path, map_location='cpu')
                it = None
                for k in ('iteration', 'iter', 'start_iter'):
                    if k in ckpt:
                        it = ckpt[k]
                        break
                if it is not None:
                    try:
                        self.start_iter = int(it)
                        try:
                            setattr(self._trainer, 'iter', int(it))
                        except Exception:
                            pass
                        # Clamp scheduler state to valid range when resuming.
                        # Some ParamScheduler implementations compute a ratio
                        # using `last_epoch / _max_iter`. If `last_epoch` slightly
                        # exceeds `_max_iter` (e.g., due to off-by-one in saved
                        # metadata), the scheduler can raise. Ensure the saved
                        # iteration is clamped to the scheduler's max range.
                        try:
                            sched = getattr(self, 'scheduler', None)
                            if sched is not None:
                                # prefer scheduler's _max_iter if present
                                max_for_sched = getattr(sched, '_max_iter', None)
                                if max_for_sched is None:
                                    max_for_sched = getattr(self, 'max_iter', None)
                                if max_for_sched is not None:
                                    max_for_sched = int(max_for_sched)
                                    last = int(it)
                                    if last > max_for_sched:
                                        last = max_for_sched
                                    # set common scheduler bookkeeping fields
                                    try:
                                        if hasattr(sched, 'last_epoch'):
                                            sched.last_epoch = last
                                    except Exception:
                                        pass
                                    try:
                                        # torch schedulers may expose _step_count
                                        if hasattr(sched, '_step_count'):
                                            sched._step_count = last + 1
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        logging.getLogger('detectron2').info(f"Resuming: set start_iter to {self.start_iter} from {ckpt_path}")
                    except Exception:
                        pass
        except Exception:
            pass

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
    # If the user provided a COCO train JSON, infer number of classes and
    # enforce the NUM_CLASSES overrides early so model construction matches
    # the dataset (important when resuming from checkpoints saved with a
    # different class count).
    try:
        train_json_path = getattr(args, "train_json", None)
        if train_json_path and os.path.isfile(train_json_path):
            import json
            with open(train_json_path, "r") as _jf:
                _j = json.load(_jf)
            if isinstance(_j.get("categories"), list) and len(_j.get("categories", [])) > 0:
                _num_classes = len(_j["categories"])
            else:
                anns = _j.get("annotations", [])
                cat_ids = {a.get("category_id") for a in anns if "category_id" in a}
                _num_classes = len(cat_ids)
            if _num_classes and _num_classes > 0:
                try:
                    cfg.MODEL.ROI_HEADS.NUM_CLASSES = _num_classes
                except Exception:
                    cfg.MODEL.ROI_HEADS = CN() if not hasattr(cfg.MODEL, "ROI_HEADS") else cfg.MODEL.ROI_HEADS
                    cfg.MODEL.ROI_HEADS.NUM_CLASSES = _num_classes
                try:
                    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = _num_classes
                except Exception:
                    if not hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
                        cfg.MODEL.SEM_SEG_HEAD = CN()
                    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = _num_classes
                print(f"Inferred num_classes={_num_classes} from {train_json_path}; forcing MODEL.*.NUM_CLASSES")
            # If user requested from-scratch behavior, ensure MODEL.WEIGHTS is empty
            if getattr(args, "from_scratch", False):
                try:
                    cfg.MODEL.WEIGHTS = ""
                except Exception:
                    cfg.MODEL.WEIGHTS = ""
                print("from_scratch requested: clearing MODEL.WEIGHTS to initialise from scratch")
    except Exception:
        pass
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
    # Log basic distributed/runtime info to help debug multi-node divergence
    try:
        import socket
        rank = int(os.environ.get("RANK", -1))
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))
        print(f"[DDP] host={socket.gethostname()} rank={rank} local_rank={local_rank} world_size={world_size}")
        try:
            print(f"[DDP] config_file={args.config_file} OUTPUT_DIR={getattr(cfg, 'OUTPUT_DIR', None)} DATASETS.TRAIN={getattr(cfg, 'DATASETS', None)}")
        except Exception:
            pass
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
    # Extra convenience args (also accepted by scripts/launch_maskdino.py)
    parser.add_argument("--train-json", default="/workspace/output_annotations/train_polygons_clean.json")
    parser.add_argument("--val-json", default="/workspace/output_annotations/val_polygons_clean.json")
    parser.add_argument("--images-root", default="/workspace/dataset/images/train")
    parser.add_argument("--val-images-root", default="/workspace/dataset/images/val")
    parser.add_argument("--fix-json-root", action="store_true")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--base-lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--ims-per-batch", type=int, default=None)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--EVAL_FLAG', type=int, default=1)
    # Convenience: output directory (mapped to OUTPUT_DIR override)
    parser.add_argument('--output', default='/workspace/output')
    args = parser.parse_args()
    # If user provided train/val JSONs via these convenience flags, register them
    # under repository-local names and translate into cfg overrides appended
    # to args.opts so the rest of the pipeline receives them.
    try:
        dataset_overrides = []
        # Optionally rewrite JSON file paths to use provided image roots
        train_json = getattr(args, "train_json", None)
        val_json = getattr(args, "val_json", None)
        train_reg_root = getattr(args, "images_root", "") or ""
        val_reg_root = getattr(args, "val_images_root", "") or getattr(args, "images_root", "") or ""
        if getattr(args, "fix_json_root", False):
            import json, os, tempfile
            if train_json and getattr(args, "images_root", None):
                try:
                    with open(train_json, "r") as _f:
                        _j = json.load(_f)
                    for img in _j.get("images", []):
                        img["file_name"] = os.path.join(getattr(args, "images_root"), os.path.basename(img.get("file_name", "")))
                    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
                    with open(tf.name, "w") as _wf:
                        json.dump(_j, _wf)
                    train_json = tf.name
                    train_reg_root = ""
                except Exception:
                    train_reg_root = getattr(args, "images_root", "") or ""
            if val_json:
                val_root = getattr(args, "val_images_root", None) or getattr(args, "images_root", None)
                if val_root:
                    try:
                        with open(val_json, "r") as _f:
                            _jv = json.load(_f)
                        for img in _jv.get("images", []):
                            img["file_name"] = os.path.join(val_root, os.path.basename(img.get("file_name", "")))
                        tfv = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
                        with open(tfv.name, "w") as _wv:
                            json.dump(_jv, _wv)
                        val_json = tfv.name
                        val_reg_root = ""
                    except Exception:
                        val_reg_root = getattr(args, "val_images_root", None) or getattr(args, "images_root", None) or ""

        # register datasets if paths exist
        try:
            from detectron2.data.datasets import register_coco_instances
        except Exception:
            register_coco_instances = None

        train_name = "drone_train_polygons"
        val_name = "drone_val_polygons"
        if train_json and os.path.isfile(train_json) and register_coco_instances is not None:
            register_coco_instances(train_name, {}, train_json, train_reg_root)
            dataset_overrides.extend(["DATASETS.TRAIN", "('" + train_name + "',)"])
        if val_json and os.path.isfile(val_json) and register_coco_instances is not None:
            register_coco_instances(val_name, {}, val_json, val_reg_root)
            dataset_overrides.extend(["DATASETS.TEST", "('" + val_name + "',)"])

        # output override
        if getattr(args, "output", None):
            dataset_overrides.extend(["OUTPUT_DIR", getattr(args, "output")])

        # from-scratch: infer num classes and clear MODEL.WEIGHTS
        if getattr(args, "from_scratch", False) and train_json and os.path.isfile(train_json):
            try:
                import json
                with open(train_json, "r") as _f:
                    _j = json.load(_f)
                if isinstance(_j.get("categories"), list) and len(_j.get("categories", [])) > 0:
                    num_classes = len(_j["categories"])
                else:
                    anns = _j.get("annotations", [])
                    cat_ids = {a.get("category_id") for a in anns if "category_id" in a}
                    num_classes = len(cat_ids)
                if num_classes and num_classes > 0:
                    dataset_overrides.extend(["MODEL.WEIGHTS", ""])
                    dataset_overrides.extend(["MODEL.ROI_HEADS.NUM_CLASSES", str(num_classes)])
                    dataset_overrides.extend(["MODEL.SEM_SEG_HEAD.NUM_CLASSES", str(num_classes)])
            except Exception:
                dataset_overrides.extend(["MODEL.WEIGHTS", ""])

        # hyperparameter overrides
        if getattr(args, "max_iter", None) is not None:
            dataset_overrides.extend(["SOLVER.MAX_ITER", str(getattr(args, "max_iter"))])
        if getattr(args, "base_lr", None) is not None:
            dataset_overrides.extend(["SOLVER.BASE_LR", str(getattr(args, "base_lr"))])
        if getattr(args, "num_workers", None) is not None:
            dataset_overrides.extend(["DATALOADER.NUM_WORKERS", str(getattr(args, "num_workers"))])
        if getattr(args, "ims_per_batch", None) is not None:
            dataset_overrides.extend(["SOLVER.IMS_PER_BATCH", str(getattr(args, "ims_per_batch"))])

        # If dataset_overrides exist, append them to args.opts so setup() will merge them
        if dataset_overrides:
            if not hasattr(args, "opts") or args.opts is None:
                args.opts = []
            args.opts = list(args.opts) + dataset_overrides

        # Defensive runtime monkeypatch for numpy segmentations -> lists, similar to launcher
        try:
            import numpy as _np
            from detectron2.data import detection_utils as _dutils

            _orig_annotations_to_instances = _dutils.annotations_to_instances

            def _patched_annotations_to_instances(annotations, image_size):
                new_annos = []
                for ann in annotations:
                    if not isinstance(ann, dict):
                        new_annos.append(ann)
                        continue
                    seg = ann.get("segmentation", None)
                    if seg is None:
                        new_annos.append(ann)
                        continue
                    ann_copy = ann.copy()
                    if isinstance(seg, _np.ndarray):
                        if seg.ndim == 1:
                            ann_copy["segmentation"] = seg.tolist()
                        else:
                            try:
                                ann_copy["segmentation"] = [p.tolist() for p in seg]
                            except Exception:
                                ann_copy["segmentation"] = seg.flatten().tolist()
                    elif isinstance(seg, list):
                        changed = False
                        new_seg = []
                        for poly in seg:
                            if isinstance(poly, _np.ndarray):
                                changed = True
                                new_seg.append(poly.tolist())
                            else:
                                new_seg.append(poly)
                        if changed:
                            ann_copy["segmentation"] = new_seg

                    new_annos.append(ann_copy)

                return _orig_annotations_to_instances(new_annos, image_size)

            _dutils.annotations_to_instances = _patched_annotations_to_instances
        except Exception:
            pass
    except Exception:
        # if anything goes wrong, continue with original args
        pass
    # If this script is launched via `torchrun` / `torch.distributed.run`, the
    # distributed environment variables (WORLD_SIZE / RANK / LOCAL_RANK) will be
    # present and each process should call `main()` directly. Otherwise, use
    # Detectron2's `launch()` helper to spawn local processes (and multi-node
    # support when requested).
    is_torchrun = any(k in os.environ for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    if is_torchrun:
        # When launched by torchrun, initialize the process group and set the
        # CUDA device for this local rank so Detectron2's comm utilities see
        # the correct world size and rank.
        import torch.distributed as dist
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(local_rank)
            except Exception:
                pass
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            try:
                dist.init_process_group(backend=backend, init_method="env://")
            except Exception:
                pass
        # Create Detectron2's local process group so comm.get_local_rank()
        # and related utilities work (required by create_ddp_model).
        try:
            # Prefer torchrun's per-node count; fall back to WORLD_SIZE when missing.
            try:
                local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", None) or os.environ.get("LOCAL_SIZE", None) or 0)
            except Exception:
                local_world_size = 0
            try:
                global_world_size = int(os.environ.get("WORLD_SIZE", "1"))
            except Exception:
                global_world_size = 1

            # If LOCAL_WORLD_SIZE is not provided, attempt to infer per-node
            # worker count from global world size and visible CUDA devices.
            if local_world_size <= 0:
                try:
                    local_world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
                except Exception:
                    local_world_size = 1

            should_create = (local_world_size > 1) or (global_world_size > 1)
            if should_create:
                try:
                    print(f"[DDP] creating local process group: local_world_size={local_world_size} global_world_size={global_world_size} LOCAL_WORLD_SIZE={os.environ.get('LOCAL_WORLD_SIZE')} LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
                    comm.create_local_process_group(local_world_size)
                    print("[DDP] detectron2 local process group created")
                except Exception as e:
                    # Log and re-raise to make failure visible rather than silently
                    print("[DDP] failed to create detectron2 local process group:", repr(e))
                    raise
        except Exception:
            # If anything goes wrong here, surface the error later during model creation
            pass
        print("Detected torchrun / torch.distributed.run environment.")
        print("Command Line Args:", args)
        print("pwd:", os.getcwd())
        main(args)
    else:
        port = random.randint(1000, 20000)
        args.dist_url = 'tcp://127.0.0.1:' + str(port)
        print("Command Line Args:", args)
        print("pwd:", os.getcwd())
        launch(
            main,
            getattr(args, "num_gpus", 1),
            num_machines=getattr(args, "num_machines", 1),
            machine_rank=getattr(args, "machine_rank", 0),
            dist_url=args.dist_url,
            args=(args,),
        )
