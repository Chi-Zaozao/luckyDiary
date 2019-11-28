# python3 train_sim.py --config-file='../../projects/TridentNet/configs/tridentnet_fast_R_50_C4_1x.yaml' --num-gpus 4 SOLVER.IMS_PER_BATCH 4 
import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from get_tinysim import get_tinysim
from detectron2.data import DatasetCatalog, MetadataCatalog

from projects.TridentNet.tridentnet import add_tridentnet_config


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_tridentnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TRAIN = ('tiny_simdata/train',)
    cfg.DATASETS.TEST = ('tiny_simdata/val',)
    cfg.SOLVER.MAX_ITER = 2000
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    for d in ['train','val']:
        DatasetCatalog.register('tiny_simdata/'+d, lambda d = d: get_tinysim('../tiny_simdata/'+d))
        MetadataCatalog.get('tiny_simdata/'+d).set(thing_classes=['gun','lighter'])
        MetadataCatalog.get('tiny_simdata/'+d).set(json_file='../tiny_simdata/'+d+'/annotations.json')
    object_metadata=MetadataCatalog.get('tiny_simdata/train') 
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
#     specify the visible gpus
#     os.environ["CUDA_VISIBLE_DEVICES"] = '0,3'
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
