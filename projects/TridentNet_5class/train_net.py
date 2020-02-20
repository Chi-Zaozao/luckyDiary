import os
from detectron2.data import MetadataCatalog
from detectron2.config import CfgNode as CN
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from projects.TridentNet.tridentnet import add_tridentnet_config
from detectron2.engine import default_setup, DefaultTrainer, default_argument_parser, launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator, verify_results
import detectron2.utils.comm as comm

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
    cfg.MODEL.PIXEL_MEAN = [28, 28, 28]
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.05, 0.3]
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.3]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8,16,32,64,128]]
    cfg.SOLVER.MAX_ITER = 1200000
    cfg.DATASETS.TRAIN = ('sim_multi_train',)
    cfg.DATASETS.TEST = ('sim_multi_test',)
    cfg.INPUT.CROP = CN({"ENABLED": True})
    cfg.INPUT.CROP.TYPE = "relative_range"
    cfg.INPUT.CROP.SIZE = [0.75, 0.95]
    cfg.TEST.EVAL_PERIOD = 50000
    cfg.MODEL.WEIGHTS = './output/model_1054999.pth'
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # register the datasets
#     register_coco_instances('sim_multi_train', {},\
#                         '../../small_multi/train_annotations.json', \
#                         '../../../sim_data_center/multi_class_detection/down/JPEGImages/')
#     register_coco_instances('sim_multi_test', {},\
#                         '../../small_multi/test_annotations.json', \
#                         '../../../sim_data_center/multi_class_detection/down/JPEGImages/')
    register_coco_instances('sim_multi_train', {},\
                        './dataset/train_annotations.json', \
                        '../../../sim_data_center/multi_class_detection/down/JPEGImages/')
    register_coco_instances('sim_multi_test', {},\
                        './dataset/test_annotations.json', \
                        '../../../sim_data_center/multi_class_detection/down/JPEGImages/')
#     register_coco_instances('sim_multi_train', {},\
#                         './projects/TridentNet_5class/dataset/train_annotations.json', \
#                         '../sim_data_center/multi_class_detection/down/JPEGImages/')
#     register_coco_instances('sim_multi_test', {},\
#                         './projects/TridentNet_5class/dataset/test_annotations.json', \
#                         '../sim_data_center/multi_class_detection/down/JPEGImages/')
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        cfg.defrost()
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#         cfg.MODEL.RPN.IOU_THRESHOLDS = [0.05, 0.3]
#         cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.2]
        cfg.freeze()
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
    # specify the gpus to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
