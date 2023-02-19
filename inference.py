import omegaconf
from openhands.apis.inference import InferenceModel
import openhands.datasets.isolated
import sys



#cfg = omegaconf.OmegaConf.load("pose_inference_lstm.yaml")
#cfg = omegaconf.OmegaConf.load("pose_inference_bert.yaml")
cfg = omegaconf.OmegaConf.load("pose_inference_st_gcn.yaml")
model = InferenceModel(cfg=cfg)

model.init_from_checkpoint_if_available()
if cfg.data.test_pipeline.dataset.inference_mode:
    model.test_inference()
else:
    model.compute_test_accuracy()


