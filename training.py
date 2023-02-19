import torch.multiprocessing
import omegaconf
from openhands.apis.classification_model import ClassificationModel
from openhands.core.exp_utils import get_trainer

#if __name__ == '__main__':
 #   torch.multiprocessing.freeze_support()

    # cfg = omegaconf.OmegaConf.load("path/to/config.yaml")
cfg = omegaconf.OmegaConf.load("pose_lstm.yaml") #works
#cfg = omegaconf.OmegaConf.load("pose_st_gcn.yaml") #works
#cfg = omegaconf.OmegaConf.load("pose_bert.yaml") #works

#maybe try with attention
trainer = get_trainer(cfg)

model = ClassificationModel(cfg=cfg, trainer=trainer)
model.init_from_checkpoint_if_available()
model.fit()


