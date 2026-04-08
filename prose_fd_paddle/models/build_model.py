import sys

sys.path.append("/home/lkyu/baidu/prose-fd/prose_fd_paddle")
from logging import getLogger

import paddle
from paddle_utils import *
from tabulate import tabulate

from .baselines import FNO, DeepONet, UNet, ViT
from .transformer_wrappers import PROSE_1to1, PROSE_2to1

logger = getLogger()


def build_model(params, model_config, data_config, symbol_env):
    modules = {}
    name = model_config.name
    if name == "prose_1to1":
        modules["model"] = PROSE_1to1(
            model_config,
            symbol_env,
            data_config.x_num,
            data_config.max_output_dimension,
            data_config.t_num - params.input_len,
        )
    elif name == "prose_2to1":
        modules["model"] = PROSE_2to1(
            model_config,
            symbol_env,
            data_config.x_num,
            data_config.max_output_dimension,
            data_config.t_num - params.input_len,
        )
    elif name == "fno":
        modules["model"] = FNO(model_config, data_config.max_output_dimension)
    elif name == "vit":
        modules["model"] = ViT(
            model_config, data_config.x_num, data_config.max_output_dimension
        )
    elif name == "unet":
        modules["model"] = UNet(model_config, data_config.max_output_dimension)
    elif name == "deeponet":
        modules["model"] = DeepONet(model_config, data_config, params.input_len)
    else:
        assert False, f"Model {name} hasn't been implemented"
    if params.reload_model:
        logger.info(f"Reloading modules from {params.reload_model} ...")
        reloaded = paddle.load(path=str(params.reload_model))
        for k, v in modules.items():
            assert k in reloaded, f"{k} not in save"
            if all([k2.startswith("module.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {
                    k2[len("module.") :]: v2 for k2, v2 in reloaded[k].items()
                }
            if all([k2.startswith("_orig_mod.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {
                    k2[len("_orig_mod.") :]: v2 for k2, v2 in reloaded[k].items()
                }
            v.load_state_dict(reloaded[k])
    for k, v in modules.items():
        logger.info(f"{k}: {v}")
    for k, v in modules.items():
        s = f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if not p.stop_gradient]):,}"
        if hasattr(v, "summary"):
            s += v.summary()
        logger.info(s)
    if not params.cpu:
        for v in modules.values():
            v.cuda()
    return modules
