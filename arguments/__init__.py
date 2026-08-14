#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, ArgumentTypeError, Namespace
import sys
import os

TRUTHY = ("true", "t", "yes", "y", "1")
FALSY = ("false", "f", "no", "n", "0")


def str2bool(text):
    """Parse a command-line boolean word.

    Needed for flags that DEFAULT TO TRUE: argparse's "store_true" can only ever
    turn a flag on, so a True default is stuck on forever and there is no way to
    ask for the other behaviour. Registering those with this as the type (and
    nargs="?", const=True) keeps the bare `--flag` form working while also
    accepting `--flag False`.
    """
    lowered = str(text).strip().lower()
    if lowered in TRUTHY:
        return True
    if lowered in FALSY:
        return False
    raise ArgumentTypeError("expected a boolean word, got {!r}".format(text))


class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            # A True default has to be switchable off; a False default keeps the
            # historical bare-flag form so no existing command line changes.
            default_true = (t == bool and value is True)
            value = value if not fill_none else None
            if shorthand:
                if default_true:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value,
                                       type=str2bool, nargs="?", const=True)
                elif t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if default_true:
                    group.add_argument("--" + key, default=value,
                                       type=str2bool, nargs="?", const=True)
                elif t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.depths = ""            # empty string disables depth supervision
        self.depth_scale_file = ""  # defaults to <model_dir>/depth_scale.json when empty
        self.depth_max = 30.0       # metres
        self.depth_on_cpu = False
        self.init_from_depth = False   # replace COLMAP sparse points with dense backprojection
        self.depth_init_voxel = 0.02   # voxel size (COLMAP units) for downsampling the backprojection
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.separate_sh = True
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025 
        self.shfeature_lr = 0.005 
        self.opacity_lr = 0.025 
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.001
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        
        # fastgs parameters
        self.loss_thresh = 0.1
        self.grad_abs_thresh = 0.0012  
        self.highfeature_lr = 0.005
        self.lowfeature_lr = 0.0025
        self.grad_thresh = 0.0002
        self.dense = 0.001
        self.mult = 0.5      # multiplier for the compact box to control the tile number of each splat

        self.random_background = False
        self.optimizer_type = "default"

        # depth supervision
        self.lambda_depth = 0.5
        self.depth_from_iter = 0
        self.depth_loss = "edgeaware_logl1"
        # Supervise D / A rather than raw D. On by default: raw D lets the
        # optimiser cut depth error by fading splats instead of moving them,
        # which hollows out surfaces. `--depth_normalize False` restores the
        # old un-normalised behaviour for comparison.
        self.depth_normalize = True
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
