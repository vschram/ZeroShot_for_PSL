from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()


# -----------------------------------------------------------------------------
# Synthetic Data
# -----------------------------------------------------------------------------
_C.SYN = CN()
_C.SYN.NUM_REPLICATES = 10 ## The number of replicates
_C.SYN.NUM_DATA_IN_REPLICATES = 10 ## The number of data points in each replicate
_C.SYN.NUM_OUTPUTS = 3 ## The number of outputs
_C.SYN.NOISE = [0.02]
_C.SYN.MAX_RANGE = 10
_C.SYN.LINE_RANGE = 1
_C.SYN.TRAIN_PERCENTAGE = 0.5


# -----------------------------------------------------------------------------
# Model Parameter
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.Q = 2 ## The dimension of H
_C.MODEL.GAP = 2


# -----------------------------------------------------------------------------
# Optimization Step
# -----------------------------------------------------------------------------
_C.OPTIMIZATION = CN()
_C.OPTIMIZATION.TRAINING_NUM_EACH_STEP = 100


# ---------------------------------------------------------------------------- #
# Paths
# ---------------------------------------------------------------------------- #
_C.PATH = CN()
_C.PATH.SAVING_GENERAL = '/home/chunchao/Desktop/Second_project/Second_project_result_test' # Output directory
_C.PATH.LOSS = '/Loss/synthetic-same-input/Loss-HMOGPLV' # Output loss directory
_C.PATH.PARAMETERS = '/Model_parameter/synthetic-same-input/Parameter-HMOGPLV' # Output parameters directory
_C.PATH.PLOT = '/All-plot/synthetic-same-input/Plot-HMOGPLV' # Output plot directory
_C.PATH.RESULT = '/Result/synthetic-same-input/Result-HMOGPLV/' # Output result directory
_C.PATH.DATA_PATH = '/Users/chunchaoma/Desktop/Dataset_second/'


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.MISC = CN()
_C.MISC.NUM_REPETITION = 2
_C.MISC.NUM_KERNEL = 2
_C.MISC.MR = 2
_C.MISC.MODEL_NAME = 'HMOGPLV'
_C.MISC.DATA_SPEC = 'None'
_C.MISC.DATA_NAME = 'Gene'
_C.MISC.EXPERIMENTTYPE = 'Train_test_in_each_replica'
_C.MISC.VARIANCE_BOUND = 1e-6
_C.MISC.NUM_SAMPLED = 5
_C.MISC.MARKER_LIST = [([0], 0), ([0], 1), ([0], 2), ([0], 3), ([0], 4), ([0], 5), ([1], 0), ([1], 1), ([1], 2),
 ([2], 0), ([2], 1), ([2], 2), ([3], 0), ([3], 1), ([3], 2), ([4], 1), ([5], 0), ([5], 1), ([5], 2), ([6], 0),
 ([6], 1), ([6], 2), ([8], 1), ([8], 2), ([9], 0), ([10], 0), ([15], 0), ([15], 1), ([15], 2), ([16], 0), ([17], 0),
 ([21], 0), ([21], 1), ([21], 2), ([22], 0), ([23], 0), ([23], 1), ([25], 0), ([25], 1), ([25], 2), ([26], 0), ([27], 1)]

def get_cfg_defaults_all():
    return _C.clone()
