from src.learners.baselines.scr import SCRLearner
from src.learners.supcon import SupConLearner
from src.learners.ce import CELearner
from src.learners.baselines.er import ERLearner
from src.learners.sscr import SSCLLearner

from src.buffers.reservoir import Reservoir



learners = {
    'ER':   ERLearner,
    'CE':   CELearner,
    'SC':   SupConLearner,
    'SCR':  SCRLearner,
    'SSCL': SSCLLearner,
}

buffers = {
    'reservoir': Reservoir,
}
