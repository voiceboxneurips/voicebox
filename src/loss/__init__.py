# Adversarial losses
from src.loss.cross_entropy import CELoss
from src.loss.cw import CWLoss
from src.loss.speaker_embedding import SpeakerEmbeddingLoss

# Auxiliary losses
from src.loss.l1 import L1Loss
from src.loss.l2 import L2Loss
from src.loss.mrstft import MRSTFTLoss
from src.loss.demucs_mrstft import DemucsMRSTFTLoss
from src.loss.mfcc_cosine import MFCCCosineLoss
from src.loss.speech_features import SpeechFeatureLoss
from src.loss.frequency_masking import FrequencyMaskingLoss
from src.loss.sum import SumLoss
from src.loss.control import ControlSignalLoss
