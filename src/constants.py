from pathlib import Path

################################################################################
# Project-wide constants
################################################################################

# Static directories
CACHE_DIR = Path(__file__).parent.parent / 'cache'
DATA_DIR = Path(__file__).parent.parent / 'data'
RUNS_DIR = Path(__file__).parent.parent / 'runs'
TEST_DIR = Path(__file__).parent.parent / 'test'
CONFIGS_DIR = Path(__file__).parent.parent / 'configs'
MODELS_DIR = Path(__file__).parent.parent / 'pretrained'

# Set constant properties for streaming operations
WIN_LENGTH = 256
HOP_LENGTH = 128
SAMPLE_RATE = 16000

# VoxCeleb1 dataset
VOXCELEB1_DATA_DIR = DATA_DIR / 'VoxCeleb1'
VOXCELEB1_EXT = 'wav'

# VoxCeleb2 dataset
VOXCELEB2_DATA_DIR = DATA_DIR / 'VoxCeleb2'

# Pretrained phoneme prediction model
PPG_PRETRAINED_PATH = MODELS_DIR / 'phoneme' / 'causal_ppg_256_hidden.pt'

# Pretrained VoiceBox attack
VOICEBOX_PRETRAINED_PATH = MODELS_DIR / 'voicebox' / 'voicebox_final.pt'

# Pretrained universal additive attack
UNIVERSAL_PRETRAINED_PATH = MODELS_DIR / 'universal' / 'universal_final.pt'

# LibriSpeech dataset
LIBRISPEECH_DATA_DIR = DATA_DIR / 'LibriSpeech'
LIBRISPEECH_CACHE_DIR = CACHE_DIR / 'LibriSpeech'
LIBRISPEECH_SIG_LEN = 4.0
LIBRISPEECH_EXT = 'flac'
LIBRISPEECH_PHONEME_EXT = 'TextGrid'
LIBRISPEECH_NUM_PHONEMES = 70  # first phoneme corresponds to silence
LIBRISPEECH_PHONEME_DICT = {
    'sil': 0, '': 0, 'sp': 0, 'spn': 0,
    'AE1': 1, 'P': 2, 'T': 3, 'ER0': 4,
    'W': 5, 'AH1': 6, 'N': 7, 'M': 8,
    'IH1': 9, 'S': 10, 'IH0': 11, 'Z': 12,
    'R': 13, 'EY1': 14, 'AH0': 15, 'L': 16,
    'D': 17, 'AY1': 18, 'V': 19, 'JH': 20,
    'EH1': 21, 'DH': 22, 'IY0': 23, 'IY2': 24,
    'OW1': 25, 'AW1': 26, 'UW1': 27, 'HH': 28,
    'AA1': 29, 'OW0': 30, 'F': 31, 'TH': 32,
    'AO1': 33, 'AA2': 34, 'ER1': 35, 'B': 36,
    'UH1': 37, 'K': 38, 'Y': 39, 'IY1': 40,
    'AO2': 41, 'NG': 42, 'AE0': 43, 'G': 44,
    'SH': 45, 'IH2': 46, 'EH2': 47, 'UW0': 48,
    'AY2': 49, 'EY2': 50, 'AA0': 51, 'OY1': 52,
    'AE2': 53, 'ZH': 54, 'EH0': 55, 'OW2': 56,
    'AH2': 57, 'UH2': 58, 'AO0': 59, 'UW2': 60,
    'EY0': 61, 'AW2': 62, 'AY0': 63, 'ER2': 64,
    'OY0': 65, 'OY2': 66, 'UH0': 67, 'AW0': 68,
    'CH': 69}
LIBRISPEECH_FILLER_PHONEMES = ['', 'sil', 'sp', 'spn']

# Streamer Conditioning
CONDITIONING_FOLDER = DATA_DIR / 'streamer'
CONDITIONING_FILENAME = CONDITIONING_FOLDER / 'conditioning.pt'
