import os
import torch

CACHE_DIR = os.getenv('QA_CACHE_PATH', 'cache')
DATA_DIR = os.getenv('QA_DATA_PATH', 'data')

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
SQUAD_TRAIN = f"{DATA_DIR}/train-v2.0.json"
LEMMA_CACHE = f"{CACHE_DIR}/lemmas.feather"
VECTOR_CACHE = f"{CACHE_DIR}/vectors-v1.pickle"
DOCBIN_CACHE = f"{CACHE_DIR}/docs.bin"
VOCAB_CACHE = f"{CACHE_DIR}/vocab.bin"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOPK = 10 if DEVICE.type == 'cuda' else 5
