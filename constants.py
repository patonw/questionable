import os

CACHE_DIR = os.getenv('QA_CACHE_PATH', 'cache')
DATA_DIR = os.getenv('QA_DATA_PATH', 'data')

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
SQUAD_TRAIN = f"{DATA_DIR}/train-v2.0.json"
LEMMA_CACHE = f"{CACHE_DIR}/lemmas.feather"
VECTOR_CACHE = os.getenv('QA_VECTOR_CACHE', f"{CACHE_DIR}/vectors.pickle")
