import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.exceptions import NotFittedError

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from keras.optimizers import Adam

# Forçar uso de CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# Configurações de threads para TensorFlow
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Ignorar warnings
warnings.filterwarnings("ignore")

# Configurar logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuração de conexão com o banco de dados
CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'Paralelepipedo123',
    'port': 3306,
    'database': 'dengue_db'
}

CONN_STRING = (f"mysql+mysqlconnector://{CONFIG['user']}:{CONFIG['password']}"
               f"@{CONFIG['host']}:{CONFIG['port']}/{CONFIG['database']}")
