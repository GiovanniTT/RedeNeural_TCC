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
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

# Configurações
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuração do banco de dados
CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'Paralelepipedo123',
    'port': 3306,
    'database': 'dengue_db'
}

CONN_STRING = (f"mysql+mysqlconnector://{CONFIG['user']}:{CONFIG['password']}"
               f"@{CONFIG['host']}:{CONFIG['port']}/{CONFIG['database']}")

def carregar_dados():
    """Carrega os dados do banco de dados"""
    try:
        engine = create_engine(CONN_STRING)
        with engine.connect() as conn:
            df = pd.read_sql("SELECT * FROM dengue_db.dengue_dados", conn)
        engine.dispose()
        if df.empty:
            logging.warning("Nenhum dado encontrado no banco.")
            return None
        logging.info("Dados carregados com sucesso.")
        return df
    except Exception as e:
        logging.error(f"Erro ao carregar dados: {e}")
        return None

def salvar_metricas_em_arquivo(history, caminho_arquivo):
    """Salva as métricas de treinamento em arquivo CSV"""
    ultimos = 25
    metricas = {
        'epoch': list(range(len(history.history['loss']) - ultimos, len(history.history['loss']))),
        'loss': history.history['loss'][-ultimos:],
        'val_loss': history.history['val_loss'][-ultimos:],
        'mae': history.history['mae'][-ultimos:],
        'val_mae': history.history['val_mae'][-ultimos:]
    }
    df_metricas = pd.DataFrame(metricas)
    df_metricas['timestamp'] = pd.Timestamp.now()

    linha_em_branco = pd.DataFrame([[''] * len(df_metricas.columns)], columns=df_metricas.columns)

    if os.path.exists(caminho_arquivo) and os.path.getsize(caminho_arquivo) > 0:
        try:
            df_antigo = pd.read_csv(caminho_arquivo)
            df_final = pd.concat([df_antigo, linha_em_branco, df_metricas], ignore_index=True)
        except pd.errors.EmptyDataError:
            logging.warning("Arquivo CSV está vazio. Será sobrescrito.")
            df_final = df_metricas
    else:
        df_final = df_metricas

    df_final.to_csv(caminho_arquivo, index=False)
    logging.info(f"Últimas métricas salvas em {caminho_arquivo}")

def obter_estados_no_banco(df):
    """Retorna a lista de estados únicos no DataFrame"""
    return df['estado'].unique()

def obter_meses_no_banco(df):
    """Retorna a lista de meses únicos no DataFrame"""
    return sorted(df['mes'].unique())

def salvar_previsoes_em_csv(lista_previsoes, caminho_arquivo):
    """Salva as previsões em arquivo CSV"""
    df_previsoes = pd.DataFrame(lista_previsoes)
    df_previsoes['timestamp'] = pd.Timestamp.now()

    if os.path.exists(caminho_arquivo):
        df_antigo = pd.read_csv(caminho_arquivo)
        df_final = pd.concat([df_antigo, df_previsoes], ignore_index=True)
    else:
        df_final = df_previsoes

    df_final.to_csv(caminho_arquivo, index=False)
    logging.info(f"Previsões salvas em {caminho_arquivo}")

def salvar_tempo_execucao_csv(tempo_execucao, caminho_arquivo):
    """Salva o tempo de execução em arquivo CSV"""
    timestamp_str = pd.Timestamp.now()
    dados = {
        'data_hora': [timestamp_str],
        'tempo_execucao_segundos': [tempo_execucao]
    }
    df = pd.DataFrame(dados)

    if os.path.exists(caminho_arquivo):
        try:
            df_antigo = pd.read_csv(caminho_arquivo)
            df_final = pd.concat([df_antigo, df], ignore_index=True)
        except pd.errors.EmptyDataError:
            logging.warning(f"Arquivo {caminho_arquivo} está vazio. Será sobrescrito.")
            df_final = df
    else:
        df_final = df

    df_final.to_csv(caminho_arquivo, index=False)
    logging.info(f"Tempo de execução salvo em {caminho_arquivo}")
    
def salvar_metricas_execucao_csv(mse, mae, arquivo_metricas):
    """Salva as métricas de execução em arquivo CSV"""
    dados = {
        'mse_teste': [mse],
        'mae_teste': [mae],
    }
    df = pd.DataFrame(dados)

    if os.path.exists(arquivo_metricas):
        try:
            df_antigo = pd.read_csv(arquivo_metricas)
            df_final = pd.concat([df_antigo, df], ignore_index=True)
        except pd.errors.EmptyDataError:
            logging.warning(f"Arquivo {arquivo_metricas} está vazio. Será sobrescrito.")
            df_final = df
    else:
        df_final = df

    df_final.to_csv(arquivo_metricas, index=False)
    logging.info(f"Métricas de desempenho salvas em {arquivo_metricas}")

def plotar_grafico_perda(history):
    """Plota o gráfico de perda durante o treinamento"""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Treinamento')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Histórico de Perda', fontsize=14)
    plt.xlabel('Época')
    plt.ylabel('MSE')
    plt.legend()    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def processar_dados_para_lstm(df, look_back=3):
    """Prepara os dados no formato adequado para LSTM"""
    try:
        if 'mes' not in df.columns:
            logging.error("Coluna 'mes' não encontrada nos dados.")
            return None, None, None, None, None

        # Ordenar por estado e tempo
        df = df.sort_values(['estado', 'ano', 'mes'])
        
        # Criar features
        df = df[['ano', 'mes', 'estado', 'mortes', 'casos', 'temperatura', 'precipitacao']]
        X = pd.get_dummies(df[['ano', 'mes', 'estado', 'mortes', 'temperatura', 'precipitacao']], 
                         columns=['estado', 'mes'])
        y = df[['casos']]
        
        # Normalização
        scaler_X = StandardScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        # Preparar sequências para LSTM
        X_seq, y_seq = [], []
        estados = df['estado'].unique()
        
        for estado in estados:
            estado_mask = df['estado'] == estado
            X_estado = X_scaled[estado_mask]
            y_estado = y_scaled[estado_mask]
            
            for i in range(look_back, len(X_estado)):
                X_seq.append(X_estado[i-look_back:i])
                y_seq.append(y_estado[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        return X_seq, y_seq, scaler_X, scaler_y, X.columns
        
    except KeyError as e:
        logging.error(f"Erro no pré-processamento: Coluna ausente {e}")
        return None, None, None, None, None

def criar_modelo_lstm(input_shape):
    """Cria modelo LSTM com arquitetura adequada para previsão de séries temporais"""
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4))),
        Dropout(0.5),
        Bidirectional(LSTM(64, kernel_regularizer=l2(1e-4))),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0005), 
                 loss='mse', 
                 metrics=['mae'])
    
    return model

def criar_e_treinar_modelo(X, y, input_shape):
    """Função para treinar o modelo LSTM"""
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)  # shuffle=False para dados temporais

    model = criar_modelo_lstm(input_shape)

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True, 
        verbose=1, 
        min_delta=0.001,
        mode='min')
        
    checkpoint = ModelCheckpoint(
        'Dados/melhor_modelo_lstm.keras', 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1)

    history = model.fit(
        x_train, y_train,
        epochs=150,
        batch_size=64,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    salvar_metricas_em_arquivo(history, "Dados/historico_metricas_lstm.csv")
    
    return model, history

def criar_entrada_lstm(ano, mes, estado, mortes, encoder_columns, scaler_X, historico, look_back=3):
    """Prepara entrada no formato adequado para LSTM"""
    # Criar entrada atual
    entrada_atual = pd.DataFrame(0, index=[0], columns=encoder_columns)
    
    if 'ano' in entrada_atual.columns:
        entrada_atual['ano'] = ano
    if 'mortes' in entrada_atual.columns:
        entrada_atual['mortes'] = mortes
    
    col_estado = f"estado_{estado}"
    col_mes = f"mes_{mes}"
    
    if col_estado in entrada_atual.columns:
        entrada_atual[col_estado] = 1
    if col_mes in entrada_atual.columns:
        entrada_atual[col_mes] = 1
    
    # Normalizar
    entrada_atual = scaler_X.transform(entrada_atual)
    
    # Combinar com histórico
    if len(historico) >= look_back:
        seq = np.vstack([historico[-(look_back-1):], entrada_atual])
    else:
        # Preencher com zeros se não houver histórico suficiente
        padding = np.zeros((look_back - len(historico) - 1, entrada_atual.shape[1]))
        seq = np.vstack([padding, historico, entrada_atual])
    
    return seq.reshape(1, look_back, -1)

def fazer_previsoes_lstm(model, scaler_X, scaler_y, encoder_columns, df, ano_previsao=2024, look_back=3):
    """Adaptada para fazer previsões com LSTM"""
    estados_no_banco = obter_estados_no_banco(df)
    meses_no_banco = obter_meses_no_banco(df)
    previsoes = []
    
    print(f"\n=== PREVISÕES LSTM PARA {ano_previsao} ===\n")
    
    # Para cada estado, precisamos do histórico
    for estado in sorted(estados_no_banco):
        # Obter dados históricos do estado
        df_estado = df[df['estado'] == estado].sort_values(['ano', 'mes'])
        
        # Preparar dados históricos normalizados
        X_estado = pd.get_dummies(df_estado[['ano', 'mes', 'estado', 'mortes', 'temperatura', 'precipitacao']], 
                                columns=['estado', 'mes'])
        X_estado = X_estado.reindex(columns=encoder_columns, fill_value=0)
        historico = scaler_X.transform(X_estado)
        
        for mes in meses_no_banco:
            entrada = criar_entrada_lstm(
                ano_previsao, mes, estado, 0, encoder_columns, scaler_X, historico, look_back)
            
            previsao_normalizada = model.predict(entrada)
            previsao = scaler_y.inverse_transform(previsao_normalizada)
            
            casos = int(previsao[0][0])
            print(f"{estado} - {mes}/{ano_previsao}: Casos: {casos:,}")
            
            previsoes.append({
                "ano": ano_previsao,
                "mes": mes,
                "estado": estado,
                "casos_Previstos": casos,
            })
    
    salvar_previsoes_em_csv(previsoes, "Dados/previsoes_lstm.csv")

def executar_codigo_lstm():
    """Função principal para execução do código LSTM"""
    start_time = time.time()
    
    df = carregar_dados()
    if df is None:
        return

    # Processamento específico para LSTM
    X, y, scaler_X, scaler_y, encoder_columns = processar_dados_para_lstm(df, look_back=3)
    if X is None or y is None:
        return

    # Treinar modelo LSTM
    input_shape = (X.shape[1], X.shape[2])  # (timesteps, features)
    model, history = criar_e_treinar_modelo(X, y, input_shape)
    
    # Avaliação
    score = model.evaluate(X, y, verbose=0)
    final_mse = score[0]
    final_mae = score[1]

    elapsed_time = time.time() - start_time
    print(f"\nTempo de execução: {elapsed_time:.2f} segundos.")

    salvar_metricas_execucao_csv(final_mse, final_mae, arquivo_metricas="Dados/metricas_lstm.csv")
    salvar_tempo_execucao_csv(elapsed_time, caminho_arquivo="Dados/tempo_execucao_lstm.csv")

    fazer_previsoes_lstm(model, scaler_X, scaler_y, encoder_columns, df, ano_previsao=2024)
    plotar_grafico_perda(history)

if __name__ == "__main__":
    start_time = time.time()
    print(f"\n=== Execução LSTM ===")
    executar_codigo_lstm()
    elapsed_time = time.time() - start_time
    print(f"\nTempo total de execução: {elapsed_time:.2f} segundos.")