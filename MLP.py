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

def carregar_dados():
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

def processar_dados(df):
    try:
        if 'mes' not in df.columns:
            logging.error("Coluna 'Mes' não encontrada nos dados.")
            return None, None, None

        df = df[['ano', 'mes', 'estado', 'mortes', 'casos', 'temperatura', 'precipitacao']]
        X = pd.get_dummies(df[['ano', 'mes', 'estado', 'mortes', 'temperatura', 'precipitacao']], columns=['estado', 'mes'])
        y = df[['casos']]
        return X, y, X.columns
    except KeyError as e:
        logging.error(f"Erro no pré-processamento: Coluna ausente {e}")
        return None, None, None

def criar_e_treinar_modelo(X_scaled, y_scaled, input_dim):
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(256, input_dim=input_dim, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.4),
        Dense(1, activation='linear')  # Saída para 1 variável: Casos
    ])

    model.compile(optimizer=Adam(learning_rate=0.0006), loss='mse', metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1, min_delta=0.001)
    checkpoint = ModelCheckpoint('Dados/melhor_modelo_mensal.keras', monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(
        x_train, y_train,
        epochs=200,
        batch_size=64,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    salvar_metricas_em_arquivo(history, "Dados/historico_metricas_mensais_configB.csv")

    return model, history

def salvar_dados_normalizados(X_scaled, y_scaled, encoder_columns, caminho_x="Dados/amostra_X_normalizado.xlsx", caminho_y="Dados/amostra_Y_normalizado.xlsx"):
    
    df_x = pd.DataFrame(X_scaled[:5], columns=encoder_columns)
    df_y = pd.DataFrame(y_scaled[:5], columns=['casos_normalizados'])
    
    df_x.to_excel(caminho_x, index=False)
    df_y.to_excel(caminho_y, index=False)
    
    logging.info(f"Amostra das variáveis de entrada normalizadas salva em {caminho_x}")
    logging.info(f"Amostra das variáveis de saída normalizadas salva em {caminho_y}")

def salvar_metricas_em_arquivo(history, caminho_arquivo="Dados/historico_metricas_mensais_configB.csv"):
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

def criar_entrada_mensal(ano, mes, estado, mortes, encoder_columns):
    entrada = pd.DataFrame(0, index=[0], columns=encoder_columns)
    
    # Atribui ano e mortes, se as colunas existirem no encoder
    if 'ano' in entrada.columns:
        entrada['ano'] = ano
    if 'mortes' in entrada.columns:
        entrada['mortes'] = mortes
    
    col_estado = f"estado_{estado}"
    col_mes = f"mes_{mes}"
    
    if col_estado in entrada.columns:
        entrada[col_estado] = 1
    if col_mes in entrada.columns:
        entrada[col_mes] = 1
        
    return entrada

def obter_estados_no_banco(df):
    return df['estado'].unique()

def obter_meses_no_banco(df):
    return sorted(df['mes'].unique())

def salvar_previsoes_em_csv(lista_previsoes, caminho_arquivo="Dados/previsoes_mensais.csv"):
    df_previsoes = pd.DataFrame(lista_previsoes)
    df_previsoes['timestamp'] = pd.Timestamp.now()

    if os.path.exists(caminho_arquivo):
        df_antigo = pd.read_csv(caminho_arquivo)
        df_final = pd.concat([df_antigo, df_previsoes], ignore_index=True)
    else:
        df_final = df_previsoes

    df_final.to_csv(caminho_arquivo, index=False)
    logging.info(f"Previsões salvas em {caminho_arquivo}")

def fazer_previsoes_mensais(model, scaler_X, scaler_y, encoder_columns, df, ano_previsao=2024):
    estados_no_banco = obter_estados_no_banco(df)
    meses_no_banco = obter_meses_no_banco(df)
    previsoes = []

    print(f"\n=== PREVISÕES MENSAL PARA {ano_previsao} ===\n")

    for estado in sorted(estados_no_banco):
        for mes in meses_no_banco:
            entrada_mensal = criar_entrada_mensal(ano_previsao, mes, estado, 0, encoder_columns)

            if entrada_mensal.isnull().values.any():
                logging.warning(f"Estado '{estado}' ou mês '{mes}' não possui coluna correspondente.")
                continue

            try:
                entrada_normalizada = scaler_X.transform(entrada_mensal)
            except NotFittedError:
                logging.error("Scaler_X não foi ajustado corretamente.")
                continue

            previsao_normalizada = model.predict(entrada_normalizada)
            previsao = scaler_y.inverse_transform(previsao_normalizada)

            casos = int(previsao[0][0])
            print(f"{estado} - {mes}/{ano_previsao}: Casos: {casos:,}")

            previsoes.append({
                "ano": ano_previsao,
                "mes": mes,
                "estado": estado,
                "casos_Previstos": casos,
            })

    salvar_previsoes_em_csv(previsoes, "Dados/previsoes_mensais.csv")
    
def salvar_tempo_execucao_csv(tempo_execucao, caminho_arquivo="Dados/tempo_execucao.csv"):
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
    
def salvar_metricas_execucao_csv(mse, mae, arquivo_metricas="Dados/metricas.csv"):
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
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Treinamento')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Histórico de Perda', fontsize=14)
    plt.xlabel('Época')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def executar_codigo():
    start_time = time.time()  # <- iniciar contagem aqui
    
    df = carregar_dados()
    if df is None:
        return

    X, y, encoder_columns = processar_dados(df)
    if X is None or y is None:
        return

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)

    model, history = criar_e_treinar_modelo(X_scaled, y_scaled, input_dim=X.shape[1])
    
    # Calcular métricas gerais
    score = model.evaluate(X_scaled, y_scaled, verbose=0)
    final_mse = score[0]
    final_mae = score[1]

    elapsed_time = time.time() - start_time  # <- calcula antes de salvar
    print(f"\nTempo de execução: {elapsed_time:.2f} segundos.")

    salvar_metricas_execucao_csv(final_mse, final_mae, arquivo_metricas="Dados/metricas.csv")
    
    salvar_tempo_execucao_csv(elapsed_time, caminho_arquivo="Dados/tempo_execucao.csv")

    fazer_previsoes_mensais(model, scaler_X, scaler_y, encoder_columns, df, ano_previsao=2024)
    
    plotar_grafico_perda(history)
    
    salvar_dados_normalizados(X_scaled, y_scaled, encoder_columns)

if __name__ == "__main__":
    start_time = time.time()
    print(f"\n=== Execução única ===")
    executar_codigo()
    elapsed_time = time.time() - start_time
    print(f"\nTempo de execução: {elapsed_time:.2f} segundos.")

    executar_codigo()