import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar os CSVs
df_mlp = pd.read_csv(r'C:\Users\gi-to\OneDrive\Documentos\Área de Trabalho\RedeNeural\Dados\metricas.csv')
df_lstm = pd.read_csv(r'C:\Users\gi-to\OneDrive\Documentos\Área de Trabalho\RedeNeural\Dados\metricas_lstm.csv')

# Extrair métricas
mse_mlp = df_mlp['mse_teste'].iloc[0]
mae_mlp = df_mlp['mae_teste'].iloc[0]
mse_lstm = df_lstm['mse_teste'].iloc[0]
mae_lstm = df_lstm['mae_teste'].iloc[0]

# Criar DataFrame para plotagem
df_metricas = pd.DataFrame({
    'Modelo': ['MLP', 'LSTM'],
    'MSE': [mse_mlp, mse_lstm],
    'MAE': [mae_mlp, mae_lstm]
})

# Configurações do gráfico
plt.figure(figsize=(10, 6))
width = 0.35  # Largura das barras
x = np.arange(len(df_metricas['Modelo']))  # Posições no eixo x

# Criar barras
bars1 = plt.bar(x - width/2, df_metricas['MSE'], width, label='MSE', color='red', edgecolor='black')
bars2 = plt.bar(x + width/2, df_metricas['MAE'], width, label='MAE', color='blue', edgecolor='black')

# Adicionar detalhes
plt.ylabel('Valor do Erro', fontsize=12)
plt.xticks(x, df_metricas['Modelo'], fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionar valores nas barras
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.5f}',
                 ha='center', va='bottom', fontsize=10)

add_value_labels(bars1)
add_value_labels(bars2)

# Ajustar layout e mostrar
plt.tight_layout()
plt.show()