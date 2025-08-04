import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Leitura dos dados
df_lstm = pd.read_csv(r'C:\Users\gi-to\OneDrive\Documentos\Área de Trabalho\RedeNeural\Dados\tempo_execucao_lstm.csv')
df_mlp = pd.read_csv(r'C:\Users\gi-to\OneDrive\Documentos\Área de Trabalho\RedeNeural\Dados\tempo_execucao.csv')

# Supondo que a coluna com tempo de execução se chama 'tempo'
# Ajuste se o nome for outro
tempo_lstm = df_lstm['tempo_execucao_segundos'].mean()  # média dos tempos LSTM
tempo_mlp = df_mlp['tempo_execucao_segundos'].mean()    # média dos tempos MLP

# Dados para o gráfico
modelos = ['LSTM', 'MLP']
tempos_execucao = [tempo_lstm, tempo_mlp]

# Configurações estéticas
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))

# Criar gráfico de barras
cores = ['blue', 'red']
bars = sns.barplot(x=modelos, y=tempos_execucao, palette=cores)

# Adicionar valores no topo das barras
for bar in bars.patches:
    height = bar.get_height()
    bars.annotate(f'{height:.2f}s',
                  xy=(bar.get_x() + bar.get_width() / 2, height),
                  xytext=(0, 5),  # deslocamento vertical
                  textcoords="offset points",
                  ha='center', va='bottom',
                  fontsize=12)

# Títulos e labels
plt.title('Gráfico 2 – Comparação do tempo de Execução dos Modelos', fontsize=16, weight='bold')
plt.xlabel('Modelos', fontsize=14)
plt.ylabel('Tempo de Execução (segundos)', fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()
