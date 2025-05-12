import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# Carregar os dois históricos
df_a = pd.read_csv("Dados/historico_metricas_configA.csv")
df_b = pd.read_csv("Dados/historico_metricas_configB.csv")

# Adicionar identificador de execução se não existir
if "execucao_id" not in df_a.columns:
    df_a["execucao_id"] = (df_a.index // 25)
    df_b["execucao_id"] = (df_b.index // 25)

# Obter a menor val_mae de cada execução
melhores_a = df_a.groupby("execucao_id")["val_mae"].min()
melhores_b = df_b.groupby("execucao_id")["val_mae"].min()

# Teste estatístico
stat, p = ttest_rel(melhores_a, melhores_b)
print(f"T-statistic: {stat:.3f}, p-valor: {p:.4f}")

# Boxplot
df_plot = pd.DataFrame({
    "Config A": melhores_a.values,
    "Config B": melhores_b.values
})
sns.boxplot(data=df_plot)
plt.title("Comparação de val_mae entre Configurações")
plt.ylabel("Melhor val_mae por execução")
plt.grid(True)
plt.show()
