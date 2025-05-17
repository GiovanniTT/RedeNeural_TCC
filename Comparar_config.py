import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# Carregar os dois históricos
df_a = pd.read_csv("Dados/historico_metricas_mensais_configA.csv")
df_b = pd.read_csv("Dados/historico_metricas_mensais_configB.csv")

# Adicionar identificador de execução se não existir
if "execucao_id" not in df_a.columns:
    df_a["execucao_id"] = (df_a.index // 25)
    df_b["execucao_id"] = (df_b.index // 25)

# Obter a menor val_mae e val_loss de cada execução
melhores_a_mae = df_a.groupby("execucao_id")["val_mae"].min()
melhores_b_mae = df_b.groupby("execucao_id")["val_mae"].min()

melhores_a_loss = df_a.groupby("execucao_id")["val_loss"].min()
melhores_b_loss = df_b.groupby("execucao_id")["val_loss"].min()

# Testes estatísticos
stat_mae, p_mae = ttest_rel(melhores_a_mae, melhores_b_mae)
stat_loss, p_loss = ttest_rel(melhores_a_loss, melhores_b_loss)

print(f"Val_MAE - T-statistic: {stat_mae:.3f}, p-valor: {p_mae:.4f}")
print(f"Val_LOSS - T-statistic: {stat_loss:.3f}, p-valor: {p_loss:.4f}")
print(f"Config A - Média Val_MAE: {melhores_a_mae.mean():.4f}, Config B - Média Val_MAE: {melhores_b_mae.mean():.4f}")
print(f"Config A - Média Val_LOSS: {melhores_a_loss.mean():.4f}, Config B - Média Val_LOSS: {melhores_b_loss.mean():.4f}")

# DataFrame para boxplots
df_plot = pd.DataFrame({
    "Config A MAE": melhores_a_mae.values,
    "Config B MAE": melhores_b_mae.values,
    "Config A Loss": melhores_a_loss.values,
    "Config B Loss": melhores_b_loss.values
})

# Plot
plt.figure(figsize=(12,6))
sns.boxplot(data=df_plot)
plt.title("Comparação de val_mae e val_loss entre Configurações")
plt.ylabel("Valor")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
