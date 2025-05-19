import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados de treinamento
try:
    df = pd.read_csv("Dados/historico_metricas_mensais_configA.csv")
    
    # Encontrar a época com menor val_loss
    melhor_epoca = df.loc[df['val_loss'].idxmin()]
    epoch_num = int(melhor_epoca['epoch'])
    val_loss_min = melhor_epoca['val_loss']
    train_loss = melhor_epoca['loss']
    
    print("═"*50)
    print(f"MELHOR ÉPOCA: {epoch_num}")
    print("═"*50)
    print(f"Menor val_loss: {val_loss_min:.6f}")
    print(f"MSE Treino correspondente: {train_loss:.6f}")
    print(f"Diferença Treino-Validação: {abs(train_loss-val_loss_min):.6f}")
    print("\nMétricas completas:")
    print(melhor_epoca.to_string())

    # Gráfico de evolução
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['val_loss'], 'r-', label='Validação', alpha=0.7)
    plt.plot(df['epoch'], df['loss'], 'b-', label='Treino', alpha=0.7)
    plt.scatter(epoch_num, val_loss_min, color='gold', s=200, 
                edgecolor='black', label=f'Melhor (Época {epoch_num})')
    plt.axvline(epoch_num, color='green', linestyle='--', alpha=0.5)
    
    plt.title(f'Evolução do MSE (Melhor época: {epoch_num})')
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('melhor_epoca.png', dpi=300)
    plt.show()

except FileNotFoundError:
    print("Arquivo não encontrado. Verifique o caminho:")
    print("Caminho atual tentado: Dados/historico_metricas_mensais_configA.csv")
except Exception as e:
    print(f"Erro inesperado: {str(e)}")