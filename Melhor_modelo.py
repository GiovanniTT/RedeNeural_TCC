import os
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Configurações
MODEL_PATH = "Dados/melhor_modelo_mensal.keras"
SCALER_X_PATH = "Dados/scaler_X.pkl"
SCALER_Y_PATH = "Dados/scaler_y.pkl"
HISTORICO_A_PATH = "Dados/historico_metricas_mensais_configA.csv"
HISTORICO_B_PATH = "Dados/historico_metricas_mensais_configB.csv"

def comparar_historicos():
    print("\n🔍 Comparando arquivos de histórico...")
    
    # Carregar ambos os históricos se existirem
    historicos = {}
    for path, nome in [(HISTORICO_A_PATH, "Configuração A"), 
                      (HISTORICO_B_PATH, "Configuração B")]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                historicos[nome] = df
                print(f"\n✅ {nome} carregado com sucesso")
                print(f"- Registros: {len(df)}")
                print(f"- Métricas disponíveis: {list(df.columns)}")
            except Exception as e:
                print(f"\n❌ Erro ao carregar {path}: {e}")
        else:
            print(f"\n⚠ Arquivo {path} não encontrado")
    
    # Comparação detalhada se ambos existirem
    if len(historicos) == 2:
        print("\n" + "="*50 + "\n📊 Análise Comparativa\n" + "="*50)
        
        # 1. Comparação básica de métricas
        print("\n🔢 Estatísticas Comparativas:")
        for metrica in historicos['Configuração A'].columns:
            print(f"\nMétrica: {metrica}")
            print("Config A | Config B")
            print("------------------")
            print(f"{historicos['Configuração A'][metrica].mean():.4f} | {historicos['Configuração B'][metrica].mean():.4f} (Média)")
            print(f"{historicos['Configuração A'][metrica].min():.4f} | {historicos['Configuração B'][metrica].min():.4f} (Mínimo)")
            print(f"{historicos['Configuração A'][metrica].max():.4f} | {historicos['Configuração B'][metrica].max():.4f} (Máximo)")
        
        # 2. Comparação visual simplificada
        print("\n📈 Tendências:")
        for metrica in historicos['Configuração A'].columns:
            diff = historicos['Configuração A'][metrica].mean() - historicos['Configuração B'][metrica].mean()
            if diff > 0:
                print(f"- {metrica}: Config A é {abs(diff):.4f} maior em média")
            else:
                print(f"- {metrica}: Config B é {abs(diff):.4f} maior em média")
        
        # 3. Sugestão baseada nos dados
        melhor = {}
        for metrica in historicos['Configuração A'].columns:
            if 'loss' in metrica or 'mae' in metrica:  # Quanto menor, melhor
                melhor[metrica] = 'A' if historicos['Configuração A'][metrica].mean() < historicos['Configuração B'][metrica].mean() else 'B'
            else:  # Quanto maior, melhor (accuracy, etc)
                melhor[metrica] = 'A' if historicos['Configuração A'][metrica].mean() > historicos['Configuração B'][metrica].mean() else 'B'
        
        print("\n🏆 Melhor configuração por métrica:")
        for metrica, config in melhor.items():
            print(f"- {metrica}: Config {config}")

def verificar_recursos():
    print("\n" + "="*50 + "\n🔧 Verificando recursos adicionais\n" + "="*50)
    
    # Verificar scalers e modelo (mantido da versão original)
    for scaler_path, nome in [(SCALER_X_PATH, "Scaler X"), (SCALER_Y_PATH, "Scaler y")]:
        if os.path.exists(scaler_path):
            try:
                joblib.load(scaler_path)
                print(f"\n✅ {nome} encontrado em {scaler_path}")
            except Exception as e:
                print(f"\n❌ Erro ao carregar {scaler_path}: {e}")
        else:
            print(f"\n⚠ {nome} não encontrado em {scaler_path}")
    
    if os.path.exists(MODEL_PATH):
        try:
            load_model(MODEL_PATH)
            print(f"\n✅ Modelo encontrado em {MODEL_PATH}")
        except Exception as e:
            print(f"\n❌ Erro ao carregar modelo: {e}")
    else:
        print(f"\n⚠ Modelo não encontrado em {MODEL_PATH}")

if __name__ == "__main__":
    comparar_historicos()
    verificar_recursos()