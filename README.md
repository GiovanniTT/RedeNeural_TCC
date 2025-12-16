Predição de Casos de Dengue no Brasil com Redes Neurais (MLP e LSTM)

Este repositório contém o código desenvolvido no Trabalho de Conclusão de Curso (TCC) que realiza uma análise comparativa entre modelos de Redes Neurais Artificiais MLP (Multilayer Perceptron) e LSTM (Long Short-Term Memory) aplicados à predição de casos de dengue no Brasil, utilizando dados epidemiológicos e climáticos.

📌 Objetivo do Projeto

Desenvolver, treinar e avaliar modelos de aprendizado de máquina capazes de prever casos mensais de dengue em todos os estados brasileiros, considerando variáveis climáticas e epidemiológicas, além de comparar:

- Precisão dos modelos
- Métricas de erro (MAE e MSE)
- Tempo de execução

🧠 Modelos Utilizados

- MLP (Multilayer Perceptron)
  Modelo feedforward utilizado para regressão, com camadas densas, regularização L2, dropout e early stopping.

- LSTM (Long Short-Term Memory)
  Modelo recorrente aplicado a séries temporais, utilizando camadas LSTM bidirecionais para capturar dependências temporais.

📊 Resultados Principais

Os experimentos demonstraram que o modelo MLP apresentou melhor desempenho geral em relação ao LSTM:

- MSE: 0,00126 (24,5% menor que o LSTM)
- MAE: 0,00525 (20,5% menor que o LSTM)
- Tempo de execução: aproximadamente 31,4% mais rápido

Esses resultados indicam que, para a série histórica analisada (2014–2023), o MLP foi mais eficiente e preciso.

🗂️ Fontes de Dados

Os dados utilizados no projeto foram obtidos a partir de fontes oficiais:

- DATASUS – Casos e óbitos por dengue
- INMET – Dados climáticos (temperatura média e precipitação)

Período analisado: 2014 a 2023
Abrangência: Todos os estados brasileiros

🔄 Pipeline do Projeto

- Coleta de dados epidemiológicos e climáticos
- Tratamento e limpeza dos dados (ETL)
- Normalização e codificação de variáveis
- Treinamento dos modelos MLP e LSTM
- Avaliação com MAE, MSE e tempo de execução
- Geração de previsões mensais para 2024
- Exportação dos resultados em arquivos CSV

▶️ Como Executar o Projeto

Clone o repositório:

- git clone https://github.com/GiovanniTT/RedeNeural_TCC.git

Instale as dependências:

- pip install -r requirements.txt

Configure o banco de dados MySQL com os dados tratados.

Execute os scripts de treinamento:

- python treinamento_mlp.py
- python treinamento_lstm.py
