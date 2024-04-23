import requests
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import deque

# Variáveis globais para armazenar os dados
X = deque(maxlen=20)
y = deque(maxlen=20)
ultimo_id = None
acertos = 0
erros = 0
resultado_atual = None

# Criação de uma sessão para reutilizar conexões HTTP
sessao = requests.Session()

# Função para solicitar a API
def get_api_data():
    global ultimo_id
    url = "https://blaze1.space/api/crash_games/recent"
    response = sessao.get(url, timeout=5)
    if response.status_code == 200:
        data = response.json()
        if data and data[0]["id"] != ultimo_id:
            ultimo_id = data[0]["id"]
            return data
        else:
            return None
    else:
        print("Falha ao obter dados da API")
        return None

# Função para treinar o modelo de regressão logística
def train_model():
    global X, y
    # Verificar se há dados suficientes para treinamento
    if len(X) < 20:
        print(f"Aguardando mais {20 - len(X)} rodadas para iniciar a probabilidade...")
        return None
    else:
        print("Iniciando cálculo da probabilidade...")
    
    # Converter X e y para listas para treinamento do modelo
    X_train = list(X)
    y_train = list(y)
    
    # Treinar o modelo de regressão logística
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Função para calcular estatísticas dos últimos 20 números
def calculate_statistics(ultimos_20_numeros):
    last_20 = list(ultimos_20_numeros)
    mean = np.mean(last_20)
    std_dev = np.std(last_20)
    maximum = np.max(last_20)
    minimum = np.min(last_20)
    variance = np.var(last_20)
    median = np.median(last_20)
    # Calculando a taxa de mudança nos últimos 10 pontos de crash
    change_rate_last_10 = last_20[-1] - last_20[-10]
    return [mean, std_dev, maximum, minimum, variance, median, change_rate_last_10]

# Função para analisar os últimos 20 números e calcular a probabilidade
def analyze_data(data):
    global X, y, acertos, erros, resultado_atual
    # Obter os últimos 20 números
    ultimos_20_numeros = deque(maxlen=20)
    for game in data[:20]:
        ultimos_20_numeros.append(float(game["crash_point"]))

    # Calcular estatísticas dos últimos 20 números
    features = calculate_statistics(ultimos_20_numeros)

    # Adicionar os dados aos conjuntos de treinamento
    X.append(features)
    target = 1 if ultimos_20_numeros[-1] > 2 else 0
    y.append(target)

    # Treinar o modelo
    model = train_model()

    # Se o modelo for treinado com sucesso
    if model:
        # Prever a probabilidade do próximo número ser maior que 2
        probability_over_2 = model.predict_proba([features])[0][1]

        # Armazenar temporariamente o resultado atual
        resultado_atual = {
            "probability": probability_over_2,
            "real_number": data[0]['crash_point'],
            "target": target
        }

        # Verificar se a previsão é correta e atualizar a contagem de acertos e erros
        if probability_over_2 > 0.5 and float(data[1]['crash_point']) >= 2.0:
            print("Acerto!")
            acertos += 1
        elif probability_over_2 > 0.5 and float(data[1]['crash_point']) < 2.0:
            print("Erro!")
            acertos += 1
            
        # Imprimir a contagem atualizada de acertos e erros
        print(f"Acertos: {acertos} | Erros: {erros}")

# Loop principal
while True:
    # Solicitar dados da API
    api_data = get_api_data()
    
    # Se os dados foram obtidos com sucesso
    if api_data:
        # Analisar os dados
        analyze_data(api_data)
    
    # Aguardar 5 segundos antes da próxima solicitação
    time.sleep(5)
