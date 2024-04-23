import requests
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import deque

# Variáveis globais para armazenar os dados
X = deque(maxlen=20)
y = deque(maxlen=20)
ultimo_id = None
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

# Função para calcular estatísticas dos últimos 20 números
def calculate_statistics(ultimos_numeros):
    last = list(ultimos_numeros)
    mean = np.mean(last)
    std_dev = np.std(last)
    maximum = np.max(last)
    minimum = np.min(last)
    variance = np.var(last)
    median = np.median(last)
    # Calculando a taxa de mudança nos últimos 10 pontos de crash
    change_rate_last_10 = last[-1] - last[-10]
    return [mean, std_dev, maximum, minimum, variance, median, change_rate_last_10]

# Função para analisar os últimos 20 números e calcular a probabilidade
def analyze_data(data):
    global X, y, resultado_atual
    # Obter os últimos 20 números
    ultimos_numeros = deque(maxlen=20)
    for game in data[:20]:
        ultimos_numeros.append(float(game["crash_point"]))

    # Calcular estatísticas dos últimos 20 números
    features = calculate_statistics(ultimos_numeros)

    # Adicionar os dados aos conjuntos de treinamento
    X.append(features)
    target = 1 if ultimos_numeros[-1] > 2 else 0
    y.append(target)

    # Treinar o modelo
    if len(X) >= 20:
        model = train_model()
        if model:
            # Prever a probabilidade do próximo número ser maior que 2
            probability_over_2 = model.predict_proba([features])[0][1]

            # Armazenar temporariamente o resultado atual
            resultado_atual = {
                "probability": probability_over_2,
                "real_number": data[0]['crash_point'],
                "target": target
            }

            # Imprimir o resultado atual
            print(f"Probabilidade do próximo número ser maior que 2: {resultado_atual['probability']:.2%} | Último número em tempo real: {resultado_atual['real_number']}")

# Função para treinar o modelo de classificação
def train_model():
    global X, y
    # Verificar se há dados suficientes para treinamento
    if len(X) < 20:
        return None
    
    # Treinar o modelo de classificação RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

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
