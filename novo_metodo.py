import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Criação de uma sessão para reutilizar conexões HTTP
sessao = requests.Session()

# Variáveis globais para armazenar os dados
X = []
y = []

# Função para solicitar a API
def get_api_data():
    url = "https://blaze1.space/api/crash_games/all"
    response = sessao.get(url, timeout=5)
    if response.status_code == 200:
        return response.json()
    else:
        print("Falha ao obter dados da API")
        return None

# Função para calcular estatísticas dos últimos números
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

# Função para analisar os dados e treinar o modelo
def analyze_and_train(data):
    global X, y
    # Adicionar os novos dados aos conjuntos de treinamento
    for game in data:
        ultimos_numeros = [float(game["crash_point"]) for game in data]
        features = calculate_statistics(ultimos_numeros)
        target = 1 if ultimos_numeros[-1] > 2 else 0
        X.append(features)
        y.append(target)

    # Verificar se há dados suficientes para treinar o modelo
    if len(X) < 100:
        print("Aguardando mais dados para treinar o modelo...")
        return

    # Treinar o modelo
    model = train_model(X, y)
    
    # Prever a probabilidade do próximo número ser maior que 2
    probability_over_2 = model.predict_proba([features])[0][1]
    
    # Se a probabilidade for alta e a certeza for grande, enviar mensagem
    if probability_over_2 >= 0.8:
        print(f"Provável crash alto! Probabilidade: {probability_over_2:.2%}")
        print(f"Crash atual: {data[-1]['crash_point']}x")
        # Implemente a lógica para enviar uma mensagem de notificação aqui

# Função para treinar o modelo de classificação
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Loop principal
while True:
    # Solicitar dados da API
    api_data = get_api_data()
    
    # Se os dados foram obtidos com sucesso
    if api_data:
        # Analisar os dados e treinar o modelo
        analyze_and_train(api_data)
