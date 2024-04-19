import requests
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import deque

# Variáveis globais para armazenar os dados
X = []
y = []
ultimo_id = None
rodadas_restantes = 40
acertos = 0
erros = 0

# Lista para armazenar os últimos 40 números
ultimos_40_numeros = deque(maxlen=40)

# Criação de uma sessão para reutilizar conexões HTTP
sessao = requests.Session()

# Função para solicitar a API
def get_api_data():
    global ultimo_id
    url = "https://blaze1.space/api/crash_games/recent"
    response = sessao.get(url, timeout=5)  # Define um tempo limite de 5 segundos para a solicitação
    if response.status_code == 200:
        data = response.json()
        # Verificar se o ID mais recente é diferente do último
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
    if len(X) < 40:
        print(f"Aguardando mais {rodadas_restantes} rodadas para iniciar a probabilidade...")
        return None
    
    # Treinar o modelo de regressão logística
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Função para calcular estatísticas dos últimos 40 números
def calculate_statistics(ultimos_40_numeros):
    last_40 = list(ultimos_40_numeros)
    mean = np.mean(last_40)
    std_dev = np.std(last_40)
    maximum = np.max(last_40)
    minimum = np.min(last_40)
    variance = np.var(last_40)
    median = np.median(last_40)
    return [mean, std_dev, maximum, minimum, variance, median]

# Função para analisar os últimos 40 números e calcular a probabilidade
def analyze_data(data):
    global X, y, rodadas_restantes, ultimos_40_numeros, acertos, erros
    # Verificar se há dados suficientes para análise
    if len(data) < 40:
        rodadas_restantes -= 1
        primeiro_numero = data[0]['crash_point'] if data else "Nenhum dado disponível"
        print(f"Aguardando mais {rodadas_restantes} rodadas para iniciar a probabilidade... Último número em tempo real: {primeiro_numero}")
        return
    
    # Resetar o contador de rodadas restantes
    rodadas_restantes = 40
    
    # Obter os últimos 40 números
    for game in data[:40]:
        ultimos_40_numeros.append(float(game["crash_point"]))

    # Calcular estatísticas dos últimos 40 números
    features = calculate_statistics(ultimos_40_numeros)

    # Adicionar os dados aos conjuntos de treinamento
    X.append(features)
    target = 1 if ultimos_40_numeros[-1] > 2 else 0
    y.append(target)

    # Treinar o modelo
    model = train_model()

    # Se o modelo for treinado com sucesso
    if model:
        # Prever a probabilidade do próximo número ser maior que 2
        probability_over_2 = model.predict_proba([features])[0][1]

        # Imprimir resultados
        print(f"Probabilidade do próximo número ser maior que 2: {probability_over_2:.2%} | Último número em tempo real: {ultimos_40_numeros[-1]}")

        # Verificar se a previsão é correta
        if probability_over_2 > 0.5 and target == 1:
            print("Acerto!")
            acertos += 1
        elif probability_over_2 <= 0.5 and target == 0:
            print("Acerto!")
            acertos += 1
        else:
            print("Erro!")
            erros += 1

        # Imprimir contagem de acertos e erros
        print(f"Acertos: {acertos} | Erros: {erros}")

# Loop principal
while True:
    # Solicitar dados da API
    api_data = get_api_data()
    
    # Se os dados foram obtidos com sucesso
    if api_data:
        # Obter os últimos 40 números
        for game in api_data[:40]:
            ultimos_40_numeros.append(float(game["crash_point"]))
        
        # Analisar os dados
        analyze_data(api_data)
    
    # Aguardar 5 segundos antes da próxima solicitação
    time.sleep(5)

