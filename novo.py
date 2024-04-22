import requests
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import deque

# Variáveis globais para armazenar os dados
X = deque(maxlen=40)
y = deque(maxlen=40)
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
    if len(X) < 40:
        print(f"Aguardando mais {40 - len(X)} rodadas para iniciar a probabilidade...")
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
    global X, y, acertos, erros, resultado_atual
    # Obter os últimos 40 números
    ultimos_40_numeros = deque(maxlen=40)
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

        # Armazenar temporariamente o resultado atual
        resultado_atual = {
            "probability": probability_over_2,
            "real_number": data[0]['crash_point'],
            "target": target
        }

# Loop principal
while True:
    # Solicitar dados da API
    api_data = get_api_data()
    
    # Se os dados foram obtidos com sucesso
    if api_data:
        # Analisar os dados
        analyze_data(api_data)
        
        # Se houver um resultado temporário, atualizar a contagem de acertos e erros
        if resultado_atual:
            # Imprimir o resultado atual
            print(f"Probabilidade do próximo número ser maior que 2: {resultado_atual['probability']:.2%} | Último número em tempo real: {resultado_atual['real_number']}")
            
            # Verificar se a previsão é correta e atualizar a contagem de acertos e erros
            if resultado_atual['probability'] > 0.5 and resultado_atual['real_number'] >= 2.0 and resultado_atual['target'] == 1:
                print("Acerto!")
                acertos += 1
            elif resultado_atual['probability'] <= 0.5 and resultado_atual['real_number'] < 2.0 and resultado_atual['target'] == 0:
                print("Acerto!")
                acertos += 1
            else:
                print("Erro!")
                erros += 1
                
            # Imprimir a contagem atualizada de acertos e erros
            print(f"Acertos: {acertos} | Erros: {erros}")
            
            # Limpar o resultado atual para a próxima iteração
            resultado_atual = None
    
    # Aguardar 5 segundos antes da próxima solicitação
    time.sleep(5)

