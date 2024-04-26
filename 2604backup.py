import requests
import time
from collections import deque
from sklearn.linear_model import LinearRegression

# Criação de uma sessão para reutilizar conexões HTTP
sessao = requests.Session()

# Função para solicitar a API de Crash e obter os dados
def get_crash_data():
    url = "https://blaze1.space/api/crash_games/recent"
    try:
        response = sessao.get(url, timeout=5)  # Define um tempo limite de 5 segundos para a solicitação
        response.raise_for_status()  # Verifica se houve erro na requisição
        crash_data = response.json()
        return crash_data
    except requests.exceptions.RequestException as e:
        print("Erro ao obter dados da API:", e)
        return None

import numpy as np

# Função para analisar a tendência dos dados e prever o próximo crash_point
def analyze_trend(crash_points):
    X = np.arange(len(crash_points)).reshape(-1, 1)  # Cria uma matriz de features
    y = np.log(np.array(crash_points) + 1e-9)  # Aplica transformação logarítmica

    model = LinearRegression()
    model.fit(X, y)

    next_crash = np.exp(model.predict([[len(crash_points)]])[0])  # Previsão do próximo crash_point
    max_crash = np.exp(max(y)) - 1e-9  # Valor máximo esperado
    min_crash = np.exp(min(y)) - 1e-9  # Valor mínimo esperado

    return next_crash, max_crash, min_crash

# Mantém um deque de 15 crash_points mais recentes
deque_size = 15
crash_points_deque = deque(maxlen=deque_size)
last_id = None

while True:
    # Obtem dados mais recentes da API
    crash_data = get_crash_data()

    # Extrai crash_points e atualiza o deque
    if crash_data:
        for crash in crash_data:
            if crash['id'] != last_id:  # Verifica se o ID é diferente do anterior
                crash_point = float(crash['crash_point'])
                crash_points_deque.append(crash_point)
                last_id = crash['id']
        
        # Exibe apenas o crash_point mais recente
        if crash_points_deque:
            print("Crash_point mais recente: {:.2f}".format(crash_points_deque[0]))

        # Verifica se o deque está cheio antes de fazer a análise
        if len(crash_points_deque) == deque_size:
            # Analisa a tendência e prevê o próximo crash_point
            next_crash, max_crash, min_crash = analyze_trend(crash_points_deque)
            
            # Exibe resultados no terminal
            print("Próximo crash_point previsto: {:.2f}".format(next_crash))
            print("Máximo crash: {:.2f}".format(max_crash))
            print("Mínimo crash: {:.2f}".format(min_crash))
        
    # Aguarda 5 segundos antes de fazer uma nova solicitação à API
    time.sleep(5)
