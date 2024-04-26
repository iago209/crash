import requests
import time
from collections import deque
from sklearn.linear_model import LinearRegression
import numpy as np

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

# Função para calcular estatísticas dos últimos 15 números
def calculate_statistics(ultimos_15_numeros):
    last_15 = list(ultimos_15_numeros)
    mean = np.mean(last_15)
    std_dev = np.std(last_15)
    maximum = np.max(last_15)
    minimum = np.min(last_15)
    variance = np.var(last_15)
    median = np.median(last_15)
    return [mean, std_dev, maximum, minimum, variance, median]

# Função para analisar a tendência dos dados e prever o próximo crash_point
def analyze_trend(crash_points, statistics):
    X = np.arange(len(crash_points)).reshape(-1, 1)  # Cria uma matriz de features
    for i in range(len(statistics)):
        X = np.hstack((X, np.full((len(crash_points), 1), statistics[i])))  # Adiciona cada estatística como característica adicional
    y = np.log(np.array(crash_points) + 1e-9)  # Aplica transformação logarítmica

    model = LinearRegression()
    model.fit(X, y)

    next_crash = np.exp(model.predict(X[-1].reshape(1, -1))[0])  # Previsão do próximo crash_point
    max_crash = np.exp(max(y)) - 1e-9  # Valor máximo esperado
    min_crash = np.exp(min(y)) - 1e-9  # Valor mínimo esperado

    return next_crash, max_crash, min_crash

# Mantém um deque de 15 crash_points mais recentes
deque_size = 15
crash_points_deque = deque(maxlen=deque_size)
last_id = None
model = None

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
        
        # Calcula estatísticas dos últimos 15 números
        statistics = calculate_statistics(crash_points_deque)

         # Exibe apenas o crash_point mais recente
        if crash_points_deque:
            print("Crash_point mais recente: {:.2f}".format(crash_points_deque[0]))

        # Verifica se o deque está cheio antes de fazer a análise
        if len(crash_points_deque) == deque_size:
            # Se não houver modelo treinado, ou seja, na primeira iteração, ou se o modelo foi atualizado anteriormente, re-treina o modelo
            if model is None or len(crash_points_deque) == deque_size:
                model = analyze_trend(crash_points_deque, statistics)

            # Prever o próximo crash_point
            next_crash, max_crash, min_crash = analyze_trend(crash_points_deque, statistics)
            
            # Exibe resultados no terminal
            print("Próximo crash_point previsto: {:.2f}".format(next_crash))
            print("Máximo crash: {:.2f}".format(max_crash))
            print("Mínimo crash: {:.2f}".format(min_crash))
        
    # Aguarda 5 segundos antes de fazer uma nova solicitação à API
    time.sleep(5)
