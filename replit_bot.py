from flask import Flask
from threading import Thread
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from main import calcular_rsi, calcular_ichimoku, calcular_adx, calcular_atr
from main import analisar_ichimoku, analisar_adx_atr, calcular_macd, calcular_bandas_bollinger
from main import calcular_suporte_resistencia, calcular_tendencia_longo_prazo, analisar_momento_mercado
from main import TelegramNotifier

# Criar app Flask para manter o bot vivo
app = Flask('')

@app.route('/')
def home():
    return "Bot de Análise está rodando!"

def run():
    app.run(host='0.0.0.0', port=8080)

def manter_vivo():
    t = Thread(target=run)
    t.start()

def realizar_analise_continua(ticker, intervalo_minutos=30):
    """
    Realiza análise contínua de uma ação específica
    """
    notifier = TelegramNotifier()
    
    # Ajustar ticker para formato do Yahoo Finance
    yahoo_ticker = f"{ticker}.SA" if ticker.endswith('3') or ticker.endswith('4') else ticker
    
    mensagem_inicial = f"""
🤖 <b>Bot de Análise Iniciado no Replit</b>

📈 Ativo: {ticker}
⏰ Hora de Início: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
🔄 Intervalo de Análise: {intervalo_minutos} minutos

Status: Monitorando...
"""
    notifier.enviar_mensagem(mensagem_inicial)
    
    # Cache para armazenar dados históricos
    cache_historico = None
    ultima_atualizacao = None
    erros_consecutivos = 0
    max_erros = 3
    
    while True:
        try:
            # Mostrar hora atual
            hora_atual = datetime.now()
            print(f"\n{'='*50}")
            print(f"Análise Técnica - {ticker}")
            print(f"Data/Hora: {hora_atual.strftime('%Y-%m-%d %H:%M:%S')}")
            print('='*50)
            
            # Verificar se precisa atualizar o cache (a cada 15 minutos)
            precisa_atualizar = (
                cache_historico is None or
                ultima_atualizacao is None or
                (hora_atual - ultima_atualizacao).total_seconds() > 900  # 15 minutos
            )
            
            if precisa_atualizar:
                print("Atualizando dados do Yahoo Finance...")
                try:
                    # Obter dados do Yahoo Finance com retry
                    for tentativa in range(3):
                        try:
                            ativo = yf.Ticker(yahoo_ticker)
                            novo_historico = ativo.history(period="1y", interval="1d")
                            if not novo_historico.empty:
                                break
                        except Exception as e:
                            print(f"Tentativa {tentativa + 1} falhou: {str(e)}")
                            time.sleep(5)
                    
                    if not novo_historico.empty:
                        cache_historico = novo_historico
                        ultima_atualizacao = hora_atual
                        erros_consecutivos = 0
                    else:
                        raise Exception("Não foi possível obter dados após 3 tentativas")
                except Exception as e:
                    erros_consecutivos += 1
                    if cache_historico is None:
                        raise Exception(f"Erro ao obter dados: {str(e)}")
                    print("Aviso: Usando dados em cache devido a erro na atualização")
            
            if erros_consecutivos >= max_erros:
                print(f"Erro: {max_erros} falhas consecutivas ao obter dados. Verificando conexão...")
                time.sleep(300)  # Espera 5 minutos antes de tentar novamente
                continue
            
            if cache_historico is not None and not cache_historico.empty:
                # Usar dados do cache
                df = cache_historico.copy()
                # ... resto do código de análise ...
                
                # Aguardar intervalo definido com feedback mais frequente
                tempo_total = intervalo_minutos * 60
                for segundo in range(tempo_total, 0, -1):
                    minutos = segundo // 60
                    segundos = segundo % 60
                    print(f"Próxima análise em: {minutos:02d}:{segundos:02d} (Última: {ultima_atualizacao.strftime('%H:%M:%S')})", end='\r')
                    time.sleep(1)
            
        except Exception as e:
            print(f"Erro: {str(e)}")
            erros_consecutivos += 1
            tempo_espera = min(300 * erros_consecutivos, 3600)  # Máximo 1 hora
            print(f"Tentando novamente em {tempo_espera/60:.0f} minutos...")
            time.sleep(tempo_espera)

if __name__ == "__main__":
    # Iniciar servidor web para manter o bot vivo
    manter_vivo()
    
    print("Bot de Análise de Ações - Versão Replit")
    print("="*40)
    
    # Configurações do bot
    TICKER = "PETR4"  # Você pode mudar isso nas variáveis de ambiente do Replit
    INTERVALO = 30    # Intervalo em minutos
    
    print(f"\nIniciando análise contínua para {TICKER}")
    print(f"Intervalo entre análises: {INTERVALO} minutos")
    print("\nBot iniciado! Pressione Ctrl+C para parar\n")
    
    try:
        realizar_analise_continua(TICKER, INTERVALO)
    except KeyboardInterrupt:
        print("\nBot encerrado pelo usuário.") 