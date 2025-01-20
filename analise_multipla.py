import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from main import calcular_rsi, calcular_ichimoku, calcular_adx, calcular_atr
from main import analisar_ichimoku, analisar_adx_atr, calcular_macd, calcular_bandas_bollinger
from main import calcular_suporte_resistencia, calcular_tendencia_longo_prazo, analisar_momento_mercado
from main import realizar_backtesting

def analisar_multiplas_acoes(tickers, periodo_backtest="6mo"):
    """
    Analisa múltiplas ações e retorna um ranking das melhores oportunidades
    """
    resultados = []
    
    print(f"\n{'='*70}")
    print(f"ANÁLISE MÚLTIPLA DE AÇÕES")
    print(f"{'='*70}")
    
    for ticker in tickers:
        try:
            print(f"\nAnalisando {ticker}...")
            
            # Ajustar ticker para formato do Yahoo Finance
            yahoo_ticker = f"{ticker}.SA" if ticker.endswith('3') or ticker.endswith('4') else ticker
            
            # Obter dados
            ativo = yf.Ticker(yahoo_ticker)
            historico = ativo.history(period="3mo", interval="1d")
            
            if not historico.empty:
                # Preparar dados
                df = historico.copy()
                df['MM5'] = df['Close'].rolling(window=5).mean()
                df['MM20'] = df['Close'].rolling(window=20).mean()
                df['RSI'] = df['Close'].rolling(window=14).apply(lambda x: calcular_rsi(x))
                
                ultimo = df.iloc[-1]
                preco_atual = ultimo['Close']
                
                # Realizar backtesting
                ops_backtest = realizar_backtesting(ticker, periodo=periodo_backtest)
                if ops_backtest:
                    total_ops = len(ops_backtest)
                    ops_gain = len([op for op in ops_backtest if op['resultado'] > 0])
                    taxa_acerto = (ops_gain / total_ops) * 100
                    resultado_medio = sum(op['resultado'] for op in ops_backtest) / total_ops
                else:
                    taxa_acerto = 0
                    resultado_medio = 0
                
                # Analisar momento atual
                mensagens, pontos_compra, pontos_venda, stop_loss, stop_gain = analisar_momento_mercado(
                    df.rename(columns={'High': 'high', 'Low': 'low', 'Close': 'close'}),
                    ultimo['RSI'],
                    ultimo['MM5'],
                    ultimo['MM20']
                )
                
                # Calcular pontuação geral
                pontuacao = 0
                
                # Pontos pelo histórico de acerto
                pontuacao += (taxa_acerto / 20)  # Máximo 5 pontos
                pontuacao += (resultado_medio / 2)  # Máximo 5 pontos
                
                # Pontos pela análise atual
                if pontos_compra > pontos_venda:
                    pontuacao += pontos_compra
                    tipo_operacao = "COMPRA"
                    confianca = (pontos_compra / (pontos_compra + pontos_venda)) * 100
                else:
                    pontuacao -= pontos_venda
                    tipo_operacao = "VENDA"
                    confianca = (pontos_venda / (pontos_compra + pontos_venda)) * 100
                
                # Adicionar resultado
                resultados.append({
                    'ticker': ticker,
                    'preco_atual': preco_atual,
                    'tipo_operacao': tipo_operacao,
                    'pontuacao': pontuacao,
                    'confianca': confianca,
                    'taxa_acerto_backtest': taxa_acerto,
                    'resultado_medio_backtest': resultado_medio,
                    'rsi': ultimo['RSI'],
                    'stop_loss': stop_loss,
                    'stop_gain': stop_gain,
                    'mensagens': mensagens
                })
        
        except Exception as e:
            print(f"Erro ao analisar {ticker}: {str(e)}")
    
    # Ordenar resultados
    resultados.sort(key=lambda x: (-x['pontuacao'] if x['tipo_operacao'] == "COMPRA" else x['pontuacao']))
    
    # Mostrar resultados
    print(f"\n{'='*70}")
    print("RANKING DE OPORTUNIDADES")
    print(f"{'='*70}")
    
    for i, res in enumerate(resultados, 1):
        print(f"\n{i}. {res['ticker']}")
        print(f"Tipo: {res['tipo_operacao']}")
        print(f"Preço: R$ {res['preco_atual']:.2f}")
        print(f"Pontuação: {abs(res['pontuacao']):.1f}")
        print(f"Confiança: {res['confianca']:.1f}%")
        print(f"RSI: {res['rsi']:.1f}")
        print(f"Taxa de Acerto (Backtest): {res['taxa_acerto_backtest']:.1f}%")
        print(f"Resultado Médio (Backtest): {res['resultado_medio_backtest']:.2f}%")
        print(f"Stop Loss: R$ {res['stop_loss']:.2f}")
        print(f"Stop Gain: R$ {res['stop_gain']:.2f}")
        print("\nSinais Técnicos:")
        for msg in res['mensagens']:
            print(f"• {msg}")
    
    return resultados

if __name__ == "__main__":
    # Lista de ações para análise
    acoes = [
        "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3",
        "B3SA3", "RENT3", "BBAS3", "WEGE3", "RADL3",
        "JBSS3", "SUZB3", "GGBR4", "PRIO3", "RAIL3",
        "CSAN3", "VIVT3", "BBSE3", "UGPA3", "CCRO3",
        "CVCB3", "MGLU3", "VIIA3", "COGN3", "CIEL3"
    ]
    
    print("\nIniciando análise múltipla de ações...")
    melhores_oportunidades = analisar_multiplas_acoes(acoes)
    
    # Salvar resultados em um arquivo
    data_hora = datetime.now().strftime('%Y%m%d_%H%M')
    with open(f'ranking_acoes_{data_hora}.txt', 'w', encoding='utf-8') as f:
        f.write(f"Ranking de Ações - {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n")
        for i, res in enumerate(melhores_oportunidades, 1):
            f.write(f"{i}. {res['ticker']} - {res['tipo_operacao']} (Confiança: {res['confianca']:.1f}%)\n")
            f.write(f"   Preço: R$ {res['preco_atual']:.2f}\n")
            f.write(f"   Stop Loss: R$ {res['stop_loss']:.2f}\n")
            f.write(f"   Stop Gain: R$ {res['stop_gain']:.2f}\n\n")
    
    print(f"\nRanking salvo em: ranking_acoes_{data_hora}.txt") 