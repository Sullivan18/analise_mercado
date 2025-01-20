import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from main import calcular_rsi, calcular_ichimoku, calcular_adx, calcular_atr
from main import analisar_ichimoku, analisar_adx_atr, calcular_macd, calcular_bandas_bollinger
from main import calcular_suporte_resistencia, calcular_tendencia_longo_prazo, analisar_momento_mercado
from main import TelegramNotifier

def realizar_analise_continua(ticker, intervalo_minutos=30):
    """
    Realiza análise contínua de uma ação específica
    """
    notifier = TelegramNotifier()
    
    # Ajustar ticker para formato do Yahoo Finance
    yahoo_ticker = f"{ticker}.SA" if ticker.endswith('3') or ticker.endswith('4') else ticker
    
    mensagem_inicial = f"""
🤖 <b>Bot de Análise Iniciado</b>

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
                # Obter dados do Yahoo Finance
                ativo = yf.Ticker(yahoo_ticker)
                novo_historico = ativo.history(period="1y", interval="1d")
                
                if not novo_historico.empty:
                    cache_historico = novo_historico
                    ultima_atualizacao = hora_atual
                    erros_consecutivos = 0
                else:
                    erros_consecutivos += 1
                    if cache_historico is None:
                        raise Exception("Não foi possível obter dados do Yahoo Finance")
                    print("Aviso: Usando dados em cache devido a erro na atualização")
            
            if erros_consecutivos >= max_erros:
                print(f"Erro: {max_erros} falhas consecutivas ao obter dados. Verificando conexão...")
                time.sleep(300)  # Espera 5 minutos antes de tentar novamente
                continue
            
            if cache_historico is not None and not cache_historico.empty:
                # Usar dados do cache
                df = cache_historico.copy()
                df['MM5'] = df['Close'].rolling(window=5).mean()
                df['MM20'] = df['Close'].rolling(window=20).mean()
                df['MM50'] = df['Close'].rolling(window=50).mean()
                df['MM200'] = df['Close'].rolling(window=200).mean()
                df['RSI'] = df['Close'].rolling(window=14).apply(lambda x: calcular_rsi(x))
                
                ultimo = df.iloc[-1]
                preco_atual = ultimo['Close']
                
                print(f"\nPreço Atual: R$ {preco_atual:.2f}")
                print(f"RSI: {ultimo['RSI']:.2f}")
                print("\nMédias Móveis:")
                print(f"MM5: R$ {ultimo['MM5']:.2f} (Curtíssimo prazo)")
                print(f"MM20: R$ {ultimo['MM20']:.2f} (Curto prazo)")
                print(f"MM50: R$ {ultimo['MM50']:.2f} (Médio prazo)")
                print(f"MM200: R$ {ultimo['MM200']:.2f} (Longo prazo)")
                
                # Análise de Tendência
                tendencia_curta = "ALTA" if ultimo['MM5'] > ultimo['MM20'] else "BAIXA"
                tendencia_media = "ALTA" if ultimo['MM20'] > ultimo['MM50'] else "BAIXA"
                tendencia_longa = "ALTA" if ultimo['MM50'] > ultimo['MM200'] else "BAIXA"
                
                print("\nAnálise de Tendências:")
                print(f"Curto Prazo: {tendencia_curta}")
                print(f"Médio Prazo: {tendencia_media}")
                print(f"Longo Prazo: {tendencia_longa}")
                
                # Força da Tendência
                forca_tendencia = {
                    'curta': abs((ultimo['MM5'] - ultimo['MM20']) / ultimo['MM20'] * 100),
                    'media': abs((ultimo['MM20'] - ultimo['MM50']) / ultimo['MM50'] * 100),
                    'longa': abs((ultimo['MM50'] - ultimo['MM200']) / ultimo['MM200'] * 100)
                }
                
                print("\nForça das Tendências:")
                print(f"Curto Prazo: {forca_tendencia['curta']:.1f}%")
                print(f"Médio Prazo: {forca_tendencia['media']:.1f}%")
                print(f"Longo Prazo: {forca_tendencia['longa']:.1f}%")
                
                # Calcular e mostrar Ichimoku
                ichimoku = calcular_ichimoku(df)
                ultimo_ichimoku = ichimoku.iloc[-1]
                print("\nIchimoku Cloud:")
                print(f"Tenkan-sen: R$ {ultimo_ichimoku['tenkan_sen']:.2f}" if not pd.isna(ultimo_ichimoku['tenkan_sen']) else "Tenkan-sen: Sem dados suficientes")
                print(f"Kijun-sen: R$ {ultimo_ichimoku['kijun_sen']:.2f}" if not pd.isna(ultimo_ichimoku['kijun_sen']) else "Kijun-sen: Sem dados suficientes")
                print(f"Senkou Span A: R$ {ultimo_ichimoku['senkou_span_a']:.2f}" if not pd.isna(ultimo_ichimoku['senkou_span_a']) else "Senkou Span A: Sem dados suficientes")
                print(f"Senkou Span B: R$ {ultimo_ichimoku['senkou_span_b']:.2f}" if not pd.isna(ultimo_ichimoku['senkou_span_b']) else "Senkou Span B: Sem dados suficientes")
                
                # Calcular e mostrar ADX/ATR
                adx, plus_di, minus_di = calcular_adx(df)
                atr = calcular_atr(df)
                print("\nADX/ATR:")
                print(f"ADX: {adx.iloc[-1]:.2f}")
                print(f"DI+: {plus_di.iloc[-1]:.2f}")
                print(f"DI-: {minus_di.iloc[-1]:.2f}")
                print(f"ATR: R$ {atr.iloc[-1]:.2f}")
                
                # Calcular e mostrar MACD
                macd, sinal = calcular_macd(df['Close'])
                print("\nMACD:")
                print(f"MACD: {macd.iloc[-1]:.4f}")
                print(f"Sinal: {sinal.iloc[-1]:.4f}")
                
                # Calcular e mostrar Bandas de Bollinger
                banda_sup, banda_inf = calcular_bandas_bollinger(df['Close'])
                print("\nBandas de Bollinger:")
                print(f"Banda Superior: R$ {banda_sup.iloc[-1]:.2f}")
                print(f"Banda Inferior: R$ {banda_inf.iloc[-1]:.2f}")
                
                # Analisar momento atual
                mensagens, pontos_compra, pontos_venda, stop_loss, stop_gain = analisar_momento_mercado(
                    df.rename(columns={'High': 'high', 'Low': 'low', 'Close': 'close'}),
                    ultimo['RSI'],
                    ultimo['MM5'],
                    ultimo['MM20']
                )
                
                print("\nSinais Técnicos:")
                for msg in mensagens:
                    print(f"• {msg}")
                
                print(f"\nPontuação dos Sinais:")
                print(f"Compra: {pontos_compra} pontos")
                print(f"Venda: {pontos_venda} pontos")
                
                confianca = max(pontos_compra, pontos_venda) / (pontos_compra + pontos_venda) * 100 if (pontos_compra + pontos_venda) > 0 else 0
                
                print(f"\nConclusão Final:")
                if pontos_compra > pontos_venda and pontos_compra >= 3:
                    print(f"🎯 MOMENTO EXCELENTE PARA COMPRA! (Confiança: {confianca:.0f}%)")
                    print(f"Stop Loss Sugerido: R$ {stop_loss:.2f}")
                    print(f"Stop Gain Sugerido: R$ {stop_gain:.2f}")
                elif pontos_venda > pontos_compra and pontos_venda >= 3:
                    print(f"🎯 MOMENTO EXCELENTE PARA VENDA! (Confiança: {confianca:.0f}%)")
                    print(f"Stop Loss Sugerido: R$ {stop_loss:.2f}")
                    print(f"Stop Gain Sugerido: R$ {stop_gain:.2f}")
                else:
                    print("⏳ MOMENTO NEUTRO - Aguarde melhor oportunidade")
                
                print("\nVariações:")
                variacao_dia = ((preco_atual - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                print(f"Variação do dia: {variacao_dia:.2f}%")
                
                # Verificar mudanças significativas
                status = None
                detalhes = []
                
                if pontos_compra > pontos_venda and pontos_compra >= 3:
                    status = "🟢 MOMENTO EXCELENTE PARA COMPRA"
                    detalhes.extend([
                        f"Confiança: {confianca:.0f}%",
                        f"Stop Loss: R$ {stop_loss:.2f}",
                        f"Stop Gain: R$ {stop_gain:.2f}"
                    ])
                elif pontos_venda > pontos_compra and pontos_venda >= 3:
                    status = "🔴 MOMENTO EXCELENTE PARA VENDA"
                    detalhes.extend([
                        f"Confiança: {confianca:.0f}%",
                        f"Stop Loss: R$ {stop_loss:.2f}",
                        f"Stop Gain: R$ {stop_gain:.2f}"
                    ])
                
                # Notificar mudanças de status
                if status:
                    notifier.notificar_mudanca(ticker, preco_atual, status, "\n".join(detalhes))
                
                # Notificar variações bruscas (mais de 5% em um dia)
                if abs(variacao_dia) > 5:
                    notifier.enviar_mensagem(f"""
⚠️ <b>Variação Brusca Detectada - {ticker}</b>

Variação do dia: {variacao_dia:.2f}%
Preço Atual: R$ {preco_atual:.2f}

⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
""")
                
                # Aguardar intervalo definido
                for minuto in range(intervalo_minutos, 0, -1):
                    print(f"Próxima análise em: {minuto} minutos... (Última atualização: {ultima_atualizacao.strftime('%H:%M:%S')})", end='\r')
                    time.sleep(60)
            
        except Exception as e:
            print(f"Erro: {str(e)}")
            erros_consecutivos += 1
            tempo_espera = min(300 * erros_consecutivos, 3600)  # Máximo 1 hora
            print(f"Tentando novamente em {tempo_espera/60:.0f} minutos...")
            time.sleep(tempo_espera)

if __name__ == "__main__":
    print("Análise Individual de Ações")
    print("="*30)
    
    ticker = input("Digite o ticker da ação para análise (ex: PETR4): ").strip().upper()
    intervalo = input("Digite o intervalo em minutos entre análises (padrão: 30): ").strip()
    
    if not intervalo:
        intervalo = 30
    else:
        intervalo = int(intervalo)
    
    print(f"\nIniciando análise contínua para {ticker}")
    print("Pressione Ctrl+C para parar\n")
    
    try:
        realizar_analise_continua(ticker, intervalo)
    except KeyboardInterrupt:
        print("\nBot encerrado pelo usuário.") 