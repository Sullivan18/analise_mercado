import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import yfinance as yf

def calcular_rsi(precos, periodo=14):
    diferencas = np.diff(precos)
    ganhos = np.where(diferencas > 0, diferencas, 0)
    perdas = np.where(diferencas < 0, -diferencas, 0)
    media_ganhos = np.mean(ganhos[:periodo])
    media_perdas = np.mean(perdas[:periodo])
    if media_perdas == 0:
        return 100
    rs = media_ganhos / media_perdas
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calcular_ichimoku(df):
    """Calcula o Ichimoku Cloud"""
    # Tenkan-sen (Linha de Conversão)
    periodo_9_high = df['High'].rolling(window=9).max()
    periodo_9_low = df['Low'].rolling(window=9).min()
    tenkan_sen = (periodo_9_high + periodo_9_low) / 2

    # Kijun-sen (Linha Base)
    periodo_26_high = df['High'].rolling(window=26).max()
    periodo_26_low = df['Low'].rolling(window=26).min()
    kijun_sen = (periodo_26_high + periodo_26_low) / 2

    # Senkou Span A (Primeira Linha Principal)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    # Senkou Span B (Segunda Linha Principal)
    periodo_52_high = df['High'].rolling(window=52).max()
    periodo_52_low = df['Low'].rolling(window=52).min()
    senkou_span_b = ((periodo_52_high + periodo_52_low) / 2).shift(26)

    # Chikou Span (Linha de Atraso)
    chikou_span = df['Close'].shift(-26)

    return pd.DataFrame({
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    })

def calcular_adx(df, periodo=14):
    """Calcula o ADX (Average Directional Index)"""
    # True Range
    df['TR'] = np.maximum(
        np.maximum(
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift(1))
        ),
        abs(df['Low'] - df['Close'].shift(1))
    )
    
    # +DM e -DM
    df['plus_DM'] = np.where(
        (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
        np.maximum(df['High'] - df['High'].shift(1), 0),
        0
    )
    df['minus_DM'] = np.where(
        (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
        np.maximum(df['Low'].shift(1) - df['Low'], 0),
        0
    )
    
    # Médias móveis exponenciais
    df['TR_smoothed'] = df['TR'].ewm(span=periodo, adjust=False).mean()
    df['plus_DI'] = 100 * (df['plus_DM'].ewm(span=periodo, adjust=False).mean() / df['TR_smoothed'])
    df['minus_DI'] = 100 * (df['minus_DM'].ewm(span=periodo, adjust=False).mean() / df['TR_smoothed'])
    
    # ADX
    df['DX'] = 100 * abs(df['plus_DI'] - df['minus_DI']) / (df['plus_DI'] + df['minus_DI'])
    df['ADX'] = df['DX'].ewm(span=periodo, adjust=False).mean()
    
    return df['ADX'], df['plus_DI'], df['minus_DI']

def calcular_atr(df, periodo=14):
    """Calcula o ATR (Average True Range)"""
    df['TR'] = np.maximum(
        np.maximum(
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift(1))
        ),
        abs(df['Low'] - df['Close'].shift(1))
    )
    return df['TR'].rolling(window=periodo).mean()

def analisar_ichimoku(ichimoku, preco_atual):
    """Analisa os sinais do Ichimoku Cloud"""
    ultimo = ichimoku.iloc[-1]
    sinais = []
    forca = 0
    
    # Análise do Preço vs Nuvem
    if preco_atual > ultimo['senkou_span_a'] and preco_atual > ultimo['senkou_span_b']:
        sinais.append("Preço acima da nuvem (Tendência de Alta)")
        forca += 2
    elif preco_atual < ultimo['senkou_span_a'] and preco_atual < ultimo['senkou_span_b']:
        sinais.append("Preço abaixo da nuvem (Tendência de Baixa)")
        forca -= 2
    
    # Análise do Tenkan-sen vs Kijun-sen (Cruzamento)
    if ultimo['tenkan_sen'] > ultimo['kijun_sen']:
        sinais.append("Tenkan-sen acima do Kijun-sen (Sinal de Compra)")
        forca += 1
    elif ultimo['tenkan_sen'] < ultimo['kijun_sen']:
        sinais.append("Tenkan-sen abaixo do Kijun-sen (Sinal de Venda)")
        forca -= 1
    
    # Análise do Chikou Span
    if not pd.isna(ultimo['chikou_span']):
        if ultimo['chikou_span'] > preco_atual:
            sinais.append("Chikou Span acima do preço (Confirmação de Alta)")
            forca += 1
        else:
            sinais.append("Chikou Span abaixo do preço (Confirmação de Baixa)")
            forca -= 1
    
    return sinais, forca

def analisar_adx_atr(adx, plus_di, minus_di, atr, preco_atual):
    """Analisa os sinais do ADX e ATR"""
    sinais = []
    forca = 0
    
    # Análise do ADX
    ultimo_adx = adx.iloc[-1]
    if ultimo_adx > 25:
        sinais.append(f"ADX forte ({ultimo_adx:.1f}) - Tendência significativa")
        
        # Análise da direção da tendência usando DI
        if plus_di.iloc[-1] > minus_di.iloc[-1]:
            sinais.append("DI+ > DI- (Tendência de Alta)")
            forca += 2
        else:
            sinais.append("DI- > DI+ (Tendência de Baixa)")
            forca -= 2
    else:
        sinais.append(f"ADX fraco ({ultimo_adx:.1f}) - Mercado sem tendência definida")
    
    # Análise do ATR
    atr_atual = atr.iloc[-1]
    atr_medio = atr.mean()
    
    if atr_atual > atr_medio * 1.5:
        sinais.append("Alta volatilidade - Cuidado com operações")
    elif atr_atual < atr_medio * 0.5:
        sinais.append("Baixa volatilidade - Possível movimento brusco próximo")
    
    # Sugestão de Stop Loss baseado no ATR
    stop_loss_atr = preco_atual - (atr_atual * 2)
    stop_gain_atr = preco_atual + (atr_atual * 3)
    
    return sinais, forca, stop_loss_atr, stop_gain_atr

def calcular_macd(precos, rapida=12, lenta=26, sinal=9):
    """Calcula MACD (Moving Average Convergence Divergence)"""
    exp1 = pd.Series(precos).ewm(span=rapida, adjust=False).mean()
    exp2 = pd.Series(precos).ewm(span=lenta, adjust=False).mean()
    macd = exp1 - exp2
    sinal = macd.ewm(span=sinal, adjust=False).mean()
    return macd, sinal

def calcular_bandas_bollinger(precos, periodo=20):
    """Calcula Bandas de Bollinger"""
    media = pd.Series(precos).rolling(window=periodo).mean()
    desvio = pd.Series(precos).rolling(window=periodo).std()
    banda_superior = media + (desvio * 2)
    banda_inferior = media - (desvio * 2)
    return banda_superior, banda_inferior

def calcular_suporte_resistencia(df, periodos=20):
    """Identifica níveis de suporte e resistência"""
    maximos = df['high'].rolling(window=periodos, center=True).max()
    minimos = df['low'].rolling(window=periodos, center=True).min()
    return minimos.iloc[-1], maximos.iloc[-1]

def calcular_tendencia_longo_prazo(df):
    """Analisa tendência usando médias móveis adaptadas para 3 meses"""
    # Ajustando para períodos menores considerando 3 meses de dados
    df['MM21'] = df['close'].rolling(window=21).mean()  # aproximadamente 1 mês
    df['MM63'] = df['close'].rolling(window=63).mean()  # aproximadamente 3 meses
    
    ultimo = df.iloc[-1]
    if pd.isna(ultimo['MM21']) or pd.isna(ultimo['MM63']):
        return "INDEFINIDA", 0
    
    tendencia = "ALTA" if ultimo['MM21'] > ultimo['MM63'] else "BAIXA"
    forca = abs((ultimo['MM21'] - ultimo['MM63']) / ultimo['MM63'] * 100)
    
    return tendencia, forca

def analisar_momento_mercado(df, rsi_atual, mm5, mm20):
    """Analisa se é um bom momento para compra ou venda"""
    mensagens = []
    pontuacao_compra = 0
    pontuacao_venda = 0
    
    # 1. Análise do RSI
    if rsi_atual < 30:
        mensagens.append("✅ RSI indica forte sobrevendido (< 30)")
        pontuacao_compra += 2
    elif rsi_atual > 70:
        mensagens.append("🔴 RSI indica forte sobrecomprado (> 70)")
        pontuacao_venda += 2
    
    # 2. Análise de Médias Móveis
    tendencia = ((mm5 - mm20) / mm20) * 100
    if mm5 > mm20:
        if tendencia > 5:
            mensagens.append("⚠️ Tendência de alta forte")
            pontuacao_compra += 2
        else:
            mensagens.append("✅ Tendência de alta iniciando")
            pontuacao_compra += 1
    else:
        if abs(tendencia) > 5:
            mensagens.append("✅ Correção forte no preço")
            pontuacao_compra += 1
        else:
            mensagens.append("🔴 Tendência de baixa iniciando")
            pontuacao_venda += 1
    
    # 3. Análise MACD
    macd, sinal = calcular_macd(df['close'])
    if macd.iloc[-1] > sinal.iloc[-1] and macd.iloc[-2] <= sinal.iloc[-2]:
        mensagens.append("✅ MACD cruzou para cima (sinal de compra)")
        pontuacao_compra += 1
    elif macd.iloc[-1] < sinal.iloc[-1] and macd.iloc[-2] >= sinal.iloc[-2]:
        mensagens.append("🔴 MACD cruzou para baixo (sinal de venda)")
        pontuacao_venda += 1
    
    # 4. Análise Bandas de Bollinger
    banda_sup, banda_inf = calcular_bandas_bollinger(df['close'])
    preco_atual = df['close'].iloc[-1]
    if preco_atual <= banda_inf.iloc[-1]:
        mensagens.append("✅ Preço tocou Banda Inferior de Bollinger")
        pontuacao_compra += 1
    elif preco_atual >= banda_sup.iloc[-1]:
        mensagens.append("🔴 Preço tocou Banda Superior de Bollinger")
        pontuacao_venda += 1
    
    # 5. Análise Suporte/Resistência
    suporte, resistencia = calcular_suporte_resistencia(df)
    if preco_atual <= suporte * 1.02:  # 2% acima do suporte
        mensagens.append("✅ Preço próximo ao suporte")
        pontuacao_compra += 1
    elif preco_atual >= resistencia * 0.98:  # 2% abaixo da resistência
        mensagens.append("🔴 Preço próximo à resistência")
        pontuacao_venda += 1
    
    # 6. Análise de Tendência de Longo Prazo
    tendencia_longo_prazo, forca_tendencia = calcular_tendencia_longo_prazo(df)
    if tendencia_longo_prazo != "INDEFINIDA":
        if tendencia_longo_prazo == "ALTA":
            mensagens.append(f"📈 Tendência de ALTA no período (Força: {forca_tendencia:.1f}%)")
            pontuacao_compra += 1
        else:
            mensagens.append(f"📉 Tendência de BAIXA no período (Força: {forca_tendencia:.1f}%)")
            pontuacao_venda += 1
    
    # 7. Análise Ichimoku Cloud
    ichimoku = calcular_ichimoku(df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low'}))
    sinais_ichimoku, forca_ichimoku = analisar_ichimoku(ichimoku, preco_atual)
    for sinal in sinais_ichimoku:
        mensagens.append(f"☁️ {sinal}")
    if forca_ichimoku > 0:
        pontuacao_compra += forca_ichimoku
    else:
        pontuacao_venda -= forca_ichimoku
    
    # 8. Análise ADX e ATR
    adx, plus_di, minus_di = calcular_adx(df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low'}))
    atr = calcular_atr(df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low'}))
    sinais_adx_atr, forca_adx, stop_loss_atr, stop_gain_atr = analisar_adx_atr(adx, plus_di, minus_di, atr, preco_atual)
    
    for sinal in sinais_adx_atr:
        mensagens.append(f"📊 {sinal}")
    if forca_adx > 0:
        pontuacao_compra += forca_adx
    else:
        pontuacao_venda -= forca_adx
    
    # Verificar qual sinal é mais forte
    if pontuacao_compra > pontuacao_venda:
        tipo_operacao = "COMPRA"
    else:
        tipo_operacao = "VENDA"
    
    # Ajustar stops com base no tipo de operação
    if tipo_operacao == "COMPRA":
        stop_loss_final = stop_loss_atr
        stop_gain_final = stop_gain_atr
    else:
        stop_loss_final = stop_gain_atr  # Invertido para venda
        stop_gain_final = stop_loss_atr   # Invertido para venda
    
    return mensagens, pontuacao_compra, pontuacao_venda, stop_loss_final, stop_gain_final

def calcular_precos_alvos(df, preco_atual):
    """Calcula preços-alvo para compra e venda usando múltiplos indicadores"""
    
    # Cálculo usando Bandas de Bollinger (20 períodos)
    banda_sup, banda_inf = calcular_bandas_bollinger(df['close'], periodo=20)
    
    # Cálculo usando médias móveis mais curtas para 3 meses de dados
    mm5 = df['close'].rolling(window=5).mean().iloc[-1]
    mm20 = df['close'].rolling(window=20).mean().iloc[-1]
    
    # Cálculo de suporte e resistência com período menor
    suporte, resistencia = calcular_suporte_resistencia(df, periodos=10)
    
    # Verificar se temos valores válidos
    valores_validos = []
    
    if not pd.isna(banda_inf.iloc[-1]):
        valores_validos.append(banda_inf.iloc[-1])
    if not pd.isna(suporte):
        valores_validos.append(suporte)
    if not pd.isna(mm20):
        valores_validos.append(mm20)
        
    # Calcular preço de compra (média dos indicadores válidos)
    if valores_validos:
        preco_compra = sum(valores_validos) / len(valores_validos)
    else:
        preco_compra = preco_atual * 0.95  # 5% abaixo do preço atual
    
    # Repetir processo para preço de venda
    valores_validos = []
    
    if not pd.isna(banda_sup.iloc[-1]):
        valores_validos.append(banda_sup.iloc[-1])
    if not pd.isna(resistencia):
        valores_validos.append(resistencia)
    if not pd.isna(mm5):
        valores_validos.append(mm5)
        
    if valores_validos:
        preco_venda = sum(valores_validos) / len(valores_validos)
    else:
        preco_venda = preco_atual * 1.05  # 5% acima do preço atual
    
    # Calcular zonas de preço com margens de segurança
    zona_compra_forte = preco_compra * 0.98
    zona_compra_moderada = preco_compra * 1.02
    zona_venda_forte = preco_venda * 1.02
    zona_venda_moderada = preco_venda * 0.98
    
    # Garantir que as zonas façam sentido em relação ao preço atual
    if zona_compra_forte > preco_atual:
        zona_compra_forte = preco_atual * 0.95
        zona_compra_moderada = preco_atual * 0.98
    
    if zona_venda_forte < preco_atual:
        zona_venda_forte = preco_atual * 1.05
        zona_venda_moderada = preco_atual * 1.02
    
    return {
        'compra_forte': zona_compra_forte,
        'compra_moderada': zona_compra_moderada,
        'venda_forte': zona_venda_forte,
        'venda_moderada': zona_venda_moderada
    }

class TelegramNotifier:
    def __init__(self):
        self.bot_token = '8189193063:AAGxS7xOo3XHE5si3Xwr9f4xojiAlgtZ7Po'
        self.chat_id = '861385777'
        self.ultimo_status = None
        
    def enviar_mensagem(self, mensagem):
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": mensagem,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data)
            
            if response.status_code != 200:
                print(f"Erro ao enviar mensagem: {response.text}")
                return False
                
            return response
        except Exception as e:
            print(f"Erro ao enviar notificação: {str(e)}")
            return False
    
    def notificar_mudanca(self, ticker, preco_atual, status, detalhes):
        if status != self.ultimo_status:
            mensagem = f"""
🔔 <b>Alerta de Mudança - {ticker}</b>

💰 Preço Atual: R$ {preco_atual:.2f}
📊 Novo Status: {status}

{detalhes}

⏰ Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
"""
            self.enviar_mensagem(mensagem)
            self.ultimo_status = status
    
    def notificar_preco_alvo(self, ticker, preco_atual, zona, preco_alvo):
        mensagem = f"""
💲 <b>Alerta de Preço - {ticker}</b>

Preço atingiu zona de {zona}!
Preço Atual: R$ {preco_atual:.2f}
Preço Alvo: R$ {preco_alvo:.2f}

⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
"""
        self.enviar_mensagem(mensagem)

def realizar_analise_continua(ticker, intervalo_minutos=30):
    notifier = TelegramNotifier()
    
    # Ajustar ticker para formato do Yahoo Finance (.SA para ações brasileiras)
    yahoo_ticker = f"{ticker}.SA" if ticker.endswith('3') or ticker.endswith('4') else ticker
    
    mensagem_inicial = f"""
🤖 <b>Bot de Análise Iniciado</b>

📈 Ativo: {ticker}
⏰ Hora de Início: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
🔄 Intervalo de Análise: {intervalo_minutos} minutos

Status: Monitorando...
"""
    notifier.enviar_mensagem(mensagem_inicial)
    
    while True:
        try:
            # Mostrar hora atual
            hora_atual = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n{'='*50}")
            print(f"Análise Técnica - {ticker}")
            print(f"Data/Hora: {hora_atual}")
            print('='*50)
            
            # Obter dados do Yahoo Finance
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
                
                print(f"\nPreço Atual: R$ {preco_atual:.2f}")
                print(f"RSI: {ultimo['RSI']:.2f}")
                print(f"MM5: R$ {ultimo['MM5']:.2f}")
                print(f"MM20: R$ {ultimo['MM20']:.2f}")
                
                # Calcular e mostrar Ichimoku
                ichimoku = calcular_ichimoku(df)
                ultimo_ichimoku = ichimoku.iloc[-1]
                print("\nIchimoku Cloud:")
                print(f"Tenkan-sen: R$ {ultimo_ichimoku['tenkan_sen']:.2f}")
                print(f"Kijun-sen: R$ {ultimo_ichimoku['kijun_sen']:.2f}")
                print(f"Senkou Span A: R$ {ultimo_ichimoku['senkou_span_a']:.2f}")
                print(f"Senkou Span B: R$ {ultimo_ichimoku['senkou_span_b']:.2f}")
                
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
                    print(f"Gain Sugerido: R$ {stop_gain:.2f}")
                elif pontos_venda > pontos_compra and pontos_venda >= 3:
                    print(f"🎯 MOMENTO EXCELENTE PARA VENDA! (Confiança: {confianca:.0f}%)")
                    print(f"Stop Loss Sugerido: R$ {stop_loss:.2f}")
                    print(f"Gain Sugerido: R$ {stop_gain:.2f}")
                else:
                    print("⏳ MOMENTO NEUTRO - Aguarde melhor oportunidade")
                
                print("\nVariações:")
                variacao_dia = ((preco_atual - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                print(f"Variação do dia: {variacao_dia:.2f}%")
                
                # Calcular preços-alvo
                precos_alvos = calcular_precos_alvos(df.rename(columns={'High': 'high', 'Low': 'low', 'Close': 'close'}), preco_atual)
                
                # Adicionar sugestões de stop loss e gain baseados nas zonas
                if pontos_compra > pontos_venda:
                    stop_loss_final = min(precos_alvos['compra_forte'] * 0.95, stop_loss)
                    stop_gain_final = max(precos_alvos['venda_moderada'], stop_gain)
                    print("\nSugestão para Compra:")
                    print(f"Stop Loss: R$ {stop_loss_final:.2f}")
                    print(f"Stop Gain: R$ {stop_gain_final:.2f}")
                elif pontos_venda > pontos_compra:
                    stop_gain_final = max(precos_alvos['venda_forte'] * 1.05, stop_gain)
                    stop_loss_final = min(precos_alvos['compra_moderada'], stop_loss)
                    print("\nSugestão para Venda:")
                    print(f"Stop Gain: R$ {stop_gain_final:.2f}")
                    print(f"Stop Loss: R$ {stop_loss_final:.2f}")

                # Verificar mudanças significativas
                status = None
                detalhes = []
                
                if pontos_compra > pontos_venda and pontos_compra >= 3:
                    status = "🟢 MOMENTO EXCELENTE PARA COMPRA"
                    detalhes.extend([
                        f"Confiança: {confianca:.0f}%",
                        f"Stop Loss: R$ {stop_loss_final:.2f}",
                        f"Stop Gain: R$ {stop_gain_final:.2f}"
                    ])
                elif pontos_venda > pontos_compra and pontos_venda >= 3:
                    status = "🔴 MOMENTO EXCELENTE PARA VENDA"
                    detalhes.extend([
                        f"Confiança: {confianca:.0f}%",
                        f"Stop Gain: R$ {stop_gain_final:.2f}",
                        f"Stop Loss: R$ {stop_loss_final:.2f}"
                    ])
                
                # Notificar mudanças de status
                if status:
                    notifier.notificar_mudanca(ticker, preco_atual, status, "\n".join(detalhes))
                
                # Verificar zonas de preço
                if preco_atual <= precos_alvos['compra_forte']:
                    notifier.notificar_preco_alvo(ticker, preco_atual, "COMPRA FORTE", precos_alvos['compra_forte'])
                elif preco_atual >= precos_alvos['venda_forte']:
                    notifier.notificar_preco_alvo(ticker, preco_atual, "VENDA FORTE", precos_alvos['venda_forte'])
                
                # Notificar variações bruscas (mais de 5% em um dia)
                if abs(variacao_dia) > 5:
                    notifier.enviar_mensagem(f"""
⚠️ <b>Variação Brusca Detectada - {ticker}</b>

Variação do dia: {variacao_dia:.2f}%
Preço Atual: R$ {preco_atual:.2f}

⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
""")
            else:
                print(f"Erro: Nenhum dado encontrado para {ticker}")
            
            # Aguardar intervalo definido
            for minuto in range(intervalo_minutos, 0, -1):
                print(f"Próxima análise em: {minuto} minutos...", end='\r')
                time.sleep(60)
            
        except Exception as e:
            print(f"Erro: {str(e)}")
            print("Tentando novamente em 5 minutos...")
            time.sleep(5 * 60)

def realizar_backtesting(ticker, periodo="1y"):
    """
    Realiza backtesting da estratégia nos últimos X meses
    periodo: 1mo, 3mo, 6mo, 1y, 2y, etc.
    """
    print(f"\n{'='*50}")
    print(f"BACKTESTING - {ticker}")
    print(f"Período: {periodo}")
    print('='*50)
    
    # Ajustar ticker para formato do Yahoo Finance
    yahoo_ticker = f"{ticker}.SA" if ticker.endswith('3') or ticker.endswith('4') else ticker
    
    # Obter dados históricos
    ativo = yf.Ticker(yahoo_ticker)
    historico = ativo.history(period=periodo, interval="1d")
    
    if historico.empty:
        print("Erro: Sem dados suficientes para backtesting")
        return
    
    # Inicializar variáveis
    operacoes = []
    posicao_aberta = False
    preco_entrada = 0
    tipo_operacao = None
    stop_loss = 0
    gain = 0
    data_entrada = None
    
    # Para cada dia, exceto o primeiro (precisamos de dados anteriores)
    for i in range(20, len(historico)-1):
        df_analise = historico[:i+1].copy()
        
        # Preparar dados para análise
        df_analise['MM5'] = df_analise['Close'].rolling(window=5).mean()
        df_analise['MM20'] = df_analise['Close'].rolling(window=20).mean()
        df_analise['RSI'] = df_analise['Close'].rolling(window=14).apply(lambda x: calcular_rsi(x))
        
        ultimo = df_analise.iloc[-1]
        preco_atual = ultimo['Close']
        
        # Analisar momento
        mensagens, pontos_compra, pontos_venda, stop_loss_atr, stop_gain_atr = analisar_momento_mercado(
            df_analise.rename(columns={'High': 'high', 'Low': 'low', 'Close': 'close'}),
            ultimo['RSI'],
            ultimo['MM5'],
            ultimo['MM20']
        )
        
        # Se não tem posição aberta, verificar sinais
        if not posicao_aberta:
            if pontos_compra > pontos_venda and pontos_compra >= 3:
                posicao_aberta = True
                tipo_operacao = "COMPRA"
                preco_entrada = preco_atual
                stop_loss = stop_loss_atr
                gain = stop_gain_atr
                data_entrada = df_analise.index[-1]
            elif pontos_venda > pontos_compra and pontos_venda >= 3:
                posicao_aberta = True
                tipo_operacao = "VENDA"
                preco_entrada = preco_atual
                stop_loss = stop_loss_atr
                gain = stop_gain_atr
                data_entrada = df_analise.index[-1]
        
        # Se tem posição aberta, verificar saída
        elif posicao_aberta:
            preco_fechamento = historico.iloc[i+1]['Close']  # Próximo dia
            resultado = 0
            
            if tipo_operacao == "COMPRA":
                if preco_fechamento <= stop_loss:  # Stop Loss
                    resultado = ((stop_loss / preco_entrada) - 1) * 100
                    operacoes.append({
                        'tipo': tipo_operacao,
                        'entrada': preco_entrada,
                        'saida': stop_loss,
                        'data_entrada': data_entrada,
                        'data_saida': historico.index[i+1],
                        'resultado': resultado,
                        'motivo': 'Stop Loss'
                    })
                    posicao_aberta = False
                elif preco_fechamento >= gain:  # Take Profit
                    resultado = ((gain / preco_entrada) - 1) * 100
                    operacoes.append({
                        'tipo': tipo_operacao,
                        'entrada': preco_entrada,
                        'saida': gain,
                        'data_entrada': data_entrada,
                        'data_saida': historico.index[i+1],
                        'resultado': resultado,
                        'motivo': 'Take Profit'
                    })
                    posicao_aberta = False
            else:  # VENDA
                if preco_fechamento >= stop_loss:  # Stop Loss
                    resultado = ((preco_entrada / stop_loss) - 1) * 100
                    operacoes.append({
                        'tipo': tipo_operacao,
                        'entrada': preco_entrada,
                        'saida': stop_loss,
                        'data_entrada': data_entrada,
                        'data_saida': historico.index[i+1],
                        'resultado': resultado,
                        'motivo': 'Stop Loss'
                    })
                    posicao_aberta = False
                elif preco_fechamento <= gain:  # Take Profit
                    resultado = ((preco_entrada / gain) - 1) * 100
                    operacoes.append({
                        'tipo': tipo_operacao,
                        'entrada': preco_entrada,
                        'saida': gain,
                        'data_entrada': data_entrada,
                        'data_saida': historico.index[i+1],
                        'resultado': resultado,
                        'motivo': 'Take Profit'
                    })
                    posicao_aberta = False
    
    # Análise dos resultados
    if not operacoes:
        print("\nNenhuma operação realizada no período")
        return
    
    total_ops = len(operacoes)
    ops_gain = len([op for op in operacoes if op['resultado'] > 0])
    ops_loss = total_ops - ops_gain
    taxa_acerto = (ops_gain / total_ops) * 100
    
    resultado_total = sum(op['resultado'] for op in operacoes)
    resultado_medio = resultado_total / total_ops
    
    print("\nRESULTADOS DO BACKTESTING:")
    print(f"Total de operações: {total_ops}")
    print(f"Operações com lucro: {ops_gain}")
    print(f"Operações com prejuízo: {ops_loss}")
    print(f"Taxa de acerto: {taxa_acerto:.1f}%")
    print(f"Resultado médio por operação: {resultado_medio:.2f}%")
    print(f"Resultado acumulado: {resultado_total:.2f}%")
    
    print("\nÚLTIMAS 5 OPERAÇÕES:")
    for op in operacoes[-5:]:
        print(f"\nTipo: {op['tipo']}")
        print(f"Data Entrada: {op['data_entrada'].strftime('%d/%m/%Y')}")
        print(f"Data Saída: {op['data_saida'].strftime('%d/%m/%Y')}")
        print(f"Preço Entrada: R$ {op['entrada']:.2f}")
        print(f"Preço Saída: R$ {op['saida']:.2f}")
        print(f"Resultado: {op['resultado']:.2f}%")
        print(f"Motivo: {op['motivo']}")
    
    return operacoes

def testar_telegram():
    """Função para testar a conexão com o Telegram"""
    notifier = TelegramNotifier()
    try:
        mensagem_teste = f"""
🧪 <b>Teste de Conexão</b>

Bot está funcionando!
⏰ Hora do teste: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
"""
        response = notifier.enviar_mensagem(mensagem_teste)
        print("Mensagem de teste enviada!")
        print("Se você não recebeu a mensagem, verifique:")
        print("1. Se você iniciou uma conversa com o bot (@seu_bot)")
        print("2. Se o token do bot está correto")
        print("3. Se o chat_id está correto")
        return True
    except Exception as e:
        print(f"Erro ao enviar mensagem de teste: {str(e)}")
        return False

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
    
    # Perguntar qual ação analisar em tempo real
    if melhores_oportunidades:
        print("\nQual ação você deseja monitorar em tempo real?")
        ticker = input("Digite o ticker (ou pressione Enter para usar a melhor opção): ").strip().upper()
        
        if not ticker:
            ticker = melhores_oportunidades[0]['ticker']
        
        print(f"\nIniciando análise contínua para {ticker}")
        print("Pressione Ctrl+C para parar\n")
        
        try:
            realizar_analise_continua(ticker)
        except KeyboardInterrupt:
            print("\nBot encerrado pelo usuário.")
