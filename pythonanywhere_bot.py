import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time
import time as time_lib
import os
from main import calcular_rsi, calcular_ichimoku, calcular_adx, calcular_atr
from main import analisar_ichimoku, analisar_adx_atr, calcular_macd, calcular_bandas_bollinger
from main import calcular_suporte_resistencia, calcular_tendencia_longo_prazo, analisar_momento_mercado
from main import TelegramNotifier
from zoneinfo import ZoneInfo

def esta_mercado_aberto():
    """
    Verifica se a B3 está aberta no momento
    Retorna: (bool) True se estiver em horário de funcionamento
    """
    # Obtém hora atual em São Paulo
    hora_sp = datetime.now(ZoneInfo("America/Sao_Paulo"))
    
    # Verifica se é dia útil (0 = Segunda, 6 = Domingo)
    if hora_sp.weekday() >= 5:
        return False
    
    # Horário de funcionamento: 10:00 às 17:55
    hora_atual = hora_sp.time()
    abertura = time(10, 0)
    fechamento = time(17, 55)
    
    return abertura <= hora_atual <= fechamento

def realizar_analise_continua(ticker, intervalo_minutos=30):
    """
    Realiza análise contínua de uma ação específica
    """
    # Configurar logging para arquivo
    log_file = f"bot_log_{datetime.now().strftime('%Y%m%d')}.txt"
    
    def log(mensagem):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {mensagem}\n")
        print(f"[{timestamp}] {mensagem}")
    
    notifier = TelegramNotifier()
    
    # Ajustar ticker para formato do Yahoo Finance
    yahoo_ticker = f"{ticker}.SA" if ticker.endswith('3') or ticker.endswith('4') else ticker
    
    mensagem_inicial = f"""
🤖 <b>Bot de Análise Iniciado no PythonAnywhere</b>

📈 Ativo: {ticker}
⏰ Hora de Início: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
🔄 Intervalo de Análise: {intervalo_minutos} minutos
⚠️ Monitoramento apenas em horário de mercado (10:00-17:55)

Status: Monitorando...
"""
    notifier.enviar_mensagem(mensagem_inicial)
    log("Bot iniciado com sucesso")
    
    # Cache para armazenar dados históricos
    cache_historico = None
    ultima_atualizacao = None
    erros_consecutivos = 0
    max_erros = 3
    
    while True:
        try:
            # Verificar se mercado está aberto
            if not esta_mercado_aberto():
                hora_sp = datetime.now(ZoneInfo("America/Sao_Paulo"))
                if hora_sp.weekday() >= 5:
                    log("Mercado fechado (fim de semana). Aguardando próximo dia útil...")
                    # Aguarda 8 horas antes de verificar novamente
                    time_lib.sleep(8 * 3600)
                else:
                    log("Fora do horário de mercado. Aguardando próxima abertura...")
                    # Aguarda 15 minutos antes de verificar novamente
                    time_lib.sleep(15 * 60)
                continue
            
            # Mostrar hora atual
            hora_atual = datetime.now()
            log(f"Iniciando análise para {ticker}")
            
            # Verificar se precisa atualizar o cache (a cada 15 minutos)
            precisa_atualizar = (
                cache_historico is None or
                ultima_atualizacao is None or
                (hora_atual - ultima_atualizacao).total_seconds() > 900  # 15 minutos
            )
            
            if precisa_atualizar:
                log("Atualizando dados do Yahoo Finance...")
                try:
                    # Obter dados do Yahoo Finance com retry
                    for tentativa in range(3):
                        try:
                            ativo = yf.Ticker(yahoo_ticker)
                            novo_historico = ativo.history(period="1y", interval="1d")
                            if not novo_historico.empty:
                                break
                        except Exception as e:
                            log(f"Tentativa {tentativa + 1} falhou: {str(e)}")
                            time.sleep(5)
                    
                    if not novo_historico.empty:
                        cache_historico = novo_historico
                        ultima_atualizacao = hora_atual
                        erros_consecutivos = 0
                        log("Dados atualizados com sucesso")
                    else:
                        raise Exception("Não foi possível obter dados após 3 tentativas")
                except Exception as e:
                    erros_consecutivos += 1
                    if cache_historico is None:
                        raise Exception(f"Erro ao obter dados: {str(e)}")
                    log("Aviso: Usando dados em cache devido a erro na atualização")
            
            if erros_consecutivos >= max_erros:
                log(f"Erro: {max_erros} falhas consecutivas ao obter dados. Verificando conexão...")
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
                
                # Análise de Tendência
                tendencia_curta = "ALTA" if ultimo['MM5'] > ultimo['MM20'] else "BAIXA"
                tendencia_media = "ALTA" if ultimo['MM20'] > ultimo['MM50'] else "BAIXA"
                tendencia_longa = "ALTA" if ultimo['MM50'] > ultimo['MM200'] else "BAIXA"
                
                # Força da Tendência
                forca_tendencia = {
                    'curta': abs((ultimo['MM5'] - ultimo['MM20']) / ultimo['MM20'] * 100),
                    'media': abs((ultimo['MM20'] - ultimo['MM50']) / ultimo['MM50'] * 100),
                    'longa': abs((ultimo['MM50'] - ultimo['MM200']) / ultimo['MM200'] * 100)
                }
                
                # Analisar momento atual
                mensagens, pontos_compra, pontos_venda, stop_loss, stop_gain = analisar_momento_mercado(
                    df.rename(columns={'High': 'high', 'Low': 'low', 'Close': 'close'}),
                    ultimo['RSI'],
                    ultimo['MM5'],
                    ultimo['MM20']
                )
                
                confianca = max(pontos_compra, pontos_venda) / (pontos_compra + pontos_venda) * 100 if (pontos_compra + pontos_venda) > 0 else 0
                
                # Verificar mudanças significativas
                status = None
                detalhes = []
                
                if pontos_compra > pontos_venda and pontos_compra >= 3:
                    status = "🟢 MOMENTO EXCELENTE PARA COMPRA"
                    detalhes.extend([
                        f"Confiança: {confianca:.0f}%",
                        f"Stop Loss: R$ {stop_loss:.2f}",
                        f"Stop Gain: R$ {stop_gain:.2f}",
                        f"Tendência Curta: {tendencia_curta}",
                        f"Tendência Média: {tendencia_media}",
                        f"Tendência Longa: {tendencia_longa}"
                    ])
                elif pontos_venda > pontos_compra and pontos_venda >= 3:
                    status = "🔴 MOMENTO EXCELENTE PARA VENDA"
                    detalhes.extend([
                        f"Confiança: {confianca:.0f}%",
                        f"Stop Loss: R$ {stop_loss:.2f}",
                        f"Stop Gain: R$ {stop_gain:.2f}",
                        f"Tendência Curta: {tendencia_curta}",
                        f"Tendência Média: {tendencia_media}",
                        f"Tendência Longa: {tendencia_longa}"
                    ])
                
                # Notificar mudanças de status
                if status:
                    notifier.notificar_mudanca(ticker, preco_atual, status, "\n".join(detalhes))
                    log(f"Notificação enviada: {status}")
                
                # Calcular variação do dia
                variacao_dia = ((preco_atual - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                
                # Notificar variações bruscas (mais de 5% em um dia)
                if abs(variacao_dia) > 5:
                    mensagem_variacao = f"""
⚠️ <b>Variação Brusca Detectada - {ticker}</b>

Variação do dia: {variacao_dia:.2f}%
Preço Atual: R$ {preco_atual:.2f}

⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
"""
                    notifier.enviar_mensagem(mensagem_variacao)
                    log(f"Variação brusca detectada: {variacao_dia:.2f}%")
                
                # Aguardar intervalo
                log(f"Análise concluída. Próxima análise em {intervalo_minutos} minutos")
                time.sleep(intervalo_minutos * 60)
            
        except Exception as e:
            log(f"Erro: {str(e)}")
            erros_consecutivos += 1
            tempo_espera = min(300 * erros_consecutivos, 3600)  # Máximo 1 hora
            log(f"Tentando novamente em {tempo_espera/60:.0f} minutos...")
            time.sleep(tempo_espera)

if __name__ == "__main__":
    # Configurações fixas do bot
    TICKER = 'CVCB3'  # Ticker fixo, sem depender de variável de ambiente
    INTERVALO = 30
    
    print(f"Iniciando bot para {TICKER} com intervalo de {INTERVALO} minutos")
    realizar_analise_continua(TICKER, INTERVALO) 