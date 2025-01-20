# Crie um arquivo chamado start_bot.py
import os
import sys
from datetime import datetime

# Configurar logs
log_dir = os.path.expanduser('~/logs')
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f'bot_log_{datetime.now().strftime("%Y%m%d")}.txt')
error_file = os.path.join(log_dir, f'bot_error_{datetime.now().strftime("%Y%m%d")}.txt')

sys.stdout = open(log_file, 'a')
sys.stderr = open(error_file, 'a')

print(f"\nBot iniciado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Importar e executar o bot
from main import realizar_analise_continua

ticker = "CVCB3"
realizar_analise_continua(ticker)