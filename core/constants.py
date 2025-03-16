"""
Модуль констант для Leon Trading Bot.

Этот модуль содержит константы, используемые в различных частях системы.
Текстовые константы перенесены в модуль локализации.
"""

# Режимы работы
TRADING_MODES = {
    "DRY": "dry",           # Режим симуляции
    "BACKTEST": "backtest", # Режим бэктестинга
    "REAL": "real"          # Режим реальной торговли
}

# Типы событий
EVENT_TYPES = {
    "SYSTEM_STARTED": "system_started",           # Система запущена
    "SYSTEM_STOPPED": "system_stopped",           # Система остановлена
    "MODE_CHANGED": "mode_changed",               # Режим работы изменен
    "TRADE_EXECUTED": "trade_executed",           # Сделка выполнена
    "POSITION_OPENED": "position_opened",         # Позиция открыта
    "POSITION_CLOSED": "position_closed",         # Позиция закрыта
    "POSITION_UPDATED": "position_updated",       # Позиция обновлена
    "PRICE_UPDATED": "price_updated",             # Цена обновлена
    "BALANCE_UPDATED": "balance_updated",         # Баланс обновлен
    "ERROR_OCCURRED": "error_occurred",           # Произошла ошибка
    "MODEL_LOADED": "model_loaded",               # Модель загружена
    "PREDICTION_RECEIVED": "prediction_received", # Получено предсказание
    "MODEL_EVALUATED": "model_evaluated",         # Модель оценена
    "TRADE_COMPLETED": "trade_completed"          # Сделка завершена
}

# Ключи локализации
LOCALIZATION_KEYS = {
    "WELCOME_PHRASES": "welcome_phrases",
    "ERROR_PHRASES": "error_phrases",
    "SUCCESS_PHRASES": "success_phrases",
    "ML_PHRASES": "ml_phrases",
    "NO_ANSWER": "no_answer",
    "BUDGET_KILLER": "budget_killer",
    "PRESS_CTRL_C": "press_ctrl_c",
    "MODE_NAMES": "mode.names",
    "MODE_WELCOME_MESSAGES": "mode.welcome_messages",
    "MODE_WARNING_MESSAGES": "mode.warning_messages",
    "MODE_EXIT_MESSAGES": "mode.exit_messages"
}

# Коды ошибок
ERROR_CODES = {
    "INITIALIZATION_ERROR": 1001,  # Ошибка инициализации
    "OPERATION_ERROR": 1002,       # Ошибка операции
    "INVALID_MODE_ERROR": 1003,    # Недопустимый режим работы
    "COMMAND_ERROR": 1004,         # Ошибка выполнения команды
    "MODEL_LOAD_ERROR": 1005,      # Ошибка загрузки модели
    "PREDICTION_ERROR": 1006,      # Ошибка предсказания
    "EVALUATION_ERROR": 1007,      # Ошибка оценки
    "API_ERROR": 1008,             # Ошибка API
    "NETWORK_ERROR": 1009,         # Ошибка сети
    "DATA_ERROR": 1010,            # Ошибка данных
    "STRATEGY_ERROR": 1011,        # Ошибка стратегии
    "TRADER_ERROR": 1012,          # Ошибка трейдера
    "VISUALIZATION_ERROR": 1013,   # Ошибка визуализации
    "NOTIFICATION_ERROR": 1014     # Ошибка уведомления
}

# Статусы системы
SYSTEM_STATUSES = {
    "INITIALIZING": "initializing", # Инициализация
    "RUNNING": "running",           # Работает
    "STOPPING": "stopping",         # Останавливается
    "STOPPED": "stopped",           # Остановлена
    "ERROR": "error"                # Ошибка
}

# Направления сделок
TRADE_DIRECTIONS = {
    "BUY": "BUY",   # Покупка
    "SELL": "SELL"  # Продажа
}

# Типы ордеров
ORDER_TYPES = {
    "MARKET": "MARKET",       # Рыночный ордер
    "LIMIT": "LIMIT",         # Лимитный ордер
    "STOP": "STOP",           # Стоп-ордер
    "STOP_MARKET": "STOP_MARKET", # Стоп-маркет ордер
    "TAKE_PROFIT": "TAKE_PROFIT", # Тейк-профит ордер
    "TAKE_PROFIT_MARKET": "TAKE_PROFIT_MARKET" # Тейк-профит маркет ордер
}

# Статусы ордеров
ORDER_STATUSES = {
    "NEW": "NEW",             # Новый ордер
    "PARTIALLY_FILLED": "PARTIALLY_FILLED", # Частично исполнен
    "FILLED": "FILLED",       # Исполнен
    "CANCELED": "CANCELED",   # Отменен
    "PENDING_CANCEL": "PENDING_CANCEL", # Ожидает отмены
    "REJECTED": "REJECTED",   # Отклонен
    "EXPIRED": "EXPIRED"      # Истек
}

# Интервалы времени
TIME_INTERVALS = {
    "1m": "1m",   # 1 минута
    "3m": "3m",   # 3 минуты
    "5m": "5m",   # 5 минут
    "15m": "15m", # 15 минут
    "30m": "30m", # 30 минут
    "1h": "1h",   # 1 час
    "2h": "2h",   # 2 часа
    "4h": "4h",   # 4 часа
    "6h": "6h",   # 6 часов
    "8h": "8h",   # 8 часов
    "12h": "12h", # 12 часов
    "1d": "1d",   # 1 день
    "3d": "3d",   # 3 дня
    "1w": "1w",   # 1 неделя
    "1M": "1M"    # 1 месяц
}

# Максимальное количество попыток
MAX_RETRIES = 3

# Таймауты
TIMEOUTS = {
    "API_TIMEOUT": 10,        # Таймаут API-запросов (в секундах)
    "WEBSOCKET_TIMEOUT": 30,  # Таймаут WebSocket-соединения (в секундах)
    "RETRY_DELAY": 1          # Задержка между повторными попытками (в секундах)
}

# Направления позиций
POSITION_SIDES = {
    "BUY": "BUY",
    "SELL": "SELL"
}

# Типы индикаторов
INDICATOR_TYPES = {
    "TREND": "trend",
    "MOMENTUM": "momentum",
    "VOLATILITY": "volatility",
    "VOLUME": "volume",
    "CUSTOM": "custom"
}

# Типы сигналов
SIGNAL_TYPES = {
    "BUY": "BUY",
    "SELL": "SELL",
    "NEUTRAL": "NEUTRAL",
    "STRONG_BUY": "STRONG_BUY",
    "STRONG_SELL": "STRONG_SELL"
}

# Уровни логирования
LOG_LEVELS = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL"
}

# Константы для Telegram бота
class TelegramCommands:
    """Команды Telegram бота."""
    START = "start"
    HELP = "help"
    STATUS = "status"
    TRADE = "trade"
    BALANCE = "balance"
    POSITIONS = "positions"
    MODE = "mode"

class TelegramCallbacks:
    """Колбэки для кнопок Telegram бота."""
    STATUS = "status"
    HELP = "help"
    MODES = "modes"
    TRADE = "trade"
    BACK_TO_MAIN = "back_to_main"
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE_ALL = "close_all"
    SET_MODE_PREFIX = "set_mode_" 