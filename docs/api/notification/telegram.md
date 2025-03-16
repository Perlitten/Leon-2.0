# Модуль Telegram

Модуль `notification.telegram` предоставляет функциональность для отправки уведомлений и управления ботом через Telegram.

## Класс TelegramBot

Основной класс для работы с Telegram API.

```python
from notification.telegram.bot import TelegramBot
from core.config_manager import ConfigManager
from core.localization import LocalizationManager

# Инициализация
config_manager = ConfigManager()
localization = LocalizationManager()
telegram_bot = TelegramBot(config_manager, localization)

# Установка ссылки на оркестратор
telegram_bot.set_orchestrator(orchestrator)

# Запуск бота
await telegram_bot.start()

# Или запуск в фоновом режиме
telegram_bot.start_in_background()

# Отправка сообщения
await telegram_bot.send_message("Привет, мир!")

# Отправка уведомления о торговой операции
await telegram_bot.send_trade_notification(
    symbol="BTCUSDT",
    direction="BUY",
    price=50000.0,
    size=0.1,
    pnl=None
)

# Отправка обновления статуса
await telegram_bot.send_status_update(
    symbol="BTCUSDT",
    mode="dry",
    balance=1000.0,
    leverage=20,
    risk_per_trade=2.0,
    stop_loss=2.0,
    take_profit=3.0
)

# Остановка бота
await telegram_bot.stop()
```

### Конструктор

```python
def __init__(self, config_manager: ConfigManager, localization: LocalizationManager)
```

Параметры:
- `config_manager` - Менеджер конфигурации
- `localization` - Менеджер локализации

### Методы

#### start

```python
async def start(self)
```

Запускает Telegram бота.

#### start_in_background

```python
def start_in_background(self)
```

Запускает Telegram бота в отдельном потоке. Это позволяет не блокировать основной поток выполнения.

#### stop

```python
async def stop(self)
```

Останавливает Telegram бота.

#### set_orchestrator

```python
def set_orchestrator(self, orchestrator)
```

Устанавливает ссылку на оркестратор.

Параметры:
- `orchestrator` - Экземпляр оркестратора

#### send_message

```python
async def send_message(self, text: str, reply_markup: Optional[InlineKeyboardMarkup] = None) -> None
```

Отправляет сообщение в чат.

Параметры:
- `text` - Текст сообщения
- `reply_markup` - Разметка для кнопок (опционально)

#### send_trade_notification

```python
async def send_trade_notification(self, 
                                 symbol: str, 
                                 direction: str, 
                                 price: float, 
                                 size: float, 
                                 pnl: Optional[float] = None) -> None
```

Отправляет уведомление о торговой операции.

Параметры:
- `symbol` - Символ торговой пары
- `direction` - Направление сделки (BUY/SELL)
- `price` - Цена
- `size` - Размер позиции
- `pnl` - Прибыль/убыток (опционально)

#### send_status_update

```python
async def send_status_update(self, 
                            symbol: str, 
                            mode: str, 
                            balance: float, 
                            leverage: int, 
                            risk_per_trade: float,
                            stop_loss: float,
                            take_profit: float) -> None
```

Отправляет обновление статуса бота.

Параметры:
- `symbol` - Символ торговой пары
- `mode` - Режим работы
- `balance` - Баланс
- `leverage` - Плечо
- `risk_per_trade` - Риск на сделку
- `stop_loss` - Стоп-лосс
- `take_profit` - Тейк-профит

## Команды бота

Telegram бот поддерживает следующие команды:

- `/start` - Начать работу с ботом
- `/help` - Показать справку
- `/status` - Показать текущий статус бота
- `/trade` - Управление торговлей
- `/balance` - Показать текущий баланс
- `/positions` - Показать открытые позиции
- `/mode` - Изменить режим работы

## Интерактивные кнопки

Бот предоставляет интерактивный интерфейс с кнопками для удобного управления:

- Основное меню:
  - Статус
  - Помощь
  - Режимы
  - Торговля

- Меню торговли:
  - Открыть LONG
  - Открыть SHORT
  - Закрыть все позиции
  - Назад

- Меню режимов:
  - Симуляция
  - Бэктестинг
  - Реальная торговля
  - Назад

## Формат сообщений

### Уведомление о торговой операции

```
StableTrade
🚀 *Симуляция запущена*

🟢 *Открыта ДЛИННАЯ позиция*
◆ Пара: *USDCUSDT*
◆ Цена: *0.99978000*
◆ Размер: *399.42226832*
🕒 *2025-03-05 16:15:46*

💬 _Не считай деньги, пока они не превратились в стабильный поток. Сосредоточься на процессе._
```

### Обновление статуса

```
StableTrade
🚀 *Симуляция запущена*

◆ Символ: *USDCUSDT*
◆ Режим: *Симуляция (Dry Mode)*
◆ Начальный баланс: *500.0 USDT*
◆ Плечо: *20x*
◆ Риск на сделку: *2.0%*
◆ Стоп-лосс: *2.0%*
◆ Тейк-профит: *3.0%*
🕒 *2025-03-05 15:26:07*
``` 