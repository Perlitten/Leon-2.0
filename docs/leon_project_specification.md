# Спецификация проекта Leon 2.0

## 1. Общее описание

Leon 2.0 - это автоматизированная торговая система для криптовалютного рынка с интеграцией машинного обучения, поддержкой режима симуляции (dry run) и уведомлениями через Telegram.

### 1.1 Цели проекта

- Создание надежной и масштабируемой архитектуры для торгового бота
- Интеграция с биржей Binance для получения данных и выполнения торговых операций
- Реализация режима симуляции (dry run) для безопасного тестирования стратегий
- Интеграция с Telegram для мониторинга и управления
- Внедрение современных методов машинного обучения для принятия торговых решений

### 1.2 Ключевые возможности

- Анализ рыночных данных в реальном времени
- Выполнение торговых операций на основе сигналов ML-модели
- Управление рисками и капиталом
- Мониторинг и уведомления через Telegram
- Режим симуляции для тестирования без реальных средств
- Сбор и анализ исторических данных для обучения ML-моделей
- Бэктестинг стратегий на исторических данных
- Продвинутая визуализация процесса торговли в консоли
- Реальная торговля с автоматическим исполнением ордеров
- Одновременное управление несколькими торговыми позициями

## 2. Архитектура системы

### 2.1 Общая структура

Система будет построена по модульному принципу с четким разделением ответственности между компонентами:

```
leon_2.0/
├── core/                  # Ядро системы
│   ├── orchestrator.py    # Оркестратор системы
│   ├── config_manager.py  # Управление конфигурацией
│   └── utils.py           # Общие утилиты
├── exchange/              # Интеграция с биржами
│   ├── binance/           # Модуль для работы с Binance
│   │   ├── client.py      # Клиент Binance API
│   │   ├── data_feed.py   # Получение данных
│   │   └── executor.py    # Исполнение ордеров
│   └── base.py            # Базовый интерфейс для бирж
├── trading/               # Торговые компоненты
│   ├── traders/           # Реализации трейдеров
│   │   ├── base.py        # Базовый класс трейдера
│   │   ├── real_trader.py # Реальная торговля
│   │   ├── dry_trader.py  # Симуляция торговли
│   │   └── backtest_trader.py # Бэктестинг
│   ├── strategies/        # Торговые стратегии
│   │   ├── base.py        # Базовый класс стратегии
│   │   └── scalping.py    # Стратегия скальпинга
│   └── risk/              # Управление рисками
│       └── risk_manager.py # Контроль рисков
├── ml/                    # Машинное обучение
│   ├── models/            # ML модели
│   │   ├── base.py        # Базовый класс модели
│   │   └── lstm.py        # LSTM модель
│   ├── features/          # Подготовка признаков
│   │   └── feature_engineering.py # Инженерия признаков
│   ├── training/          # Обучение моделей
│   │   └── trainer.py     # Тренер моделей
│   └── decision_maker.py  # Принятие решений на основе ML
├── notification/          # Уведомления
│   ├── telegram_bot.py    # Telegram бот
│   └── notification_manager.py # Управление уведомлениями
├── data/                  # Данные
│   ├── storage.py         # Хранение данных
│   └── processor.py       # Обработка данных
├── visualization/         # Визуализация
│   ├── console_ui.py      # Консольный интерфейс
│   └── chart_renderer.py  # Рендеринг графиков
├── config/                # Конфигурационные файлы
│   ├── config.yaml        # Основная конфигурация
│   └── logging_config.yaml # Настройки логирования
├── logs/                  # Логи
├── tests/                 # Тесты
└── main.py                # Точка входа
```

### 2.2 Взаимодействие компонентов

```
                                 +----------------+
                                 |                |
                                 |  Orchestrator  |
                                 |                |
                                 +-------+--------+
                                         |
                 +---------------------+ | +---------------------+
                 |                     | | |                     |
        +--------v--------+   +--------v-v------+   +-----------v---------+
        |                 |   |                 |   |                     |
        | Exchange Module <---> Trading Module <---> ML Decision Maker   |
        |                 |   |                 |   |                     |
        +-----------------+   +-----------------+   +---------------------+
                                      ^
                                      |
                 +--------------------+---------------------+
                 |                    |                     |
        +--------v--------+  +--------v--------+  +--------v--------+
        |                 |  |                 |  |                 |
        | Notification    |  | Visualization   |  | Data Storage    |
        |                 |  |                 |  |                 |
        +-----------------+  +-----------------+  +-----------------+
```

## 3. Детальное описание модулей

### 3.1 Core (Ядро)

#### 3.1.1 Orchestrator (orchestrator.py)

**Назначение**: Центральный компонент, координирующий работу всех модулей системы.

**Функциональность**:
- Инициализация и управление всеми компонентами
- Обработка событий и сигналов через шину событий
- Координация взаимодействия между модулями
- Переключение между режимами работы (dry, real, backtest)
- Обработка команд пользователя
- Управление жизненным циклом системы

**Компоненты**:
- `LeonOrchestrator`: Основной класс оркестратора
- `EventBus`: Шина событий для обмена сообщениями между компонентами
- `CommandProcessor`: Обработчик команд пользователя
- `TradingModeManager`: Менеджер режимов торговли
- `MLIntegrationManager`: Менеджер интеграции с ML-моделями

**Интерфейс**:
```python
class LeonOrchestrator:
    async def initialize() -> bool
    async def start(mode: Optional[str] = None) -> bool
    async def stop() -> bool
    async def switch_mode(mode: str) -> bool
    async def process_command(command: str, *args, **kwargs) -> Any
    def register_event_handler(event_type: str, handler: Callable) -> None
    async def emit_event(event_type: str, data: Any = None) -> None
    def get_status() -> Dict[str, Any]
```

#### 3.1.2 Config Manager (config_manager.py)

**Назначение**: Управление конфигурацией системы.

**Функциональность**:
- Загрузка конфигурации из YAML-файлов
- Валидация конфигурационных параметров
- Предоставление доступа к конфигурации для других модулей

**Интерфейс**:
```python
class ConfigManager:
    def load_config(config_path)
    def validate_config()
    def get_config(section, key=None)
    def update_config(section, key, value)
```

### 3.2 Exchange (Биржа)

#### 3.2.1 Binance Client (exchange/binance/client.py)

**Назначение**: Взаимодействие с API Binance.

**Функциональность**:
- Аутентификация на бирже
- Выполнение запросов к API
- Обработка ответов и ошибок

**Интерфейс**:
```python
class BinanceClient:
    async def initialize(api_key, api_secret)
    async def get_exchange_info()
    async def get_account_info()
    async def get_market_data(symbol, interval)
    async def place_order(symbol, side, type, quantity)
    async def cancel_order(symbol, order_id)
    async def get_open_orders(symbol)
```

#### 3.2.2 Data Feed (exchange/binance/data_feed.py)

**Назначение**: Получение и обработка рыночных данных.

**Функциональность**:
- Подписка на потоки данных (WebSocket)
- Получение исторических данных
- Обработка и нормализация данных

**Интерфейс**:
```python
class BinanceDataFeed:
    async def initialize()
    async def subscribe_to_klines(symbol, interval)
    async def subscribe_to_ticker(symbol)
    async def get_historical_klines(symbol, interval, start_time, end_time)
    async def process_kline_message(message)
```

### 3.3 Trading (Торговля)

#### 3.3.1 Base Trader (trading/traders/base.py)

**Назначение**: Базовый интерфейс для всех типов трейдеров.

**Функциональность**:
- Определение общего интерфейса для трейдеров
- Базовая логика управления позициями

**Интерфейс**:
```python
class TraderBase:
    async def initialize()
    async def enter_position(symbol, direction, size, price)
    async def exit_position(position_id, price)
    async def update_position(position_id, stop_loss, take_profit)
    async def get_open_positions()
    async def get_balance()
```

#### 3.3.2 Dry Trader (trading/traders/dry_trader.py)

**Назначение**: Симуляция торговли без реальных средств.

**Функциональность**:
- Виртуальное управление балансом
- Симуляция исполнения ордеров
- Отслеживание виртуальных позиций
- Расчет комиссий и P&L

**Интерфейс**:
```python
class DryModeTrader(TraderBase):
    async def initialize(initial_balance)
    async def enter_position(symbol, direction, size, price)
    async def exit_position(position_id, price)
    async def update_position(position_id, stop_loss, take_profit)
    async def get_performance_metrics()
```

#### 3.3.3 Risk Manager (trading/risk/risk_manager.py)

**Назначение**: Управление торговыми рисками.

**Функциональность**:
- Расчет размера позиции
- Установка стоп-лоссов и тейк-профитов
- Контроль максимальных убытков
- Анализ волатильности
- Управление портфелем позиций
- Распределение капитала между несколькими сделками

**Интерфейс**:
```python
class RiskManager:
    def calculate_position_size(balance, risk_per_trade, entry_price, stop_loss)
    def validate_trade(symbol, direction, size, current_positions)
    def calculate_risk_reward_ratio(entry, stop_loss, take_profit)
    def adjust_for_volatility(position_size, volatility)
    def calculate_max_concurrent_positions(balance, market_conditions)
    def allocate_capital(balance, open_positions, new_position_count)
    def calculate_portfolio_risk(positions)
```

#### 3.3.4 Backtest Trader (trading/traders/backtest_trader.py)

**Назначение**: Бэктестинг торговых стратегий на исторических данных.

**Функциональность**:
- Загрузка исторических данных
- Симуляция торговли на исторических данных
- Расчет метрик эффективности стратегии
- Генерация отчетов о результатах бэктестинга

**Интерфейс**:
```python
class BacktestTrader(TraderBase):
    async def initialize(initial_balance, start_date, end_date)
    async def run_backtest(strategy, symbol, interval)
    async def get_performance_metrics()
    async def generate_report()
    async def export_results(format="csv")
```

#### 3.3.5 Position Manager (trading/position_manager.py)

**Назначение**: Управление множественными торговыми позициями.

**Функциональность**:
- Отслеживание всех открытых позиций
- Управление жизненным циклом позиций
- Координация между несколькими одновременными сделками
- Приоритизация сделок на основе сигналов и рисков
- Балансировка портфеля позиций

**Интерфейс**:
```python
class PositionManager:
    async def initialize(max_concurrent_positions)
    async def open_position(symbol, direction, size, entry_price, stop_loss, take_profit)
    async def close_position(position_id, exit_price, reason)
    async def update_positions(market_data)
    async def get_open_positions()
    async def get_position_by_id(position_id)
    async def get_positions_by_symbol(symbol)
    async def calculate_portfolio_metrics()
    async def can_open_new_position(symbol, direction)
```

### 3.4 ML (Машинное обучение)

#### 3.4.1 Base Model (ml/models/base.py)

**Назначение**: Базовый интерфейс для ML-моделей.

**Функциональность**:
- Определение общего интерфейса для моделей
- Базовые методы для работы с моделями

**Интерфейс**:
```python
class BaseModel:
    def load(path)
    def save(path)
    def predict(features)
    def evaluate(features, targets)
```

#### 3.4.2 LSTM Model (ml/models/lstm.py)

**Назначение**: Реализация модели на основе LSTM для прогнозирования.

**Функциональность**:
- Создание и обучение LSTM-модели
- Прогнозирование движения цены
- Оценка точности модели

**Интерфейс**:
```python
class LSTMModel(BaseModel):
    def __init__(input_dim, hidden_dim, output_dim)
    def train(features, targets, epochs, batch_size)
    def predict(features)
    def evaluate(features, targets)
```

#### 3.4.3 Decision Maker (ml/decision_maker.py)

**Назначение**: Принятие торговых решений на основе ML-прогнозов.

**Функциональность**:
- Интерпретация результатов ML-модели
- Формирование торговых сигналов
- Оценка уверенности в прогнозе

**Интерфейс**:
```python
class MLDecisionMaker:
    def __init__(model, confidence_threshold)
    def make_decision(market_data, additional_features)
    def evaluate_confidence(prediction)
    def get_decision_metrics()
```

### 3.5 Notification (Уведомления)

#### 3.5.1 Telegram Bot (notification/telegram_bot.py)

**Назначение**: Взаимодействие с пользователем через Telegram.

**Функциональность**:
- Отправка уведомлений о торговых операциях
- Отправка отчетов о производительности
- Прием команд от пользователя
- Отображение статуса системы

**Интерфейс**:
```python
class TelegramBot:
    async def initialize(token, chat_id)
    async def send_message(message)
    async def send_trade_notification(trade_info)
    async def send_performance_report(report)
    async def process_command(command, args)
```

### 3.6 Visualization (Визуализация)

#### 3.6.1 Console UI (visualization/console_ui.py)

**Назначение**: Продвинутая визуализация процесса торговли в консоли.

**Функциональность**:
- Отображение текущих рыночных данных
- Визуализация открытых позиций
- Отображение истории торговли
- Визуализация метрик производительности
- Интерактивный интерфейс для управления ботом

**Интерфейс**:
```python
class ConsoleUI:
    def initialize()
    def update_market_data(market_data)
    def update_positions(positions)
    def update_balance(balance)
    def update_performance_metrics(metrics)
    def display_trade_signals(signals)
    def display_error(error)
    def render()
```

#### 3.6.2 Chart Renderer (visualization/chart_renderer.py)

**Назначение**: Рендеринг графиков и визуализаций для анализа.

**Функциональность**:
- Построение графиков цен
- Визуализация индикаторов
- Отображение точек входа и выхода
- Визуализация результатов бэктестинга

**Интерфейс**:
```python
class ChartRenderer:
    def render_price_chart(data, timeframe)
    def add_indicator(indicator_name, data)
    def mark_trade_points(entries, exits)
    def render_performance_chart(performance_data)
    def save_chart(filename)
    def display_chart()
```

## 4. Контракты данных

### 4.1 Рыночные данные

```python
class MarketData:
    symbol: str              # Торговая пара
    timestamp: int           # Временная метка
    open: float              # Цена открытия
    high: float              # Максимальная цена
    low: float               # Минимальная цена
    close: float             # Цена закрытия
    volume: float            # Объем
    interval: str            # Интервал (1m, 5m, 1h, etc.)
```

### 4.2 Торговое решение

```python
class TradeDecision:
    symbol: str              # Торговая пара
    direction: str           # Направление (LONG/SHORT)
    entry_price: float       # Цена входа
    stop_loss: float         # Уровень стоп-лосса
    take_profit: float       # Уровень тейк-профита
    size: float              # Размер позиции
    confidence: float        # Уверенность в решении (0-1)
    timestamp: int           # Временная метка
    reason: str              # Причина решения
```

### 4.3 Позиция

```python
class Position:
    id: str                  # Уникальный идентификатор
    symbol: str              # Торговая пара
    direction: str           # Направление (LONG/SHORT)
    entry_price: float       # Цена входа
    current_price: float     # Текущая цена
    size: float              # Размер позиции
    stop_loss: float         # Уровень стоп-лосса
    take_profit: float       # Уровень тейк-профита
    entry_time: int          # Время входа
    pnl: float               # Текущий P&L
    status: str              # Статус (OPEN/CLOSED)
```

## 5. Конфигурация

### 5.1 Основная конфигурация (config.yaml)

```yaml
general:
  symbol: "BTCUSDT"          # Торговая пара
  update_interval: 5         # Интервал обновления в секундах
  kline_interval: "1h"       # Интервал свечей
  mode: "dry"                # Режим работы: dry/real/backtest
  use_real_data: true        # Использовать реальные данные с биржи

risk:
  max_position_size: 0.01    # Максимальный размер позиции в BTC
  max_loss_percent: 2.0      # Максимальный допустимый убыток в процентах
  volatility_threshold: 0.05 # Порог волатильности
  leverage: 20               # Кредитное плечо
  max_concurrent_positions: 5 # Максимальное количество одновременных позиций
  max_positions_per_symbol: 2 # Максимальное количество позиций на одну пару
  portfolio_risk_limit: 5.0  # Максимальный риск портфеля в процентах
  capital_allocation: {      # Распределение капитала по типам позиций
    "high_confidence": 0.4,  # Высокая уверенность - 40% доступного капитала
    "medium_confidence": 0.3, # Средняя уверенность - 30% доступного капитала
    "low_confidence": 0.1    # Низкая уверенность - 10% доступного капитала
  }

strategy:
  name: "ml_based"           # Стратегия на основе ML
  params:
    confidence_threshold: 0.7 # Порог уверенности для входа в позицию

ml:
  enabled: true              # Включить ML-компонент
  model_type: "lstm"         # Тип модели
  model_path: "models/lstm_v1.h5" # Путь к модели
  feature_window: 24         # Окно признаков (количество свечей)
  prediction_horizon: 3      # Горизонт прогнозирования

telegram:
  enabled: true              # Включить уведомления в Telegram
  bot_token: "${TELEGRAM_BOT_TOKEN}" # Токен бота
  chat_id: "${TELEGRAM_CHAT_ID}"     # ID чата
  notification_level: "all"  # Уровень уведомлений (all/trades/important)

visualization:
  enabled: true              # Включить визуализацию
  console_ui: true           # Использовать консольный интерфейс
  update_interval: 1         # Интервал обновления UI в секундах
  chart_type: "candlestick"  # Тип графика (candlestick/line)
  
backtest:
  start_date: "2023-01-01"   # Начальная дата для бэктестинга
  end_date: "2023-12-31"     # Конечная дата для бэктестинга
  data_source: "binance"     # Источник данных
  
initial_balance: 1000        # Начальный баланс для dry mode и бэктестинга
```

## 6. Первая итерация

Первая итерация проекта должна включать:

1. **Интеграцию с Binance**:
   - Подключение к API Binance
   - Получение рыночных данных
   - Базовые торговые операции

2. **Интеграцию с Telegram**:
   - Отправка уведомлений
   - Базовые команды управления

3. **Режим Dry Run**:
   - Симуляция торговли без реальных средств
   - Отслеживание виртуального баланса и позиций

4. **ML-компонент**:
   - Базовая модель для прогнозирования
   - Интеграция с системой принятия решений

5. **Управление множественными позициями**:
   - Система управления несколькими одновременными сделками
   - Базовое распределение капитала
   - Контроль рисков при множественных позициях

## 7. Вторая итерация

Вторая итерация проекта должна включать:

1. **Бэктестинг**:
   - Загрузка и обработка исторических данных
   - Реализация системы бэктестинга
   - Анализ результатов и оптимизация стратегий

2. **Реальная торговля**:
   - Безопасное исполнение ордеров
   - Мониторинг позиций в реальном времени
   - Управление рисками в реальной торговле

3. **Продвинутая визуализация**:
   - Консольный интерфейс с богатой визуализацией
   - Графики и индикаторы в реальном времени
   - Интерактивное управление через консоль

## 8. Технический стек

- **Язык программирования**: Python 3.10+
- **Асинхронное программирование**: asyncio
- **Работа с API**: aiohttp, websockets
- **Машинное обучение**: TensorFlow/PyTorch, scikit-learn, pandas
- **Хранение данных**: SQLite/PostgreSQL
- **Логирование**: logging
- **Конфигурация**: PyYAML
- **Тестирование**: pytest
- **Визуализация**: rich, plotext, matplotlib
- **Бэктестинг**: pandas, numpy, custom framework

## 9. Дорожная карта

### Фаза 1: Базовая функциональность
- Настройка проекта и структуры
- Интеграция с Binance API
- Реализация режима Dry Run
- Базовая интеграция с Telegram

### Фаза 2: Бэктестинг и реальная торговля
- Разработка системы бэктестинга
- Реализация реальной торговли
- Продвинутая визуализация в консоли
- Расширенное управление рисками

### Фаза 3: ML-компонент
- Сбор и подготовка данных
- Разработка и обучение ML-модели
- Интеграция ML в систему принятия решений
- Оптимизация моделей на основе бэктестинга

### Фаза 4: Расширенная функциональность
- Улучшение управления рисками
- Расширенные возможности Telegram-бота
- Оптимизация производительности
- Расширенная аналитика и отчетность

### Фаза 5: Продвинутые функции
- Поддержка нескольких торговых пар
- Автоматическая оптимизация параметров
- Расширенные ML-модели
- Интеграция с дополнительными источниками данных

## 10. Технологический стек

### 10.1 Архитектурные паттерны

В проекте используются следующие архитектурные паттерны:

1. **Модульная архитектура** - система разделена на независимые модули, которые могут быть разработаны, протестированы и развернуты отдельно.
2. **Паттерн "Наблюдатель" (Observer)** - используется для реализации шины событий (EventBus) и обеспечения асинхронной обработки событий.
3. **Паттерн "Стратегия" (Strategy)** - используется для реализации различных торговых стратегий и алгоритмов.
4. **Паттерн "Фабрика" (Factory)** - используется для создания экземпляров трейдеров, стратегий и других компонентов.
5. **Паттерн "Команда" (Command)** - используется для реализации процессора команд и обработки пользовательских команд.
6. **Паттерн "Одиночка" (Singleton)** - используется для компонентов, которые должны существовать в единственном экземпляре (например, ConfigManager, LocalizationManager).

## 11. Текущая структура проекта

```
D:\Leon 2.0\
├── config\                # Конфигурационные файлы
├── core\                  # Ядро системы
├── data\                  # Данные
├── exchange\              # Интеграция с биржами
│   └── binance\           # Модуль для работы с Binance
├── logs\                  # Логи
├── ml\                    # Машинное обучение
│   ├── features\          # Подготовка признаков
│   ├── models\            # ML модели
│   └── training\          # Обучение моделей
├── notification\          # Уведомления
├── tests\                 # Тесты
├── trading\               # Торговые компоненты
│   ├── risk\              # Управление рисками
│   ├── strategies\        # Торговые стратегии
│   └── traders\           # Реализации трейдеров
├── venv\                  # Виртуальное окружение Python
├── config.yaml            # Текущий конфигурационный файл
├── dry_mode_trader.py     # Текущий файл трейдера в режиме симуляции
└── leon_project_specification.md  # Этот документ спецификации
```

Для поддержки новых функций бэктестинга и визуализации, необходимо добавить следующие директории:

```
D:\Leon 2.0\
├── ...
├── visualization\         # Визуализация
│   ├── console_ui.py      # Консольный интерфейс
│   └── chart_renderer.py  # Рендеринг графиков
├── ...
```

Эта структура соответствует архитектуре, описанной в разделе 2.1, и будет использоваться для организации кода проекта. В процессе разработки структура может быть дополнена новыми папками и файлами в соответствии с потребностями проекта.

## 8. Примечания по реализации

### 8.1 Миграция с LeonController на LeonOrchestrator

Функциональность класса `LeonController` из файла `leon_controller.py` была перенесена и расширена в модуле `core/orchestrator.py` с классом `LeonOrchestrator`. Новая реализация обеспечивает:

1. Более модульную архитектуру с четким разделением ответственности
2. Улучшенную обработку событий через систему `EventBus`
3. Гибкое управление режимами торговли через `TradingModeManager`
4. Интеграцию с ML-моделями через `MLIntegrationManager`
5. Централизованное управление визуализацией через `VisualizationManager`
6. Локализацию всех текстовых сообщений

Файл `leon_controller.py` был удален из проекта, так как вся его функциональность теперь реализована в новой архитектуре.

## 9. Текущая структура проекта
