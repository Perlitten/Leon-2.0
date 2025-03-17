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

# API для работы с Telegram ботом

## Обзор

API для работы с Telegram ботом предоставляет интерфейс для отправки уведомлений и управления ботом через мессенджер Telegram.

## Классы

### TelegramBot

Класс для отправки уведомлений и управления ботом через Telegram.

```python
class TelegramBot:
    def __init__(self, config_manager: ConfigManager, localization: LocalizationManager):
        """
        Инициализирует Telegram бота.
        
        Args:
            config_manager: Менеджер конфигурации
            localization: Менеджер локализации
        """
        pass
        
    def set_orchestrator(self, orchestrator: LeonOrchestrator) -> None:
        """
        Устанавливает ссылку на оркестратор.
        
        Args:
            orchestrator: Оркестратор
        """
        pass
        
    async def start(self) -> bool:
        """
        Запускает Telegram бота.
        
        Returns:
            bool: Успешность запуска
        """
        pass
        
    async def stop(self) -> bool:
        """
        Останавливает Telegram бота.
        
        Returns:
            bool: Успешность остановки
        """
        pass
        
    async def send_message(self, text: str, parse_mode: str = None) -> bool:
        """
        Отправляет сообщение.
        
        Args:
            text: Текст сообщения
            parse_mode: Режим форматирования текста
            
        Returns:
            bool: Успешность отправки
        """
        pass
        
    async def send_trade_notification(self, symbol: str, direction: str, price: float, size: float, pnl: float = None) -> bool:
        """
        Отправляет уведомление о торговой операции.
        
        Args:
            symbol: Символ торговой пары
            direction: Направление сделки ("BUY" или "SELL")
            price: Цена
            size: Размер позиции
            pnl: Прибыль/убыток (опционально)
            
        Returns:
            bool: Успешность отправки
        """
        pass
        
    async def send_status_update(self, symbol: str, mode: str, balance: float, leverage: int, risk_per_trade: float, stop_loss: float, take_profit: float) -> bool:
        """
        Отправляет обновление статуса.
        
        Args:
            symbol: Символ торговой пары
            mode: Режим работы бота
            balance: Баланс
            leverage: Кредитное плечо
            risk_per_trade: Риск на сделку
            stop_loss: Стоп-лосс
            take_profit: Тейк-профит
            
        Returns:
            bool: Успешность отправки
        """
        pass
```

## Методы

### send_message()

Отправляет сообщение через Telegram.

```python
async def send_message(self, text: str, parse_mode: str = None) -> bool:
    """
    Отправляет сообщение.
    
    Args:
        text: Текст сообщения
        parse_mode: Режим форматирования текста
        
    Returns:
        bool: Успешность отправки
    """
    try:
        # Проверяем, инициализирован ли бот
        if not self.bot:
            self.logger.error("Telegram бот не инициализирован")
            return False
            
        # Получаем ID чата из конфигурации
        chat_id = self.config_manager.get_config().get("telegram", {}).get("chat_id")
        if not chat_id:
            self.logger.error("ID чата не указан в конфигурации")
            return False
            
        # Отправляем сообщение
        await self.bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=parse_mode
        )
        
        return True
    except Exception as e:
        self.logger.error(f"Ошибка при отправке сообщения: {str(e)}")
        self.logger.debug(traceback.format_exc())
        return False
```

### send_trade_notification()

Отправляет уведомление о торговой операции.

```python
async def send_trade_notification(self, symbol: str, direction: str, price: float, size: float, pnl: float = None) -> bool:
    """
    Отправляет уведомление о торговой операции.
    
    Args:
        symbol: Символ торговой пары
        direction: Направление сделки ("BUY" или "SELL")
        price: Цена
        size: Размер позиции
        pnl: Прибыль/убыток (опционально)
        
    Returns:
        bool: Успешность отправки
    """
    try:
        # Формируем текст сообщения
        emoji = "🟢" if direction == "BUY" else "🔴"
        message = f"{emoji} {self.localization.get('trade_notification')}\n\n"
        message += f"🔹 {self.localization.get('symbol')}: {symbol}\n"
        message += f"🔹 {self.localization.get('direction')}: {direction}\n"
        message += f"🔹 {self.localization.get('price')}: {price:.2f}\n"
        message += f"🔹 {self.localization.get('size')}: {size:.4f}\n"
        
        if pnl is not None:
            emoji_pnl = "✅" if pnl >= 0 else "❌"
            message += f"🔹 {self.localization.get('pnl')}: {emoji_pnl} {pnl:.2f}\n"
            
        # Отправляем сообщение
        return await self.send_message(message)
    except Exception as e:
        self.logger.error(f"Ошибка при отправке уведомления о торговой операции: {str(e)}")
        self.logger.debug(traceback.format_exc())
        return False
```

### send_status_update()

Отправляет обновление статуса.

```python
async def send_status_update(self, symbol: str, mode: str, balance: float, leverage: int, risk_per_trade: float, stop_loss: float, take_profit: float) -> bool:
    """
    Отправляет обновление статуса.
    
    Args:
        symbol: Символ торговой пары
        mode: Режим работы бота
        balance: Баланс
        leverage: Кредитное плечо
        risk_per_trade: Риск на сделку
        stop_loss: Стоп-лосс
        take_profit: Тейк-профит
        
    Returns:
        bool: Успешность отправки
    """
    try:
        # Формируем текст сообщения
        message = f"📊 {self.localization.get('status_update')}\n\n"
        message += f"🔹 {self.localization.get('symbol')}: {symbol}\n"
        message += f"🔹 {self.localization.get('mode')}: {mode}\n"
        message += f"🔹 {self.localization.get('balance')}: {balance:.2f}\n"
        message += f"🔹 {self.localization.get('leverage')}: {leverage}x\n"
        message += f"🔹 {self.localization.get('risk_per_trade')}: {risk_per_trade:.2f}%\n"
        message += f"🔹 {self.localization.get('stop_loss')}: {stop_loss:.2f}%\n"
        message += f"🔹 {self.localization.get('take_profit')}: {take_profit:.2f}%\n"
            
        # Отправляем сообщение
        return await self.send_message(message)
    except Exception as e:
        self.logger.error(f"Ошибка при отправке обновления статуса: {str(e)}")
        self.logger.debug(traceback.format_exc())
        return False
```

## Обработчики команд

### _handle_start()

Обрабатывает команду `/start`.

```python
async def _handle_start(self, update: Update, context: CallbackContext) -> None:
    """
    Обрабатывает команду /start.
    
    Args:
        update: Объект обновления
        context: Контекст
    """
    try:
        # Проверяем авторизацию пользователя
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text(self.localization.get("unauthorized"))
            return
            
        # Отправляем приветственное сообщение
        await update.message.reply_text(
            self.localization.get("welcome_message"),
            reply_markup=self._get_main_menu_keyboard()
        )
    except Exception as e:
        self.logger.error(f"Ошибка при обработке команды /start: {str(e)}")
        self.logger.debug(traceback.format_exc())
```

### _handle_help()

Обрабатывает команду `/help`.

```python
async def _handle_help(self, update: Update, context: CallbackContext) -> None:
    """
    Обрабатывает команду /help.
    
    Args:
        update: Объект обновления
        context: Контекст
    """
    try:
        # Проверяем авторизацию пользователя
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text(self.localization.get("unauthorized"))
            return
            
        # Отправляем справку
        await update.message.reply_text(
            self.localization.get("help_message"),
            parse_mode="Markdown"
        )
    except Exception as e:
        self.logger.error(f"Ошибка при обработке команды /help: {str(e)}")
        self.logger.debug(traceback.format_exc())
```

## Обработчики колбэков

### _handle_pause_bot()

Обрабатывает колбэк паузы бота.

```python
async def _handle_pause_bot(self, query: CallbackQuery) -> None:
    """
    Обрабатывает паузу бота.
    
    Args:
        query: Объект колбэк-запроса
    """
    try:
        # Проверяем, инициализирован ли оркестратор
        if not self.orchestrator:
            await query.edit_message_text(self.localization.get("orchestrator_not_initialized"))
            return
            
        # Приостанавливаем работу бота
        success = await self.orchestrator.pause()
        
        if success:
            await query.edit_message_text(
                self.localization.get("bot_paused"),
                reply_markup=self._get_main_menu_keyboard()
            )
        else:
            await query.edit_message_text(
                self.localization.get("bot_pause_failed"),
                reply_markup=self._get_main_menu_keyboard()
            )
    except Exception as e:
        self.logger.error(f"Ошибка при обработке паузы бота: {str(e)}")
        self.logger.debug(traceback.format_exc())
        await query.edit_message_text(
            f"{self.localization.get('error')}: {str(e)}",
            reply_markup=self._get_main_menu_keyboard()
        )
```

### _handle_resume_bot()

Обрабатывает колбэк возобновления работы бота.

```python
async def _handle_resume_bot(self, query: CallbackQuery) -> None:
    """
    Обрабатывает возобновление работы бота.
    
    Args:
        query: Объект колбэк-запроса
    """
    try:
        # Проверяем, инициализирован ли оркестратор
        if not self.orchestrator:
            await query.edit_message_text(self.localization.get("orchestrator_not_initialized"))
            return
            
        # Возобновляем работу бота
        success = await self.orchestrator.resume()
        
        if success:
            await query.edit_message_text(
                self.localization.get("bot_resumed"),
                reply_markup=self._get_main_menu_keyboard()
            )
        else:
            await query.edit_message_text(
                self.localization.get("bot_resume_failed"),
                reply_markup=self._get_main_menu_keyboard()
            )
    except Exception as e:
        self.logger.error(f"Ошибка при обработке возобновления работы бота: {str(e)}")
        self.logger.debug(traceback.format_exc())
        await query.edit_message_text(
            f"{self.localization.get('error')}: {str(e)}",
            reply_markup=self._get_main_menu_keyboard()
        )
```

### _handle_train_model()

Обрабатывает колбэк обучения модели.

```python
async def _handle_train_model(self, query: CallbackQuery) -> None:
    """
    Обрабатывает обучение модели.
    
    Args:
        query: Объект колбэк-запроса
    """
    try:
        # Проверяем, инициализирован ли оркестратор
        if not self.orchestrator:
            await query.edit_message_text(self.localization.get("orchestrator_not_initialized"))
            return
            
        # Сообщаем о начале обучения
        await query.edit_message_text(self.localization.get("model_training_started"))
        
        # Запускаем обучение модели
        result = await self.orchestrator.ml_integration_manager.train_model()
        
        if result.get("success", False):
            # Получаем метрики
            metrics = result.get("metrics", {})
            accuracy = metrics.get("accuracy", 0)
            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)
            f1_score = metrics.get("f1_score", 0)
            
            # Форматируем сообщение с результатами
            message = self.localization.get("model_training_success").format(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score
            )
            
            await query.edit_message_text(
                text=message,
                reply_markup=self._get_main_menu_keyboard()
            )
        else:
            error = result.get("error", self.localization.get("unknown_error"))
            await query.edit_message_text(
                self.localization.get("model_training_failed").format(error=error),
                reply_markup=self._get_main_menu_keyboard()
            )
    except Exception as e:
        self.logger.error(f"Ошибка при обработке обучения модели: {str(e)}")
        self.logger.debug(traceback.format_exc())
        await query.edit_message_text(
            f"{self.localization.get('error')}: {str(e)}",
            reply_markup=self._get_main_menu_keyboard()
        )
```

### _handle_skip_training()

Обрабатывает колбэк пропуска обучения.

```python
async def _handle_skip_training(self, query: CallbackQuery) -> None:
    """
    Обрабатывает пропуск обучения.
    
    Args:
        query: Объект колбэк-запроса
    """
    try:
        # Проверяем, инициализирован ли оркестратор
        if not self.orchestrator:
            await query.edit_message_text(self.localization.get("orchestrator_not_initialized"))
            return
            
        # Получаем имя модели из конфигурации
        model_name = self.orchestrator.config_manager.get_config().get("ml", {}).get("model_name", "default")
        
        # Сообщаем о загрузке модели
        await query.edit_message_text(self.localization.get("model_loading").format(model_name=model_name))
        
        # Загружаем модель
        success = await self.orchestrator.ml_integration_manager.load_model(model_name)
        
        if success:
            await query.edit_message_text(
                self.localization.get("model_loading_success").format(model_name=model_name),
                reply_markup=self._get_main_menu_keyboard()
            )
        else:
            await query.edit_message_text(
                self.localization.get("model_loading_failed").format(model_name=model_name),
                reply_markup=self._get_main_menu_keyboard()
            )
    except Exception as e:
        self.logger.error(f"Ошибка при обработке пропуска обучения: {str(e)}")
        self.logger.debug(traceback.format_exc())
        await query.edit_message_text(
            f"{self.localization.get('error')}: {str(e)}",
            reply_markup=self._get_main_menu_keyboard()
        )
```

## Примеры использования

### Инициализация и запуск бота

```python
from notification.telegram import TelegramBot
from core.config_manager import ConfigManager
from core.localization import LocalizationManager
from core.orchestrator import LeonOrchestrator

# Создание экземпляра бота
config_manager = ConfigManager("config.yaml")
localization = LocalizationManager("locales")
telegram_bot = TelegramBot(config_manager, localization)

# Создание экземпляра оркестратора
orchestrator = LeonOrchestrator(config_manager, localization)

# Установка ссылки на оркестратор
telegram_bot.set_orchestrator(orchestrator)

# Запуск бота
await telegram_bot.start()
```

### Отправка уведомления

```python
# Отправка простого сообщения
await telegram_bot.send_message("Привет! Это тестовое уведомление.")

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
    leverage=10,
    risk_per_trade=1.0,
    stop_loss=2.0,
    take_profit=4.0
)
```

## Конфигурация

Параметры Telegram бота задаются в конфигурационном файле `config.yaml`:

```yaml
telegram:
  token: "YOUR_TELEGRAM_BOT_TOKEN"
  chat_id: 123456789
  authorized_users:
    - 123456789
  proxy: null  # или "socks5://user:pass@host:port"
```

## Локализация

Тексты сообщений хранятся в YAML-файлах в директории `locales/` и загружаются через `LocalizationManager`. Пример файла локализации:

```yaml
# locales/ru.yaml
welcome_message: "Добро пожаловать в Leon Trading Bot!"
help_message: "Список доступных команд:\n/start - Начало работы\n/help - Справка\n/status - Статус бота\n/trade - Меню торговли\n/balance - Баланс\n/positions - Позиции\n/mode - Режим работы"
unauthorized: "⚠️ Вы не авторизованы для использования этого бота."
orchestrator_not_initialized: "⚠️ Оркестратор не инициализирован."
bot_paused: "⏸️ Бот приостановлен. Используйте кнопку 'Продолжить' для возобновления работы."
bot_pause_failed: "⚠️ Не удалось приостановить бота. Возможно, он не запущен или уже приостановлен."
bot_resumed: "▶️ Работа бота возобновлена."
bot_resume_failed: "⚠️ Не удалось возобновить работу бота. Возможно, он не запущен или не приостановлен."
model_training_started: "🧠 Начинаю обучение модели. Это может занять некоторое время..."
model_training_success: "✅ Обучение модели завершено успешно!\n\n📊 Метрики:\n- Точность (accuracy): {accuracy:.4f}\n- Precision: {precision:.4f}\n- Recall: {recall:.4f}\n- F1-score: {f1_score:.4f}\n\nМодель готова к использованию."
model_training_failed: "❌ Ошибка при обучении модели: {error}"
model_loading: "🔄 Загрузка модели '{model_name}'..."
model_loading_success: "✅ Модель '{model_name}' успешно загружена!"
model_loading_failed: "❌ Ошибка при загрузке модели '{model_name}'."
error: "❌ Ошибка"
unknown_error: "Неизвестная ошибка"
trade_notification: "Торговая операция"
status_update: "Обновление статуса"
symbol: "Символ"
direction: "Направление"
price: "Цена"
size: "Размер"
pnl: "Прибыль/убыток"
mode: "Режим"
balance: "Баланс"
leverage: "Плечо"
risk_per_trade: "Риск на сделку"
stop_loss: "Стоп-лосс"
take_profit: "Тейк-профит"
```

## Безопасность

Бот проверяет ID пользователя при каждом взаимодействии и разрешает доступ только авторизованным пользователям, указанным в конфигурации.

```python
def _is_authorized(self, user_id: int) -> bool:
    """
    Проверяет, авторизован ли пользователь.
    
    Args:
        user_id: ID пользователя
        
    Returns:
        bool: True, если пользователь авторизован, иначе False
    """
    authorized_users = self.config_manager.get_config().get("telegram", {}).get("authorized_users", [])
    return user_id in authorized_users
``` 