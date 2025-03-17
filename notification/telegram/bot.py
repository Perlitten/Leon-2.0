"""
Модуль Telegram бота для Leon Trading Bot.

Предоставляет функциональность для отправки уведомлений и управления ботом через Telegram.
"""

import os
import logging
from typing import Dict, List, Optional, Callable, Any, Union
import asyncio
from datetime import datetime
import threading
import traceback

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    CallbackQueryHandler,
    ContextTypes,
    filters
)

from core.config_manager import ConfigManager
from core.constants import TradingModes, TradeDirections, TelegramCommands, TelegramCallbacks, TELEGRAM_COMMANDS, TELEGRAM_BUTTONS, PAUSE_BOT, RESUME_BOT, STOP_BOT, RESTART_BOT, TRAIN_MODEL, SKIP_TRAINING
from core.localization import LocalizationManager

logger = logging.getLogger(__name__)

class TelegramBot:
    """
    Класс для управления Telegram ботом.
    
    Предоставляет функциональность для отправки уведомлений и управления торговым ботом через Telegram.
    """
    
    def __init__(self, config_manager: ConfigManager, localization: LocalizationManager):
        """
        Инициализация Telegram бота.
        
        Args:
            config_manager: Менеджер конфигурации
            localization: Менеджер локализации
        """
        self.config = config_manager
        self.localization = localization
        self.token = self.config.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = self.config.get("TELEGRAM_CHAT_ID")
        
        # Флаг активности бота
        self.is_running = False
        
        # Обработчики команд
        self.command_handlers = {}
        
        # Обработчики колбэков
        self.callback_handlers = {}
        
        # Приложение Telegram
        self.app = None
        
        # Ссылка на оркестратор (будет установлена позже)
        self.orchestrator = None
        
        logger.info("Telegram бот инициализирован")
    
    async def start(self):
        """Запуск Telegram бота."""
        if self.is_running:
            logger.warning("Telegram бот уже запущен")
            return
        
        if not self.token:
            logger.error("Не указан токен Telegram бота")
            return
        
        try:
            # Создание приложения
            self.app = Application.builder().token(self.token).build()
            
            # Регистрация обработчиков команд
            self.register_handlers()
            
            # Запуск бота
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            
            self.is_running = True
            logger.info("Telegram бот запущен")
            
            # Отправка приветственного сообщения
            await self.send_message(self.localization.get("telegram.bot_started"))
            
        except Exception as e:
            logger.error(f"Ошибка при запуске Telegram бота: {e}")
    
    async def stop(self):
        """Остановка Telegram бота."""
        if not self.is_running:
            logger.warning("Telegram бот не запущен")
            return
        
        try:
            # Отправка сообщения о завершении работы
            await self.send_message(self.localization.get("telegram.bot_stopped"))
            
            # Остановка бота
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            
            self.is_running = False
            logger.info("Telegram бот остановлен")
            
        except Exception as e:
            logger.error(f"Ошибка при остановке Telegram бота: {e}")
    
    def register_handlers(self):
        """Регистрация обработчиков команд."""
        # Базовые команды
        self.app.add_handler(CommandHandler(TelegramCommands.START, self.cmd_start))
        self.app.add_handler(CommandHandler(TelegramCommands.HELP, self.cmd_help))
        self.app.add_handler(CommandHandler(TelegramCommands.STATUS, self.cmd_status))
        
        # Команды управления торговлей
        self.app.add_handler(CommandHandler(TelegramCommands.TRADE, self.cmd_trade))
        self.app.add_handler(CommandHandler(TelegramCommands.BALANCE, self.cmd_balance))
        self.app.add_handler(CommandHandler(TelegramCommands.POSITIONS, self.cmd_positions))
        
        # Команды управления режимами
        self.app.add_handler(CommandHandler(TelegramCommands.MODE, self.cmd_mode))
        
        # Обработчик колбэков от кнопок
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Обработчик текстовых сообщений
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        logger.info("Обработчики команд Telegram бота зарегистрированы")
    
    def set_orchestrator(self, orchestrator):
        """
        Установка ссылки на оркестратор.
        
        Args:
            orchestrator: Экземпляр оркестратора
        """
        self.orchestrator = orchestrator
        logger.info("Установлена ссылка на оркестратор")
    
    def start_in_background(self):
        """Запуск Telegram бота в отдельном потоке."""
        if self.is_running:
            logger.warning("Telegram бот уже запущен")
            return
        
        async def _start_bot():
            await self.start()
        
        # Создание и запуск задачи в фоновом режиме
        loop = asyncio.new_event_loop()
        t = threading.Thread(target=self._run_bot_in_thread, args=(loop, _start_bot), daemon=True)
        t.start()
        logger.info("Telegram бот запущен в фоновом режиме")
    
    def _run_bot_in_thread(self, loop, coro_func):
        """
        Запуск корутины в отдельном потоке.
        
        Args:
            loop: Цикл событий
            coro_func: Корутина для запуска
        """
        asyncio.set_event_loop(loop)
        loop.run_until_complete(coro_func())
        loop.run_forever()
    
    async def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """
        Отправляет сообщение в Telegram.
        
        Args:
            text: Текст сообщения
            parse_mode: Режим форматирования текста (Markdown, HTML)
            
        Returns:
            bool: True, если сообщение успешно отправлено, иначе False
        """
        if not self.token or not self.chat_id:
            logger.warning("Попытка отправить сообщение без токена или chat_id")
            return False
        
        try:
            # Если приложение не инициализировано, создаем временный бот
            if not self.app or not self.is_running:
                from telegram import Bot
                bot = Bot(token=self.token)
                await bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    parse_mode=parse_mode
                )
            else:
                # Используем существующее приложение
                await self.app.bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    parse_mode=parse_mode
                )
            
            logger.debug(f"Отправлено сообщение в Telegram: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Ошибка при отправке сообщения в Telegram: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    async def send_message_with_keyboard(self, message: str, keyboard: List[List[Dict[str, str]]], parse_mode: str = "Markdown") -> None:
        """
        Отправка сообщения с клавиатурой в Telegram.
        
        Args:
            message: Текст сообщения
            keyboard: Список кнопок для клавиатуры в формате [[ {"text": "Текст кнопки", "callback_data": "callback_data"} ]]
            parse_mode: Режим форматирования текста
        """
        if not self.is_running or not self.chat_id:
            logger.warning("Невозможно отправить сообщение: бот не запущен или не указан chat_id")
            return
        
        try:
            # Преобразуем словари кнопок в объекты InlineKeyboardButton
            inline_keyboard = []
            for row in keyboard:
                inline_row = []
                for button in row:
                    inline_row.append(InlineKeyboardButton(
                        text=button.get("text", "Button"),
                        callback_data=button.get("callback_data", "unknown")
                    ))
                inline_keyboard.append(inline_row)
            
            # Создаем разметку клавиатуры
            reply_markup = InlineKeyboardMarkup(inline_keyboard)
            
            # Отправляем сообщение с клавиатурой
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode,
                reply_markup=reply_markup
            )
            logger.debug(f"Сообщение с клавиатурой отправлено в Telegram")
        except Exception as e:
            logger.error(f"Ошибка при отправке сообщения с клавиатурой в Telegram: {e}")
            logger.debug(traceback.format_exc())
    
    async def send_photo(self, photo_path: str, caption: str = None) -> None:
        """
        Отправка фотографии в Telegram.
        
        Args:
            photo_path: Путь к фотографии
            caption: Подпись к фотографии
        """
        if not self.is_running or not self.chat_id:
            logger.warning("Невозможно отправить фотографию: бот не запущен или не указан chat_id")
            return
        
        try:
            await self.app.bot.send_photo(
                chat_id=self.chat_id,
                photo=open(photo_path, 'rb'),
                caption=caption
            )
            logger.debug(f"Фотография отправлена в Telegram")
        except Exception as e:
            logger.error(f"Ошибка при отправке фотографии в Telegram: {e}")
    
    async def send_trade_notification(self, 
                                     symbol: str, 
                                     direction: str, 
                                     price: float, 
                                     size: float, 
                                     pnl: float = None,
                                     is_open: bool = True) -> None:
        """
        Отправка уведомления о сделке в Telegram.
        
        Args:
            symbol: Символ торговой пары
            direction: Направление сделки (BUY/SELL)
            price: Цена исполнения
            size: Размер позиции
            pnl: Прибыль/убыток (для закрытия позиции)
            is_open: True для открытия позиции, False для закрытия
        """
        if not self.is_running or not self.chat_id:
            logger.warning("Невозможно отправить уведомление: бот не запущен или не указан chat_id")
            return
        
        try:
            # Определяем тип сообщения и эмодзи
            if is_open:
                title = "🔔 *ОТКРЫТА НОВАЯ ПОЗИЦИЯ*"
                emoji = "🟢" if direction == "BUY" or direction == "LONG" else "🔴"
                direction_text = "LONG" if direction == "BUY" else "SHORT" if direction == "SELL" else direction
            else:
                title = "🔔 *ПОЗИЦИЯ ЗАКРЫТА*"
                emoji = "💰" if pnl and pnl > 0 else "📉"
                direction_text = "LONG" if direction == "BUY" else "SHORT" if direction == "SELL" else direction
            
            # Формируем сообщение в виде таблицы
            message = f"{title}\n\n"
            message += "```\n"
            message += "╔══════════════════════════════════════════════╗\n"
            message += f"║ {emoji} {direction_text} {symbol}".ljust(49) + "║\n"
            message += "╠══════════════════════════════════════════════╣\n"
            message += f"║ Цена: {price:.2f}".ljust(49) + "║\n"
            message += f"║ Размер: {size:.6f}".ljust(49) + "║\n"
            
            # Добавляем P&L для закрытия позиции
            if not is_open and pnl is not None:
                pnl_color = "[ЗЕЛЕНЫЙ]" if pnl >= 0 else "[КРАСНЫЙ]"
                message += f"║ P&L: {pnl_color} {pnl:.2f} USDT".ljust(49) + "║\n"
            
            # Добавляем время
            import datetime
            now = datetime.datetime.now().strftime("%H:%M:%S")
            message += f"║ Время: {now}".ljust(49) + "║\n"
            message += "╚══════════════════════════════════════════════╝\n"
            message += "```\n"
            
            # Добавляем кнопки управления для открытия позиции
            if is_open:
                keyboard = [
                    [
                        {"text": "📊 Статус", "callback_data": "status"},
                        {"text": "❌ Закрыть позицию", "callback_data": f"close_position_{symbol}_{direction_text}"}
                    ]
                ]
                await self.send_message_with_keyboard(message, keyboard)
            else:
                await self.send_message(message)
            
            logger.debug(f"Уведомление о {'открытии' if is_open else 'закрытии'} позиции отправлено в Telegram")
        except Exception as e:
            logger.error(f"Ошибка при отправке уведомления о сделке в Telegram: {e}")
            logger.debug(traceback.format_exc())
    
    async def send_session_summary(self, 
                                  total_trades: int, 
                                  win_trades: int, 
                                  loss_trades: int, 
                                  total_profit: float, 
                                  max_profit: float, 
                                  max_loss: float,
                                  duration_hours: float) -> None:
        """
        Отправка итогов торговой сессии в Telegram.
        
        Args:
            total_trades: Общее количество сделок
            win_trades: Количество прибыльных сделок
            loss_trades: Количество убыточных сделок
            total_profit: Общая прибыль/убыток
            max_profit: Максимальная прибыль по одной сделке
            max_loss: Максимальный убыток по одной сделке
            duration_hours: Продолжительность сессии в часах
        """
        if not self.is_running or not self.chat_id:
            logger.warning("Невозможно отправить итоги сессии: бот не запущен или не указан chat_id")
            return
        
        try:
            # Рассчитываем процент выигрышных сделок
            win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Определяем эмодзи для итогов
            result_emoji = "🎉" if total_profit > 0 else "😢"
            
            # Формируем сообщение в виде таблицы
            message = f"📊 *ИТОГИ ТОРГОВОЙ СЕССИИ* {result_emoji}\n\n"
            
            # Получаем случайную юмористическую фразу
            humor_phrases = [
                "Ваш электронный финансовый самоубийца завершил работу!",
                "Надеюсь, вы не заложили квартиру перед запуском этого бота!",
                "Поздравляем! Теперь вы знаете, как НЕ надо торговать!",
                "Если бы вы поставили на красное в казино, результат был бы лучше!"
            ]
            import random
            humor_phrase = random.choice(humor_phrases)
            message += f"_{humor_phrase}_\n\n"
            
            message += "```\n"
            message += "╔══════════════════════════════════════════════╗\n"
            message += f"║ СТАТИСТИКА СЕССИИ:".ljust(49) + "║\n"
            message += "╠══════════════════════════════════════════════╣\n"
            message += f"║ Продолжительность: {duration_hours:.1f} ч.".ljust(49) + "║\n"
            message += f"║ Всего сделок: {total_trades}".ljust(49) + "║\n"
            message += f"║ Выигрышных: {win_trades} ({win_rate:.1f}%)".ljust(49) + "║\n"
            message += f"║ Проигрышных: {loss_trades}".ljust(49) + "║\n"
            
            # Добавляем информацию о прибыли/убытке
            profit_color = "[ЗЕЛЕНЫЙ]" if total_profit >= 0 else "[КРАСНЫЙ]"
            message += f"║ Общий P&L: {profit_color} {total_profit:.2f} USDT".ljust(49) + "║\n"
            
            # Добавляем информацию о максимальной прибыли и убытке
            message += f"║ Макс. прибыль: {max_profit:.2f} USDT".ljust(49) + "║\n"
            message += f"║ Макс. убыток: {max_loss:.2f} USDT".ljust(49) + "║\n"
            
            # Добавляем время
            import datetime
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message += f"║ Завершено: {now}".ljust(49) + "║\n"
            message += "╚══════════════════════════════════════════════╝\n"
            message += "```\n"
            
            # Добавляем кнопку для запуска нового сеанса
            keyboard = [
                [
                    {"text": "🚀 Запустить новый сеанс", "callback_data": "start_new_session"}
                ]
            ]
            
            await self.send_message_with_keyboard(message, keyboard)
            logger.debug("Итоги торговой сессии отправлены в Telegram")
        except Exception as e:
            logger.error(f"Ошибка при отправке итогов сессии в Telegram: {e}")
            logger.debug(traceback.format_exc())
    
    async def send_status_update(self, symbol: str, mode: str, 
                                balance: float, leverage: int,
                                risk_per_trade: float, stop_loss: float,
                                take_profit: float) -> bool:
        """
        Отправляет обновление статуса торгового бота.
        
        Args:
            symbol: Символ торговой пары
            mode: Режим работы (dry/real)
            balance: Баланс счета
            leverage: Кредитное плечо
            risk_per_trade: Риск на сделку (%)
            stop_loss: Стоп-лосс (%)
            take_profit: Тейк-профит (%)
            
        Returns:
            bool: True, если сообщение успешно отправлено, иначе False
        """
        try:
            mode_emoji = "🧪" if mode == "dry" else "🔥"
            
            # Используем стандартные тексты
            message = f"📊 *Статус торгового бота*\n\n"
            message += f"{mode_emoji} Режим: {mode.upper()}\n"
            message += f"💱 Пара: {symbol}\n"
            message += f"💰 Баланс: {balance:.2f} USDT\n"
            message += f"⚡ Плечо: {leverage}x\n"
            message += f"⚠️ Риск на сделку: {risk_per_trade}%\n"
            message += f"🛑 Стоп-лосс: {stop_loss}%\n"
            message += f"🎯 Тейк-профит: {take_profit}%\n"
            
            # Отправляем сообщение
            await self.send_message(message)
            logger.info(f"Отправлено обновление статуса в Telegram")
            return True
        except Exception as e:
            logger.error(f"Ошибка при отправке статуса в Telegram: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def _get_mode_display(self, mode: str) -> str:
        """
        Получение отображаемого названия режима.
        
        Args:
            mode: Код режима
            
        Returns:
            Отображаемое название режима
        """
        mode_map = {
            TradingModes.DRY: "Симуляция (Dry Mode)",
            TradingModes.BACKTEST: "Бэктестинг",
            TradingModes.REAL: "Реальная торговля"
        }
        return mode_map.get(mode, mode)
    
    # Обработчики команд
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик команды /start."""
        user_id = update.effective_user.id
        
        if not self._is_user_allowed(user_id):
            await update.message.reply_text("⛔ Доступ запрещен.")
            logger.warning(f"Попытка доступа от неавторизованного пользователя: {user_id}")
            return
        
        # Создаем клавиатуру с кнопками
        keyboard = [
            [
                InlineKeyboardButton("Статус", callback_data="status"),
                InlineKeyboardButton("Помощь", callback_data="help"),
            ],
            [
                InlineKeyboardButton("Режимы", callback_data="modes"),
                InlineKeyboardButton("Торговля", callback_data="trade"),
            ],
            [
                InlineKeyboardButton("⏸️ Пауза", callback_data=PAUSE_BOT),
                InlineKeyboardButton("▶️ Продолжить", callback_data=RESUME_BOT),
            ],
            [
                InlineKeyboardButton("🔄 Перезапуск", callback_data=RESTART_BOT),
                InlineKeyboardButton("⛔ Остановка", callback_data=STOP_BOT),
            ],
            [
                InlineKeyboardButton("🧠 Обучить модель", callback_data=TRAIN_MODEL),
                InlineKeyboardButton("⏭️ Пропустить обучение", callback_data=SKIP_TRAINING),
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Отправляем приветственное сообщение с кнопками
        await update.message.reply_text(
            f"👋 Привет, {update.effective_user.first_name}!\n\n"
            "Я Leon Trading Bot. Чем могу помочь?",
            reply_markup=reply_markup
        )
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик команды /help."""
        help_text = self.localization.get("telegram.help_text")
        await update.message.reply_text(
            help_text,
            parse_mode="Markdown"
        )
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик команды /status."""
        if not self.orchestrator:
            await update.message.reply_text(self.localization.get("telegram.no_orchestrator"))
            return
        
        # Получение статуса от оркестратора
        status = await self.orchestrator.get_status()
        
        # Отправка статуса
        await update.message.reply_text(status)
    
    async def cmd_trade(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик команды /trade."""
        keyboard = [
            [
                InlineKeyboardButton("Открыть LONG", callback_data=TelegramCallbacks.OPEN_LONG),
                InlineKeyboardButton("Открыть SHORT", callback_data=TelegramCallbacks.OPEN_SHORT)
            ],
            [
                InlineKeyboardButton("Закрыть все позиции", callback_data=TelegramCallbacks.CLOSE_ALL)
            ],
            [
                InlineKeyboardButton("Назад", callback_data=TelegramCallbacks.BACK_TO_MAIN)
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            self.localization.get("telegram.trade_options"),
            reply_markup=reply_markup
        )
    
    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик команды /balance."""
        if not self.orchestrator:
            await update.message.reply_text(self.localization.get("telegram.no_orchestrator"))
            return
        
        # Получение баланса от оркестратора
        balance = await self.orchestrator.get_balance()
        
        # Отправка баланса
        await update.message.reply_text(f"Текущий баланс: {balance:.2f} USDT")
    
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик команды /positions."""
        if not self.orchestrator:
            await update.message.reply_text(self.localization.get("telegram.no_orchestrator"))
            return
        
        # Получение открытых позиций от оркестратора
        positions = await self.orchestrator.get_positions()
        
        if not positions:
            await update.message.reply_text(self.localization.get("telegram.no_positions"))
            return
        
        # Формирование сообщения с позициями
        message = self.localization.get("telegram.positions_header") + "\n\n"
        
        for pos in positions:
            message += f"Символ: {pos['symbol']}\n"
            message += f"Направление: {pos['direction']}\n"
            message += f"Размер: {pos['size']:.2f}\n"
            message += f"Цена входа: {pos['entry_price']:.8f}\n"
            message += f"Текущая цена: {pos['current_price']:.8f}\n"
            message += f"PnL: {pos['pnl']:.2f} USDT\n\n"
        
        await update.message.reply_text(message)
    
    async def cmd_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик команды /mode."""
        keyboard = [
            [
                InlineKeyboardButton("Симуляция", callback_data=f"{TelegramCallbacks.SET_MODE_PREFIX}{TradingModes.DRY}"),
                InlineKeyboardButton("Бэктестинг", callback_data=f"{TelegramCallbacks.SET_MODE_PREFIX}{TradingModes.BACKTEST}")
            ],
            [
                InlineKeyboardButton("Реальная торговля", callback_data=f"{TelegramCallbacks.SET_MODE_PREFIX}{TradingModes.REAL}")
            ],
            [
                InlineKeyboardButton("Назад", callback_data=TelegramCallbacks.BACK_TO_MAIN)
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            self.localization.get("telegram.select_mode"),
            reply_markup=reply_markup
        )
    
    # Обработчики колбэков
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик колбэков от кнопок."""
        query = update.callback_query
        user_id = query.from_user.id
        
        if not self._is_user_allowed(user_id):
            await query.answer("⛔ Доступ запрещен.")
            logger.warning(f"Попытка доступа от неавторизованного пользователя: {user_id}")
            return
        
        # Получаем данные из кнопки
        callback_data = query.data
        
        # Отвечаем на запрос, чтобы убрать часы загрузки
        await query.answer()
        
        # Обрабатываем различные команды
        if callback_data == "status":
            # Обработка запроса статуса
            await self._handle_status(query)
        elif callback_data == "help":
            # Обработка запроса помощи
            await self._handle_help(query)
        elif callback_data == "modes":
            # Обработка запроса режимов
            await self._handle_modes(query)
        elif callback_data == "trade":
            # Обработка запроса торговли
            await self._handle_trade(query)
        elif callback_data == PAUSE_BOT:
            # Обработка паузы бота
            await self._handle_pause_bot(query)
        elif callback_data == RESUME_BOT:
            # Обработка возобновления работы бота
            await self._handle_resume_bot(query)
        elif callback_data == STOP_BOT:
            # Обработка остановки бота
            await self._handle_stop_bot(query)
        elif callback_data == RESTART_BOT:
            # Обработка перезапуска бота
            await self._handle_restart_bot(query)
        elif callback_data == TRAIN_MODEL:
            # Обработка обучения модели
            await self._handle_train_model(query)
        elif callback_data == SKIP_TRAINING:
            # Обработка пропуска обучения
            await self._handle_skip_training(query)
        else:
            # Неизвестная команда
            await query.edit_message_text(
                text=f"⚠️ Неизвестная команда: {callback_data}"
            )
            logger.warning(f"Получена неизвестная команда: {callback_data}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик текстовых сообщений."""
        text = update.message.text
        
        # Здесь можно добавить обработку текстовых команд
        
        await update.message.reply_text(
            self.localization.get("telegram.unknown_command"),
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("Помощь", callback_data=TelegramCallbacks.HELP)]
            ])
        )
    
    # Вспомогательные методы для обработки колбэков
    
    async def _handle_open_position(self, update: Update, direction: str) -> None:
        """
        Обработка открытия позиции.
        
        Args:
            update: Объект обновления
            direction: Направление сделки
        """
        if not self.orchestrator:
            await update.callback_query.edit_message_text(
                self.localization.get("telegram.no_orchestrator")
            )
            return
        
        try:
            # Открытие позиции через оркестратор
            result = await self.orchestrator.open_position(direction=direction)
            
            if result.get("success"):
                message = self.localization.get("telegram.position_opened_success")
                message += f"\nСимвол: {result.get('symbol')}"
                message += f"\nНаправление: {direction}"
                message += f"\nЦена: {result.get('price')}"
                message += f"\nРазмер: {result.get('size')}"
            else:
                message = self.localization.get("telegram.position_opened_error")
                message += f"\nОшибка: {result.get('error')}"
            
            await update.callback_query.edit_message_text(message)
            
        except Exception as e:
            logger.error(f"Ошибка при открытии позиции: {e}")
            await update.callback_query.edit_message_text(
                self.localization.get("telegram.error_occurred")
            )
    
    async def _handle_close_all_positions(self, update: Update) -> None:
        """
        Обработка закрытия всех позиций.
        
        Args:
            update: Объект обновления
        """
        if not self.orchestrator:
            await update.callback_query.edit_message_text(
                self.localization.get("telegram.no_orchestrator")
            )
            return
        
        try:
            # Закрытие всех позиций через оркестратор
            result = await self.orchestrator.close_all_positions()
            
            if result.get("success"):
                message = self.localization.get("telegram.positions_closed_success")
                message += f"\nЗакрыто позиций: {result.get('count')}"
                message += f"\nОбщий P&L: {result.get('total_pnl'):.2f} USDT"
            else:
                message = self.localization.get("telegram.positions_closed_error")
                message += f"\nОшибка: {result.get('error')}"
            
            await update.callback_query.edit_message_text(message)
            
        except Exception as e:
            logger.error(f"Ошибка при закрытии позиций: {e}")
            await update.callback_query.edit_message_text(
                self.localization.get("telegram.error_occurred")
            )
    
    async def _handle_set_mode(self, update: Update, mode: str) -> None:
        """
        Обработка изменения режима.
        
        Args:
            update: Объект обновления
            mode: Новый режим
        """
        if not self.orchestrator:
            await update.callback_query.edit_message_text(
                self.localization.get("telegram.no_orchestrator")
            )
            return
        
        try:
            # Изменение режима через оркестратор
            result = await self.orchestrator.set_mode(mode)
            
            if result.get("success"):
                message = self.localization.get("telegram.mode_changed_success")
                message += f"\nНовый режим: {self._get_mode_display(mode)}"
            else:
                message = self.localization.get("telegram.mode_changed_error")
                message += f"\nОшибка: {result.get('error')}"
            
            await update.callback_query.edit_message_text(message)
            
        except Exception as e:
            logger.error(f"Ошибка при изменении режима: {e}")
            await update.callback_query.edit_message_text(
                self.localization.get("telegram.error_occurred")
            )
    
    # Новые методы для управления ботом

    async def _handle_pause_bot(self, query: CallbackQuery) -> None:
        """
        Обрабатывает запрос на паузу бота.
        
        Args:
            query: Объект запроса
        """
        if not self.orchestrator:
            await query.edit_message_text(
                text="⚠️ Оркестратор не инициализирован."
            )
            logger.warning("Попытка приостановить бота без инициализированного оркестратора")
            return
        
        try:
            # Приостанавливаем работу бота
            success = await self.orchestrator.pause()
            
            if success:
                await query.edit_message_text(
                    text="⏸️ Бот приостановлен. Используйте кнопку 'Продолжить' для возобновления работы."
                )
                logger.info("Бот приостановлен через Telegram")
            else:
                await query.edit_message_text(
                    text="⚠️ Не удалось приостановить бота. Возможно, он не запущен или уже приостановлен."
                )
                logger.warning("Не удалось приостановить бота через Telegram")
        except Exception as e:
            await query.edit_message_text(
                text=f"❌ Ошибка при приостановке бота: {str(e)}"
            )
            logger.error(f"Ошибка при приостановке бота через Telegram: {str(e)}")
            logger.debug(traceback.format_exc())
    
    async def _handle_resume_bot(self, query: CallbackQuery) -> None:
        """
        Обрабатывает запрос на возобновление работы бота.
        
        Args:
            query: Объект запроса
        """
        if not self.orchestrator:
            await query.edit_message_text(
                text="⚠️ Оркестратор не инициализирован."
            )
            logger.warning("Попытка возобновить работу бота без инициализированного оркестратора")
            return
        
        try:
            # Возобновляем работу бота
            success = await self.orchestrator.resume()
            
            if success:
                await query.edit_message_text(
                    text="▶️ Работа бота возобновлена."
                )
                logger.info("Работа бота возобновлена через Telegram")
            else:
                await query.edit_message_text(
                    text="⚠️ Не удалось возобновить работу бота. Возможно, он не запущен или не приостановлен."
                )
                logger.warning("Не удалось возобновить работу бота через Telegram")
        except Exception as e:
            await query.edit_message_text(
                text=f"❌ Ошибка при возобновлении работы бота: {str(e)}"
            )
            logger.error(f"Ошибка при возобновлении работы бота через Telegram: {str(e)}")
            logger.debug(traceback.format_exc())
    
    async def _handle_stop_bot(self, query: CallbackQuery) -> None:
        """
        Обрабатывает запрос на остановку бота.
        
        Args:
            query: Объект запроса
        """
        if not self.orchestrator:
            await query.edit_message_text(
                text="⚠️ Оркестратор не инициализирован."
            )
            logger.warning("Попытка остановить бота без инициализированного оркестратора")
            return
        
        try:
            # Запрашиваем подтверждение
            keyboard = [
                [
                    InlineKeyboardButton("✅ Да, остановить", callback_data="confirm_stop"),
                    InlineKeyboardButton("❌ Нет, отмена", callback_data="cancel_stop"),
                ]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                text="⚠️ Вы уверены, что хотите остановить бота? Все текущие операции будут прерваны.",
                reply_markup=reply_markup
            )
            logger.info("Запрос на подтверждение остановки бота через Telegram")
        except Exception as e:
            await query.edit_message_text(
                text=f"❌ Ошибка при запросе остановки бота: {str(e)}"
            )
            logger.error(f"Ошибка при запросе остановки бота через Telegram: {str(e)}")
            logger.debug(traceback.format_exc())
    
    async def _handle_restart_bot(self, query: CallbackQuery) -> None:
        """
        Обрабатывает запрос на перезапуск бота.
        
        Args:
            query: Объект запроса
        """
        if not self.orchestrator:
            await query.edit_message_text(
                text="⚠️ Оркестратор не инициализирован."
            )
            logger.warning("Попытка перезапустить бота без инициализированного оркестратора")
            return
        
        try:
            # Запрашиваем подтверждение
            keyboard = [
                [
                    InlineKeyboardButton("✅ Да, перезапустить", callback_data="confirm_restart"),
                    InlineKeyboardButton("❌ Нет, отмена", callback_data="cancel_restart"),
                ]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                text="⚠️ Вы уверены, что хотите перезапустить бота? Все текущие операции будут прерваны.",
                reply_markup=reply_markup
            )
            logger.info("Запрос на подтверждение перезапуска бота через Telegram")
        except Exception as e:
            await query.edit_message_text(
                text=f"❌ Ошибка при запросе перезапуска бота: {str(e)}"
            )
            logger.error(f"Ошибка при запросе перезапуска бота через Telegram: {str(e)}")
            logger.debug(traceback.format_exc())
    
    async def _handle_train_model(self, query: CallbackQuery) -> None:
        """
        Обрабатывает запрос на обучение модели.
        
        Args:
            query: Объект запроса
        """
        if not self.orchestrator:
            await query.edit_message_text(
                text="⚠️ Оркестратор не инициализирован."
            )
            logger.warning("Попытка обучить модель без инициализированного оркестратора")
            return
        
        try:
            # Сообщаем о начале обучения
            await query.edit_message_text(
                text="🧠 Начинаю обучение модели. Это может занять некоторое время..."
            )
            logger.info("Запуск обучения модели через Telegram")
            
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
                message = (
                    f"✅ Обучение модели завершено успешно!\n\n"
                    f"📊 Метрики:\n"
                    f"- Точность (accuracy): {accuracy:.4f}\n"
                    f"- Precision: {precision:.4f}\n"
                    f"- Recall: {recall:.4f}\n"
                    f"- F1-score: {f1_score:.4f}\n\n"
                    f"Модель готова к использованию."
                )
                
                await query.edit_message_text(text=message)
                logger.info(f"Обучение модели через Telegram завершено успешно: {metrics}")
            else:
                error = result.get("error", "Неизвестная ошибка")
                await query.edit_message_text(
                    text=f"❌ Ошибка при обучении модели: {error}"
                )
                logger.error(f"Ошибка при обучении модели через Telegram: {error}")
        except Exception as e:
            await query.edit_message_text(
                text=f"❌ Ошибка при обучении модели: {str(e)}"
            )
            logger.error(f"Ошибка при обучении модели через Telegram: {str(e)}")
            logger.debug(traceback.format_exc())
    
    async def _handle_skip_training(self, query: CallbackQuery) -> None:
        """
        Обработчик пропуска обучения модели.
        
        Args:
            query: Объект запроса
        """
        if not self.orchestrator:
            await query.edit_message_text(
                text="⏭️ Обучение модели пропущено. Будет использована последняя сохраненная модель."
            )
            logger.info("Обучение модели пропущено через Telegram")
            return
            
        try:
            # Здесь можно добавить логику для загрузки последней сохраненной модели
            await query.edit_message_text(
                text="⏭️ Обучение модели пропущено. Будет использована последняя сохраненная модель."
            )
            logger.info("Обучение модели пропущено через Telegram")
        except Exception as e:
            await query.edit_message_text(
                text=f"❌ Ошибка при пропуске обучения модели: {str(e)}"
            )
            logger.error(f"Ошибка при пропуске обучения модели через Telegram: {str(e)}")
            logger.debug(traceback.format_exc())

    def _is_user_allowed(self, user_id: int) -> bool:
        """Проверяет, разрешен ли доступ пользователю."""
        return user_id in self.allowed_users

    async def _handle_status(self, query: CallbackQuery) -> None:
        """Обработчик запроса статуса."""
        if not self.orchestrator:
            await query.edit_message_text(self.localization.get("telegram.no_orchestrator"))
            return
        
        # Получение статуса от оркестратора
        status = await self.orchestrator.get_status()
        
        # Отправка статуса
        await query.edit_message_text(status)
    
    async def _handle_help(self, query: CallbackQuery) -> None:
        """Обработчик запроса помощи."""
        help_text = self.localization.get("telegram.help_text")
        await query.edit_message_text(help_text)
    
    async def _handle_modes(self, query: CallbackQuery) -> None:
        """Обработчик запроса режимов."""
        keyboard = [
            [
                InlineKeyboardButton("Симуляция", callback_data=f"{TelegramCallbacks.SET_MODE_PREFIX}{TradingModes.DRY}"),
                InlineKeyboardButton("Бэктестинг", callback_data=f"{TelegramCallbacks.SET_MODE_PREFIX}{TradingModes.BACKTEST}")
            ],
            [
                InlineKeyboardButton("Реальная торговля", callback_data=f"{TelegramCallbacks.SET_MODE_PREFIX}{TradingModes.REAL}")
            ],
            [
                InlineKeyboardButton("Назад", callback_data=TelegramCallbacks.BACK_TO_MAIN)
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            self.localization.get("telegram.select_mode"),
            reply_markup=reply_markup
        )
    
    async def _handle_trade(self, query: CallbackQuery) -> None:
        """Обработчик запроса торговли."""
        keyboard = [
            [
                InlineKeyboardButton("Открыть LONG", callback_data=TelegramCallbacks.OPEN_LONG),
                InlineKeyboardButton("Открыть SHORT", callback_data=TelegramCallbacks.OPEN_SHORT)
            ],
            [
                InlineKeyboardButton("Закрыть все позиции", callback_data=TelegramCallbacks.CLOSE_ALL)
            ],
            [
                InlineKeyboardButton("Назад", callback_data=TelegramCallbacks.BACK_TO_MAIN)
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            self.localization.get("telegram.trade_options"),
            reply_markup=reply_markup
        ) 