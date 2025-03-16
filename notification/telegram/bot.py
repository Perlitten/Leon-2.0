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

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    CallbackQueryHandler,
    ContextTypes,
    filters
)

from core.config_manager import ConfigManager
from core.constants import TradingModes, TradeDirections, TelegramCommands, TelegramCallbacks
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
    
    async def send_message(self, text: str, reply_markup: Optional[InlineKeyboardMarkup] = None) -> None:
        """
        Отправка сообщения в чат.
        
        Args:
            text: Текст сообщения
            reply_markup: Разметка для кнопок (опционально)
        """
        if not self.is_running:
            logger.warning("Попытка отправить сообщение, когда бот не запущен")
            return
        
        try:
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="Markdown",
                reply_markup=reply_markup
            )
            logger.debug(f"Отправлено сообщение: {text[:50]}...")
        except Exception as e:
            logger.error(f"Ошибка при отправке сообщения: {e}")
    
    async def send_trade_notification(self, 
                                     symbol: str, 
                                     direction: str, 
                                     price: float, 
                                     size: float, 
                                     pnl: Optional[float] = None) -> None:
        """
        Отправка уведомления о торговой операции.
        
        Args:
            symbol: Символ торговой пары
            direction: Направление сделки (BUY/SELL)
            price: Цена
            size: Размер позиции
            pnl: Прибыль/убыток (опционально)
        """
        # Определение типа операции
        is_open = direction == TradeDirections.BUY
        
        # Формирование сообщения
        if is_open:
            title = self.localization.get("telegram.position_opened")
            emoji = "🟢"
        else:
            title = self.localization.get("telegram.position_closed")
            emoji = "🔴"
        
        # Получение текущего режима
        mode = self.config.get("TRADING_MODE", TradingModes.DRY)
        
        # Формирование сообщения
        message = f"StableTrade\n"
        
        if mode == TradingModes.DRY:
            message += f"🚀 *Симуляция запущена*\n\n"
        
        message += f"{emoji} *{title}*\n"
        message += f"◆ Пара: *{symbol}*\n"
        message += f"◆ Цена: *{price}*\n"
        message += f"◆ Размер: *{size}*\n"
        
        if pnl is not None:
            message += f"💰 P&L: *{pnl:.2f} USDT*\n"
        
        # Добавление временной метки
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message += f"🕒 *{now}*\n"
        
        # Добавление мотивационной цитаты
        if is_open:
            quote = self.localization.get("telegram.quote_open")
            message += f"\n💬 _{quote}_"
        else:
            quote = self.localization.get("telegram.quote_close")
            message += f"\n💬 _{quote}_"
        
        await self.send_message(message)
    
    async def send_status_update(self, 
                                symbol: str, 
                                mode: str, 
                                balance: float, 
                                leverage: int, 
                                risk_per_trade: float,
                                stop_loss: float,
                                take_profit: float) -> None:
        """
        Отправка обновления статуса бота.
        
        Args:
            symbol: Символ торговой пары
            mode: Режим работы
            balance: Баланс
            leverage: Плечо
            risk_per_trade: Риск на сделку
            stop_loss: Стоп-лосс
            take_profit: Тейк-профит
        """
        # Формирование сообщения
        message = f"StableTrade\n"
        
        if mode == TradingModes.DRY:
            message += f"🚀 *Симуляция запущена*\n\n"
        elif mode == TradingModes.BACKTEST:
            message += f"📊 *Бэктестинг запущен*\n\n"
        else:
            message += f"💹 *Реальная торговля запущена*\n\n"
        
        message += f"◆ Символ: *{symbol}*\n"
        message += f"◆ Режим: *{self._get_mode_display(mode)}*\n"
        message += f"◆ Начальный баланс: *{balance:.1f} USDT*\n"
        message += f"◆ Плечо: *{leverage}x*\n"
        message += f"◆ Риск на сделку: *{risk_per_trade:.1f}%*\n"
        message += f"◆ Стоп-лосс: *{stop_loss:.1f}%*\n"
        message += f"◆ Тейк-профит: *{take_profit:.1f}%*\n"
        
        # Добавление временной метки
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message += f"🕒 *{now}*"
        
        await self.send_message(message)
    
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
        keyboard = [
            [
                InlineKeyboardButton("Статус", callback_data=TelegramCallbacks.STATUS),
                InlineKeyboardButton("Помощь", callback_data=TelegramCallbacks.HELP)
            ],
            [
                InlineKeyboardButton("Режимы", callback_data=TelegramCallbacks.MODES),
                InlineKeyboardButton("Торговля", callback_data=TelegramCallbacks.TRADE)
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            self.localization.get("telegram.welcome"),
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
        await query.answer()
        
        callback_data = query.data
        
        # Обработка колбэков для основного меню
        if callback_data == TelegramCallbacks.STATUS:
            await self.cmd_status(update, context)
        elif callback_data == TelegramCallbacks.HELP:
            await self.cmd_help(update, context)
        elif callback_data == TelegramCallbacks.MODES:
            await self.cmd_mode(update, context)
        elif callback_data == TelegramCallbacks.TRADE:
            await self.cmd_trade(update, context)
        elif callback_data == TelegramCallbacks.BACK_TO_MAIN:
            await self.cmd_start(update, context)
        
        # Обработка колбэков для торговли
        elif callback_data == TelegramCallbacks.OPEN_LONG:
            await self._handle_open_position(update, TradeDirections.BUY)
        elif callback_data == TelegramCallbacks.OPEN_SHORT:
            await self._handle_open_position(update, TradeDirections.SELL)
        elif callback_data == TelegramCallbacks.CLOSE_ALL:
            await self._handle_close_all_positions(update)
        
        # Обработка колбэков для режимов
        elif callback_data.startswith(TelegramCallbacks.SET_MODE_PREFIX):
            mode = callback_data.replace(TelegramCallbacks.SET_MODE_PREFIX, "")
            await self._handle_set_mode(update, mode)
        
        # Обработка неизвестных колбэков
        else:
            logger.warning(f"Неизвестный колбэк: {callback_data}")
            await query.edit_message_text(self.localization.get("telegram.unknown_callback"))
    
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