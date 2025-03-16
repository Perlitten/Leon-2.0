"""
Модуль для интеграции с Telegram API.

Предоставляет функциональность для отправки уведомлений через Telegram.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
import traceback

try:
    import telegram
    from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

from core.localization import LocalizationManager

logger = logging.getLogger(__name__)

class TelegramIntegration:
    """
    Класс для интеграции с Telegram API.
    
    Позволяет отправлять уведомления через Telegram бота.
    """
    
    def __init__(self, token: str, chat_id: str, config: Dict[str, Any] = None, localization: LocalizationManager = None):
        """
        Инициализация интеграции с Telegram.
        
        Args:
            token: Токен Telegram бота
            chat_id: ID чата для отправки сообщений
            config: Дополнительная конфигурация
            localization: Менеджер локализации
        """
        self.logger = logging.getLogger("TelegramIntegration")
        
        if not TELEGRAM_AVAILABLE:
            self.logger.error("Библиотека python-telegram-bot не установлена")
            self.available = False
            return
            
        self.token = token
        self.chat_id = chat_id
        self.config = config or {}
        self.bot = None
        self.connected = False
        self.available = True
        self.localization = localization
        
        # Инициализация бота
        try:
            self.bot = telegram.Bot(token=self.token)
            self.connected = True
            self.logger.info("Telegram бот инициализирован")
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации Telegram бота: {str(e)}")
            self.connected = False
            
    async def _async_check_connection(self) -> bool:
        """
        Асинхронная проверка подключения к Telegram API.
        
        Returns:
            bool: True, если подключение установлено, иначе False
        """
        try:
            await self.bot.get_me()
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при проверке подключения к Telegram API: {str(e)}")
            return False
            
    async def ensure_connected(self) -> bool:
        """
        Обеспечивает подключение к Telegram API.
        
        Returns:
            bool: True, если подключение установлено, иначе False
        """
        if self.connected:
            return True
            
        try:
            await self._async_check_connection()
            self.connected = True
            self.logger.info("Подключение к Telegram API восстановлено")
            return True
        except Exception as e:
            self.logger.error(f"Не удалось подключиться к Telegram API: {e}")
            return False
            
    async def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """
        Отправляет сообщение в Telegram.
        
        Args:
            text: Текст сообщения
            parse_mode: Режим форматирования текста
            
        Returns:
            bool: True, если сообщение успешно отправлено, иначе False
        """
        # Проверяем подключение
        if not await self.ensure_connected():
            return False
            
        # Разбиваем длинные сообщения
        max_length = 4096
        if len(text) > max_length:
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            success = True
            for chunk in chunks:
                if not await self._send_message_internal(chunk, parse_mode):
                    success = False
            return success
        else:
            return await self._send_message_internal(text, parse_mode)
            
    async def _send_message_internal(self, text: str, parse_mode: str) -> bool:
        """
        Внутренний метод для отправки сообщения в Telegram.
        
        Args:
            text: Текст сообщения
            parse_mode: Режим форматирования текста
            
        Returns:
            bool: True, если сообщение успешно отправлено, иначе False
        """
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode
            )
            return True
        except telegram.error.NetworkError as e:
            self.logger.error(f"Ошибка сети при отправке сообщения: {e}")
            self.connected = False
            return False
        except telegram.error.Unauthorized as e:
            self.logger.error(f"Ошибка авторизации: {e}")
            # Требуется перенастройка бота
            return False
        except Exception as e:
            self.logger.error(f"Неизвестная ошибка при отправке сообщения: {e}")
            self.connected = False
            return False
    
    async def send_trade_notification(self, symbol: str, direction: str, 
                                     price: float, size: float, 
                                     pnl: Optional[float] = None) -> bool:
        """
        Отправляет уведомление о торговой операции.
        
        Args:
            symbol: Символ торговой пары
            direction: Направление сделки (BUY/SELL)
            price: Цена исполнения
            size: Размер позиции
            pnl: Прибыль/убыток (опционально)
            
        Returns:
            bool: True, если сообщение успешно отправлено, иначе False
        """
        emoji = "🟢" if direction == "BUY" else "🔴"
        
        if self.localization:
            # Используем локализованные тексты
            direction_text = self.localization.get_text("trading.directions.buy") if direction == "BUY" else self.localization.get_text("trading.directions.sell")
            price_text = self.localization.get_text("trading.price")
            volume_text = self.localization.get_text("trading.volume")
            pnl_text = self.localization.get_text("trading.pnl")
            
            # Добавляем цитату, если она есть в локализации
            quote = self.localization.get_text("notifications.telegram.quote_open", default="")
            
            message = f"{emoji} *{direction_text}* {symbol}\n"
            message += f"💰 {price_text}: {price}\n"
            message += f"📊 {volume_text}: {size}\n"
            
            if pnl is not None:
                pnl_emoji = "✅" if pnl >= 0 else "❌"
                message += f"{pnl_emoji} {pnl_text}: {pnl:.2f}%\n"
            
            if quote:
                message += f"\n💬 _{quote}_"
        else:
            # Используем стандартные тексты
            message = f"{emoji} *{direction}* {symbol}\n"
            message += f"💰 Цена: {price}\n"
            message += f"📊 Объем: {size}\n"
            
            if pnl is not None:
                pnl_emoji = "✅" if pnl >= 0 else "❌"
                message += f"{pnl_emoji} P&L: {pnl:.2f}%\n"
        
        return await self.send_message(message)
    
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
        mode_emoji = "🧪" if mode == "dry" else "🔥"
        
        if self.localization:
            # Используем локализованные тексты
            status_title = self.localization.get_text("notifications.telegram.status_update")
            mode_text = self.localization.get_text("notifications.telegram.mode")
            symbol_text = self.localization.get_text("notifications.telegram.symbol")
            balance_text = self.localization.get_text("notifications.telegram.balance")
            leverage_text = self.localization.get_text("notifications.telegram.leverage")
            risk_text = self.localization.get_text("notifications.telegram.risk_per_trade")
            sl_text = self.localization.get_text("notifications.telegram.stop_loss")
            tp_text = self.localization.get_text("notifications.telegram.take_profit")
            
            # Получаем локализованное название режима
            mode_name = self.localization.get_text(f"trading.modes.{mode}", default=mode.upper())
            
            message = f"📊 *{status_title}*\n\n"
            message += f"{mode_emoji} {mode_text}: {mode_name}\n"
            message += f"💱 {symbol_text}: {symbol}\n"
            message += f"💰 {balance_text}: {balance:.2f} USDT\n"
            message += f"⚡ {leverage_text}: {leverage}x\n"
            message += f"⚠️ {risk_text}: {risk_per_trade}%\n"
            message += f"🛑 {sl_text}: {stop_loss}%\n"
            message += f"🎯 {tp_text}: {take_profit}%\n"
        else:
            # Используем стандартные тексты
            message = f"📊 *Статус торгового бота*\n\n"
            message += f"{mode_emoji} Режим: {mode.upper()}\n"
            message += f"💱 Пара: {symbol}\n"
            message += f"💰 Баланс: {balance:.2f} USDT\n"
            message += f"⚡ Плечо: {leverage}x\n"
            message += f"⚠️ Риск: {risk_per_trade}%\n"
            message += f"🛑 SL: {stop_loss}%\n"
            message += f"🎯 TP: {take_profit}%\n"
        
        return await self.send_message(message)
    
    async def send_message_direct(self, text: str, parse_mode: str = "Markdown") -> bool:
        """
        Отправляет сообщение в Telegram напрямую, без проверки подключения.
        Используется для отладки.
        
        Args:
            text: Текст сообщения
            parse_mode: Режим форматирования текста (Markdown, HTML)
            
        Returns:
            bool: True, если сообщение успешно отправлено, иначе False
        """
        try:
            # Создаем новый экземпляр бота для прямой отправки
            bot = telegram.Bot(token=self.token)
            
            # Отправляем сообщение
            await bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode
            )
            logger.info(f"[DIRECT] Отправлено сообщение в Telegram: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"[DIRECT] Ошибка при отправке сообщения в Telegram: {str(e)}")
            logger.debug(traceback.format_exc())
            return False 