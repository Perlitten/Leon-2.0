"""
Пример использования Telegram бота.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Добавляем корневую директорию проекта в путь импорта
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config_manager import ConfigManager
from core.localization import LocalizationManager
from notification.telegram.bot import TelegramBot

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class MockOrchestrator:
    """Мок-класс оркестратора для тестирования."""
    
    async def get_status(self):
        """Возвращает статус бота."""
        return "Бот работает в режиме симуляции"
    
    async def get_balance(self):
        """Возвращает текущий баланс."""
        return 1000.0
    
    async def get_positions(self):
        """Возвращает открытые позиции."""
        return [
            {
                "symbol": "BTCUSDT",
                "direction": "BUY",
                "size": 0.1,
                "entry_price": 50000.0,
                "current_price": 51000.0,
                "pnl": 100.0
            }
        ]
    
    async def open_position(self, direction):
        """Открывает позицию."""
        return {
            "success": True,
            "symbol": "BTCUSDT",
            "direction": direction,
            "price": 50000.0,
            "size": 0.1
        }
    
    async def close_all_positions(self):
        """Закрывает все позиции."""
        return {
            "success": True,
            "count": 1,
            "total_pnl": 100.0
        }
    
    async def set_mode(self, mode):
        """Изменяет режим работы."""
        return {
            "success": True,
            "mode": mode
        }

async def main():
    """Основная функция примера."""
    try:
        # Инициализация менеджера конфигурации
        config_manager = ConfigManager()
        
        # Инициализация менеджера локализации
        localization = LocalizationManager()
        
        # Инициализация Telegram бота
        telegram_bot = TelegramBot(config_manager, localization)
        
        # Создание мок-оркестратора
        orchestrator = MockOrchestrator()
        
        # Установка ссылки на оркестратор
        telegram_bot.set_orchestrator(orchestrator)
        
        # Запуск бота
        await telegram_bot.start()
        
        # Отправка статусного сообщения
        await telegram_bot.send_status_update(
            symbol="BTCUSDT",
            mode="dry",
            balance=1000.0,
            leverage=20,
            risk_per_trade=2.0,
            stop_loss=2.0,
            take_profit=3.0
        )
        
        # Ожидание 5 секунд
        logger.info("Бот запущен. Ожидание 5 секунд...")
        await asyncio.sleep(5)
        
        # Отправка уведомления о торговой операции
        await telegram_bot.send_trade_notification(
            symbol="BTCUSDT",
            direction="BUY",
            price=50000.0,
            size=0.1
        )
        
        # Ожидание 60 секунд для взаимодействия с ботом
        logger.info("Бот готов к взаимодействию. Нажмите Ctrl+C для завершения...")
        await asyncio.sleep(60)
        
        # Остановка бота
        await telegram_bot.stop()
        
    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
        if telegram_bot and telegram_bot.is_running:
            await telegram_bot.stop()
    except Exception as e:
        logger.error(f"Ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 