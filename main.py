#!/usr/bin/env python
"""
Основной файл для запуска Leon Trading Bot.

Точка входа в приложение.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any
import asyncio
from pathlib import Path

from core.config_manager import ConfigManager
from core.localization import LocalizationManager
from core.orchestrator import LeonOrchestrator
from core.constants import TradingModes


def setup_logging(log_level: str = "INFO") -> None:
    """
    Настройка логирования.
    
    Args:
        log_level: Уровень логирования
    """
    # Создание директории для логов, если она не существует
    os.makedirs("logs", exist_ok=True)
    
    # Настройка логирования
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/leon.log"),
            logging.StreamHandler()
        ]
    )


def parse_args() -> Dict[str, Any]:
    """
    Парсинг аргументов командной строки.
    
    Returns:
        Словарь с аргументами командной строки
    """
    parser = argparse.ArgumentParser(description="Leon Trading Bot")
    
    parser.add_argument(
        "--mode",
        choices=[TradingModes.DRY, TradingModes.BACKTEST, TradingModes.REAL],
        default=TradingModes.DRY,
        help="Режим работы бота"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Уровень логирования"
    )
    
    return vars(parser.parse_args())


async def main():
    """Основная функция запуска бота."""
    # Парсинг аргументов командной строки
    args = parse_args()
    
    # Настройка логирования
    setup_logging(args["log_level"])
    
    # Получение логгера
    logger = logging.getLogger(__name__)
    
    logger.info("Запуск Leon Trading Bot")
    
    orchestrator = None
    
    try:
        # Инициализация менеджера конфигурации
        config_manager = ConfigManager()
        
        # Инициализация менеджера локализации
        localization = LocalizationManager()
        
        # Инициализация оркестратора
        orchestrator = LeonOrchestrator(config_manager, localization)
        
        # Запуск системы
        await orchestrator.start()
        
        # Установка режима работы
        if args["mode"]:
            await orchestrator.set_mode(args["mode"])
        
        # Ожидание завершения работы
        logger.info("Бот запущен. Нажмите Ctrl+C для завершения...")
        
        # Бесконечный цикл для поддержания работы бота
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
        if orchestrator:
            await orchestrator.stop()
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}")
        if orchestrator:
            await orchestrator.stop()


if __name__ == "__main__":
    # Запуск основной функции
    asyncio.run(main()) 