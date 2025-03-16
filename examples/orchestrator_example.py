"""
Пример использования оркестратора Leon Trading Bot.
"""

import logging
import time
import sys
import os

# Добавление корневой директории проекта в путь для импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.orchestrator import LeonOrchestrator
from core.constants import TRADING_MODES


def setup_logging():
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/orchestrator_example.log')
        ]
    )


def main():
    """Основная функция примера."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Запуск примера использования оркестратора")
    
    # Создание экземпляра оркестратора
    orchestrator = LeonOrchestrator(dry_mode=True)
    
    try:
        # Запуск оркестратора
        logger.info("Запуск оркестратора")
        orchestrator.start()
        
        # Получение статуса оркестратора
        status_result = orchestrator.process_command("status")
        logger.info(f"Статус оркестратора: {status_result}")
        
        # Переключение режима работы на бэктестинг
        logger.info("Переключение режима работы на бэктестинг")
        mode_result = orchestrator.process_command("set_mode", [TRADING_MODES["BACKTEST"]])
        logger.info(f"Результат переключения режима: {mode_result}")
        
        # Получение обновленного статуса оркестратора
        status_result = orchestrator.process_command("status")
        logger.info(f"Обновленный статус оркестратора: {status_result}")
        
        # Установка активной модели
        logger.info("Установка активной модели")
        model_result = orchestrator.process_command("set_model", ["lstm"])
        logger.info(f"Результат установки модели: {model_result}")
        
        # Имитация работы системы
        logger.info("Имитация работы системы в течение 5 секунд")
        time.sleep(5)
        
        # Остановка оркестратора
        logger.info("Остановка оркестратора")
        orchestrator.stop()
        
        logger.info("Пример успешно завершен")
    except Exception as e:
        logger.error(f"Ошибка при выполнении примера: {e}")
        orchestrator.stop()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 