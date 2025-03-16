"""
Модульные тесты для модуля оркестратора.
"""

import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from core.orchestrator import (
    LeonOrchestrator, TradingMode, TradingModeManager, 
    MLIntegrationManager, EventBus, CommandProcessor
)


class TestLeonOrchestrator(unittest.TestCase):
    """Тесты для класса LeonOrchestrator."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        # Создаем моки для зависимостей
        self.config_manager_mock = MagicMock()
        self.config_manager_mock.get_config.return_value = {
            "trading": {"mode": "dry"},
            "localization": {"locales_dir": "locales", "default_language": "ru"}
        }
        
        # Патчим зависимости
        self.config_manager_patcher = patch('core.orchestrator.ConfigManager', return_value=self.config_manager_mock)
        self.localization_patcher = patch('core.orchestrator.LocalizationManager')
        
        # Запускаем патчи
        self.config_manager_mock = self.config_manager_patcher.start()
        self.localization_mock = self.localization_patcher.start()
        
        # Создаем экземпляр оркестратора
        self.orchestrator = LeonOrchestrator()
        
        # Заменяем методы на моки
        self.orchestrator._init_notification_service = AsyncMock()
        self.orchestrator._init_exchange_client = AsyncMock()
        self.orchestrator._init_risk_controller = AsyncMock()
        self.orchestrator._init_strategy = AsyncMock()
        self.orchestrator._init_trader = AsyncMock()
        self.orchestrator._init_visualizer = AsyncMock()
        self.orchestrator._init_ml_manager = AsyncMock()
    
    def tearDown(self):
        """Очистка после тестов."""
        # Останавливаем патчи
        self.config_manager_patcher.stop()
        self.localization_patcher.stop()
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Тест инициализации оркестратора."""
        # Вызываем метод initialize
        result = await self.orchestrator.initialize()
        
        # Проверяем, что метод вернул True
        self.assertTrue(result)
        
        # Проверяем, что все методы инициализации были вызваны
        self.orchestrator._init_notification_service.assert_called_once()
        self.orchestrator._init_exchange_client.assert_called_once()
        self.orchestrator._init_risk_controller.assert_called_once()
        self.orchestrator._init_strategy.assert_called_once()
        self.orchestrator._init_trader.assert_called_once()
        self.orchestrator._init_visualizer.assert_called_once()
        self.orchestrator._init_ml_manager.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start(self):
        """Тест запуска оркестратора."""
        # Мокаем методы
        self.orchestrator.is_running = False
        self.orchestrator.trader = MagicMock()
        self.orchestrator.trader.start = AsyncMock()
        self.orchestrator._start_background_tasks = MagicMock()
        self.orchestrator.notification_service = MagicMock()
        self.orchestrator.notification_service.send_notification = AsyncMock()
        
        # Вызываем метод start
        result = await self.orchestrator.start()
        
        # Проверяем, что метод вернул True
        self.assertTrue(result)
        
        # Проверяем, что флаг is_running установлен в True
        self.assertTrue(self.orchestrator.is_running)
        
        # Проверяем, что методы были вызваны
        self.orchestrator.trader.start.assert_called_once()
        self.orchestrator._start_background_tasks.assert_called_once()
        self.orchestrator.notification_service.send_notification.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop(self):
        """Тест остановки оркестратора."""
        # Мокаем методы
        self.orchestrator.is_running = True
        self.orchestrator.trader = MagicMock()
        self.orchestrator.trader.stop = AsyncMock()
        self.orchestrator._cancel_background_tasks = MagicMock()
        self.orchestrator.notification_service = MagicMock()
        self.orchestrator.notification_service.send_notification = AsyncMock()
        
        # Вызываем метод stop
        result = await self.orchestrator.stop()
        
        # Проверяем, что метод вернул True
        self.assertTrue(result)
        
        # Проверяем, что флаг is_running установлен в False
        self.assertFalse(self.orchestrator.is_running)
        
        # Проверяем, что методы были вызваны
        self.orchestrator.trader.stop.assert_called_once()
        self.orchestrator._cancel_background_tasks.assert_called_once()
        self.orchestrator.notification_service.send_notification.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_switch_mode(self):
        """Тест переключения режима работы."""
        # Мокаем методы
        self.orchestrator.trading_mode = TradingMode.DRY
        self.orchestrator.is_running = True
        self.orchestrator.stop = AsyncMock()
        self.orchestrator._init_trader = AsyncMock()
        self.orchestrator.start = AsyncMock()
        self.orchestrator.config_manager = MagicMock()
        self.orchestrator.config_manager.update_config = MagicMock()
        
        # Вызываем метод switch_mode
        result = await self.orchestrator.switch_mode(TradingMode.BACKTEST)
        
        # Проверяем, что метод вернул True
        self.assertTrue(result)
        
        # Проверяем, что режим изменился
        self.assertEqual(self.orchestrator.trading_mode, TradingMode.BACKTEST)
        
        # Проверяем, что методы были вызваны
        self.orchestrator.stop.assert_called_once()
        self.orchestrator._init_trader.assert_called_once()
        self.orchestrator.start.assert_called_once()
        self.orchestrator.config_manager.update_config.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_command(self):
        """Тест обработки команды."""
        # Мокаем методы
        self.orchestrator.command_handlers = {
            "test_command": AsyncMock(return_value={"success": True, "message": "Test command executed"})
        }
        
        # Вызываем метод process_command
        result = await self.orchestrator.process_command("test_command", ["arg1", "arg2"])
        
        # Проверяем результат
        self.assertEqual(result, {"success": True, "message": "Test command executed"})
        
        # Проверяем, что обработчик команды был вызван
        self.orchestrator.command_handlers["test_command"].assert_called_once_with(["arg1", "arg2"])
    
    @pytest.mark.asyncio
    async def test_process_command_unknown(self):
        """Тест обработки неизвестной команды."""
        # Мокаем методы
        self.orchestrator.command_handlers = {}
        self.orchestrator.localization = MagicMock()
        self.orchestrator.localization.get_text = MagicMock(return_value="Unknown command: {command}")
        
        # Вызываем метод process_command
        result = await self.orchestrator.process_command("unknown_command")
        
        # Проверяем результат
        self.assertEqual(result, {"success": False, "message": "Unknown command: {command}"})


class TestTradingModeManager(unittest.TestCase):
    """Тесты для класса TradingModeManager."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        # Создаем мок для оркестратора
        self.orchestrator_mock = MagicMock()
        
        # Создаем экземпляр менеджера режимов торговли
        self.trading_mode_manager = TradingModeManager(self.orchestrator_mock)
    
    @pytest.mark.asyncio
    async def test_switch_to_dry_mode(self):
        """Тест переключения в режим симуляции."""
        # Мокаем метод switch_mode оркестратора
        self.orchestrator_mock.switch_mode = AsyncMock(return_value=True)
        
        # Вызываем метод switch_to_dry_mode
        result = await self.trading_mode_manager.switch_to_dry_mode()
        
        # Проверяем, что метод вернул True
        self.assertTrue(result)
        
        # Проверяем, что метод switch_mode оркестратора был вызван с правильным аргументом
        self.orchestrator_mock.switch_mode.assert_called_once_with(TradingMode.DRY)
    
    @pytest.mark.asyncio
    async def test_switch_to_backtest_mode(self):
        """Тест переключения в режим бэктестинга."""
        # Мокаем метод switch_mode оркестратора
        self.orchestrator_mock.switch_mode = AsyncMock(return_value=True)
        
        # Вызываем метод switch_to_backtest_mode
        result = await self.trading_mode_manager.switch_to_backtest_mode()
        
        # Проверяем, что метод вернул True
        self.assertTrue(result)
        
        # Проверяем, что метод switch_mode оркестратора был вызван с правильным аргументом
        self.orchestrator_mock.switch_mode.assert_called_once_with(TradingMode.BACKTEST)
    
    @pytest.mark.asyncio
    async def test_switch_to_real_mode(self):
        """Тест переключения в режим реальной торговли."""
        # Мокаем метод switch_mode оркестратора
        self.orchestrator_mock.switch_mode = AsyncMock(return_value=True)
        
        # Вызываем метод switch_to_real_mode
        result = await self.trading_mode_manager.switch_to_real_mode()
        
        # Проверяем, что метод вернул True
        self.assertTrue(result)
        
        # Проверяем, что метод switch_mode оркестратора был вызван с правильным аргументом
        self.orchestrator_mock.switch_mode.assert_called_once_with(TradingMode.REAL)


class TestMLIntegrationManager(unittest.TestCase):
    """Тесты для класса MLIntegrationManager."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        # Создаем мок для оркестратора
        self.orchestrator_mock = MagicMock()
        
        # Создаем экземпляр менеджера ML-интеграции
        self.ml_manager = MLIntegrationManager(self.orchestrator_mock)
    
    @pytest.mark.asyncio
    async def test_load_model(self):
        """Тест загрузки модели."""
        # Вызываем метод load_model
        result = await self.ml_manager.load_model("test_model")
        
        # Проверяем, что метод вернул True
        self.assertTrue(result)
        
        # Проверяем, что модель была добавлена в словарь моделей
        self.assertIn("test_model", self.ml_manager.models)
    
    @pytest.mark.asyncio
    async def test_train_model(self):
        """Тест обучения модели."""
        # Вызываем метод train_model
        result = await self.ml_manager.train_model("test_model", {"data": "test_data"})
        
        # Проверяем, что метод вернул True
        self.assertTrue(result)
    
    @pytest.mark.asyncio
    async def test_predict(self):
        """Тест получения предсказания от модели."""
        # Добавляем модель в словарь моделей
        self.ml_manager.models["test_model"] = {"name": "test_model"}
        
        # Вызываем метод predict
        result = await self.ml_manager.predict("test_model", {"data": "test_data"})
        
        # Проверяем, что метод вернул не None
        self.assertIsNotNone(result)


class TestEventBus(unittest.TestCase):
    """Тесты для класса EventBus."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        # Создаем экземпляр шины событий
        self.event_bus = EventBus()
    
    def test_subscribe(self):
        """Тест подписки на событие."""
        # Создаем обработчик события
        handler = MagicMock()
        
        # Подписываемся на событие
        self.event_bus.subscribe("test_event", handler)
        
        # Проверяем, что обработчик был добавлен в словарь обработчиков
        self.assertIn("test_event", self.event_bus.handlers)
        self.assertIn(handler, self.event_bus.handlers["test_event"])
    
    def test_unsubscribe(self):
        """Тест отписки от события."""
        # Создаем обработчик события
        handler = MagicMock()
        
        # Подписываемся на событие
        self.event_bus.subscribe("test_event", handler)
        
        # Отписываемся от события
        result = self.event_bus.unsubscribe("test_event", handler)
        
        # Проверяем, что метод вернул True
        self.assertTrue(result)
        
        # Проверяем, что обработчик был удален из словаря обработчиков
        self.assertNotIn(handler, self.event_bus.handlers["test_event"])
    
    @pytest.mark.asyncio
    async def test_publish(self):
        """Тест публикации события."""
        # Создаем обработчик события
        handler = AsyncMock()
        
        # Подписываемся на событие
        self.event_bus.subscribe("test_event", handler)
        
        # Публикуем событие
        await self.event_bus.publish("test_event", {"data": "test_data"})
        
        # Проверяем, что обработчик был вызван с правильными аргументами
        handler.assert_called_once_with({"data": "test_data"})


class TestCommandProcessor(unittest.TestCase):
    """Тесты для класса CommandProcessor."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        # Создаем мок для оркестратора
        self.orchestrator_mock = MagicMock()
        
        # Создаем экземпляр процессора команд
        self.command_processor = CommandProcessor(self.orchestrator_mock)
    
    @pytest.mark.asyncio
    async def test_process_command_line(self):
        """Тест обработки командной строки."""
        # Мокаем метод process_command оркестратора
        self.orchestrator_mock.process_command = AsyncMock(return_value={"success": True, "message": "Command executed"})
        
        # Вызываем метод process_command_line
        result = await self.command_processor.process_command_line("test_command arg1 arg2")
        
        # Проверяем результат
        self.assertEqual(result, {"success": True, "message": "Command executed"})
        
        # Проверяем, что метод process_command оркестратора был вызван с правильными аргументами
        self.orchestrator_mock.process_command.assert_called_once_with("test_command", ["arg1", "arg2"])
    
    @pytest.mark.asyncio
    async def test_process_command_line_empty(self):
        """Тест обработки пустой командной строки."""
        # Вызываем метод process_command_line с пустой строкой
        result = await self.command_processor.process_command_line("")
        
        # Проверяем результат
        self.assertEqual(result, {"success": False, "message": "Пустая команда"})
        
        # Проверяем, что метод process_command оркестратора не был вызван
        self.orchestrator_mock.process_command.assert_not_called()


if __name__ == "__main__":
    unittest.main() 