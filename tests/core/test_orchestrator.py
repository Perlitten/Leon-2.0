"""
Тесты для модуля оркестратора.
"""

import unittest
from unittest.mock import MagicMock, patch

from core.orchestrator import (
    LeonOrchestrator, EventBus, TradingModeManager, 
    MLIntegrationManager, CommandProcessor
)
from core.constants import TRADING_MODES, EVENT_TYPES, SYSTEM_STATUSES
from core.exceptions import InvalidModeError, CommandError, ModelLoadError


class TestEventBus(unittest.TestCase):
    """Тесты для класса EventBus."""
    
    def setUp(self):
        """Настройка тестов."""
        self.event_bus = EventBus()
    
    def test_subscribe_and_publish(self):
        """Тест подписки на событие и публикации события."""
        # Создание обработчика события
        handler = MagicMock()
        
        # Подписка на событие
        self.event_bus.subscribe("TEST_EVENT", handler)
        
        # Публикация события
        test_data = {"test": "data"}
        self.event_bus.publish("TEST_EVENT", test_data)
        
        # Проверка вызова обработчика
        handler.assert_called_once_with(test_data)
    
    def test_unsubscribe(self):
        """Тест отписки от события."""
        # Создание обработчика события
        handler = MagicMock()
        
        # Подписка на событие
        self.event_bus.subscribe("TEST_EVENT", handler)
        
        # Отписка от события
        self.event_bus.unsubscribe("TEST_EVENT", handler)
        
        # Публикация события
        self.event_bus.publish("TEST_EVENT", {"test": "data"})
        
        # Проверка, что обработчик не был вызван
        handler.assert_not_called()
    
    def test_multiple_handlers(self):
        """Тест нескольких обработчиков для одного события."""
        # Создание обработчиков события
        handler1 = MagicMock()
        handler2 = MagicMock()
        
        # Подписка на событие
        self.event_bus.subscribe("TEST_EVENT", handler1)
        self.event_bus.subscribe("TEST_EVENT", handler2)
        
        # Публикация события
        test_data = {"test": "data"}
        self.event_bus.publish("TEST_EVENT", test_data)
        
        # Проверка вызова обработчиков
        handler1.assert_called_once_with(test_data)
        handler2.assert_called_once_with(test_data)
    
    def test_handler_exception(self):
        """Тест обработки исключения в обработчике события."""
        # Создание обработчика события, который вызывает исключение
        def handler_with_exception(data):
            raise Exception("Test exception")
        
        # Создание обычного обработчика
        normal_handler = MagicMock()
        
        # Подписка на событие
        self.event_bus.subscribe("TEST_EVENT", handler_with_exception)
        self.event_bus.subscribe("TEST_EVENT", normal_handler)
        
        # Публикация события
        test_data = {"test": "data"}
        self.event_bus.publish("TEST_EVENT", test_data)
        
        # Проверка, что обычный обработчик был вызван, несмотря на исключение в первом обработчике
        normal_handler.assert_called_once_with(test_data)


class TestTradingModeManager(unittest.TestCase):
    """Тесты для класса TradingModeManager."""
    
    def setUp(self):
        """Настройка тестов."""
        self.config_manager = MagicMock()
        self.event_bus = MagicMock()
        self.trading_mode_manager = TradingModeManager(self.config_manager, self.event_bus)
    
    def test_initialize(self):
        """Тест инициализации менеджера режимов торговли."""
        # Настройка мока конфигурации
        self.config_manager.get_config.return_value = {
            "trading": {
                "default_mode": TRADING_MODES["DRY"]
            }
        }
        
        # Инициализация менеджера режимов торговли
        self.trading_mode_manager.initialize()
        
        # Проверка, что был вызван метод set_mode с правильным режимом
        self.assertEqual(self.trading_mode_manager._current_mode, TRADING_MODES["DRY"])
    
    def test_set_mode(self):
        """Тест установки режима работы."""
        # Установка режима работы
        self.trading_mode_manager.set_mode(TRADING_MODES["BACKTEST"])
        
        # Проверка, что режим работы был установлен
        self.assertEqual(self.trading_mode_manager.get_current_mode(), TRADING_MODES["BACKTEST"])
        
        # Проверка, что было опубликовано событие о смене режима
        self.event_bus.publish.assert_called_once_with(
            EVENT_TYPES["SYSTEM"]["MODE_CHANGED"],
            {"mode": TRADING_MODES["BACKTEST"]}
        )
    
    def test_set_invalid_mode(self):
        """Тест установки недопустимого режима работы."""
        # Попытка установки недопустимого режима работы
        with self.assertRaises(InvalidModeError):
            self.trading_mode_manager.set_mode("INVALID_MODE")


class TestMLIntegrationManager(unittest.TestCase):
    """Тесты для класса MLIntegrationManager."""
    
    def setUp(self):
        """Настройка тестов."""
        self.config_manager = MagicMock()
        self.event_bus = MagicMock()
        self.ml_integration_manager = MLIntegrationManager(self.config_manager, self.event_bus)
    
    def test_initialize(self):
        """Тест инициализации менеджера интеграции с ML-моделями."""
        # Настройка мока конфигурации
        self.config_manager.get_config.return_value = {
            "ml": {
                "default_model": "lstm",
                "models": {
                    "lstm": {
                        "type": "lstm",
                        "path": "models/lstm.h5"
                    }
                }
            }
        }
        
        # Патчинг метода _load_model
        with patch.object(self.ml_integration_manager, '_load_model') as mock_load_model:
            # Инициализация менеджера интеграции с ML-моделями
            self.ml_integration_manager.initialize()
            
            # Проверка, что был вызван метод _load_model с правильными параметрами
            mock_load_model.assert_called_once_with(
                "lstm",
                {
                    "type": "lstm",
                    "path": "models/lstm.h5"
                }
            )
    
    def test_set_active_model(self):
        """Тест установки активной модели."""
        # Добавление модели в список моделей
        self.ml_integration_manager._models = {
            "lstm": {
                "name": "lstm",
                "config": {},
                "instance": None
            }
        }
        
        # Установка активной модели
        self.ml_integration_manager.set_active_model("lstm")
        
        # Проверка, что активная модель была установлена
        self.assertEqual(self.ml_integration_manager.get_active_model(), "lstm")
        
        # Проверка, что было опубликовано событие о смене активной модели
        self.event_bus.publish.assert_called_once_with(
            EVENT_TYPES["ML"]["ACTIVE_MODEL_CHANGED"],
            {"model_name": "lstm"}
        )
    
    def test_set_invalid_model(self):
        """Тест установки недопустимой модели."""
        # Попытка установки недопустимой модели
        with self.assertRaises(ModelLoadError):
            self.ml_integration_manager.set_active_model("invalid_model")


class TestCommandProcessor(unittest.TestCase):
    """Тесты для класса CommandProcessor."""
    
    def setUp(self):
        """Настройка тестов."""
        self.event_bus = MagicMock()
        self.command_processor = CommandProcessor(self.event_bus)
    
    def test_register_handler(self):
        """Тест регистрации обработчика команды."""
        # Создание обработчика команды
        handler = MagicMock()
        
        # Регистрация обработчика команды
        self.command_processor.register_handler("test_command", handler)
        
        # Проверка, что обработчик был зарегистрирован
        self.assertIn("test_command", self.command_processor._command_handlers)
        self.assertEqual(self.command_processor._command_handlers["test_command"], handler)
    
    def test_process_command(self):
        """Тест обработки команды."""
        # Создание обработчика команды
        handler = MagicMock(return_value={"success": True, "message": "Test command executed"})
        
        # Регистрация обработчика команды
        self.command_processor.register_handler("test_command", handler)
        
        # Обработка команды
        result = self.command_processor.process_command("test_command", ["arg1", "arg2"])
        
        # Проверка, что обработчик был вызван с правильными аргументами
        handler.assert_called_once_with(["arg1", "arg2"])
        
        # Проверка результата выполнения команды
        self.assertEqual(result, {"success": True, "message": "Test command executed"})
        
        # Проверка, что было опубликовано событие о выполнении команды
        self.event_bus.publish.assert_called_once_with(
            EVENT_TYPES["SYSTEM"]["COMMAND_EXECUTED"],
            {
                "command": "test_command",
                "args": ["arg1", "arg2"],
                "result": {"success": True, "message": "Test command executed"}
            }
        )
    
    def test_process_invalid_command(self):
        """Тест обработки недопустимой команды."""
        # Попытка обработки недопустимой команды
        with self.assertRaises(CommandError):
            self.command_processor.process_command("invalid_command")


class TestLeonOrchestrator(unittest.TestCase):
    """Тесты для класса LeonOrchestrator."""
    
    def setUp(self):
        """Настройка тестов."""
        # Патчинг классов
        self.config_manager_patch = patch('core.orchestrator.ConfigManager')
        self.localization_patch = patch('core.orchestrator.LocalizationManager')
        self.event_bus_patch = patch('core.orchestrator.EventBus')
        self.trading_mode_manager_patch = patch('core.orchestrator.TradingModeManager')
        self.ml_integration_manager_patch = patch('core.orchestrator.MLIntegrationManager')
        self.command_processor_patch = patch('core.orchestrator.CommandProcessor')
        
        # Получение моков
        self.config_manager_mock = self.config_manager_patch.start()
        self.localization_mock = self.localization_patch.start()
        self.event_bus_mock = self.event_bus_patch.start()
        self.trading_mode_manager_mock = self.trading_mode_manager_patch.start()
        self.ml_integration_manager_mock = self.ml_integration_manager_patch.start()
        self.command_processor_mock = self.command_processor_patch.start()
        
        # Создание экземпляра оркестратора
        self.orchestrator = LeonOrchestrator()
    
    def tearDown(self):
        """Очистка после тестов."""
        # Остановка патчей
        self.config_manager_patch.stop()
        self.localization_patch.stop()
        self.event_bus_patch.stop()
        self.trading_mode_manager_patch.stop()
        self.ml_integration_manager_patch.stop()
        self.command_processor_patch.stop()
    
    def test_init(self):
        """Тест инициализации оркестратора."""
        # Проверка, что все компоненты были инициализированы
        self.config_manager_mock.assert_called_once_with("config/config.yaml")
        self.localization_mock.assert_called_once_with(dry_mode=False)
        self.event_bus_mock.assert_called_once()
        self.trading_mode_manager_mock.assert_called_once()
        self.ml_integration_manager_mock.assert_called_once()
        self.command_processor_mock.assert_called_once()
    
    def test_start(self):
        """Тест запуска оркестратора."""
        # Запуск оркестратора
        self.orchestrator.start()
        
        # Проверка, что был вызван метод load_config у менеджера конфигурации
        self.orchestrator._config_manager.load_config.assert_called_once()
        
        # Проверка, что был вызван метод initialize у менеджера режимов торговли
        self.orchestrator._trading_mode_manager.initialize.assert_called_once()
        
        # Проверка, что был вызван метод initialize у менеджера интеграции с ML-моделями
        self.orchestrator._ml_integration_manager.initialize.assert_called_once()
        
        # Проверка, что статус был установлен в "RUNNING"
        self.assertEqual(self.orchestrator._status, SYSTEM_STATUSES["RUNNING"])
        
        # Проверка, что было опубликовано событие о запуске системы
        self.orchestrator._event_bus.publish.assert_called_once_with(EVENT_TYPES["SYSTEM"]["STARTED"])
    
    def test_stop(self):
        """Тест остановки оркестратора."""
        # Остановка оркестратора
        self.orchestrator.stop()
        
        # Проверка, что статус был установлен в "STOPPED"
        self.assertEqual(self.orchestrator._status, SYSTEM_STATUSES["STOPPED"])
        
        # Проверка, что было опубликовано событие об остановке системы
        self.orchestrator._event_bus.publish.assert_called_with(EVENT_TYPES["SYSTEM"]["STOPPED"])
    
    def test_process_command(self):
        """Тест обработки команды."""
        # Обработка команды
        self.orchestrator.process_command("test_command", ["arg1", "arg2"])
        
        # Проверка, что был вызван метод process_command у процессора команд
        self.orchestrator._command_processor.process_command.assert_called_once_with("test_command", ["arg1", "arg2"])
    
    def test_handle_start_command(self):
        """Тест обработчика команды "start"."""
        # Патчинг метода start
        with patch.object(self.orchestrator, 'start') as mock_start:
            # Вызов обработчика команды "start"
            result = self.orchestrator._handle_start_command([])
            
            # Проверка, что был вызван метод start
            mock_start.assert_called_once()
            
            # Проверка результата выполнения команды
            self.assertEqual(result, {"success": True, "message": "Оркестратор запущен"})
    
    def test_handle_stop_command(self):
        """Тест обработчика команды "stop"."""
        # Патчинг метода stop
        with patch.object(self.orchestrator, 'stop') as mock_stop:
            # Вызов обработчика команды "stop"
            result = self.orchestrator._handle_stop_command([])
            
            # Проверка, что был вызван метод stop
            mock_stop.assert_called_once()
            
            # Проверка результата выполнения команды
            self.assertEqual(result, {"success": True, "message": "Оркестратор остановлен"})
    
    def test_handle_status_command(self):
        """Тест обработчика команды "status"."""
        # Настройка моков
        self.orchestrator._status = SYSTEM_STATUSES["RUNNING"]
        self.orchestrator._trading_mode_manager.get_current_mode.return_value = TRADING_MODES["DRY"]
        self.orchestrator._ml_integration_manager.get_active_model.return_value = "lstm"
        
        # Вызов обработчика команды "status"
        result = self.orchestrator._handle_status_command([])
        
        # Проверка результата выполнения команды
        self.assertEqual(result, {
            "status": SYSTEM_STATUSES["RUNNING"],
            "mode": TRADING_MODES["DRY"],
            "active_model": "lstm"
        })
    
    def test_handle_set_mode_command(self):
        """Тест обработчика команды "set_mode"."""
        # Вызов обработчика команды "set_mode" без аргументов
        result = self.orchestrator._handle_set_mode_command([])
        
        # Проверка результата выполнения команды
        self.assertEqual(result, {"success": False, "message": "Не указан режим работы"})
        
        # Вызов обработчика команды "set_mode" с аргументами
        result = self.orchestrator._handle_set_mode_command([TRADING_MODES["BACKTEST"]])
        
        # Проверка, что был вызван метод set_mode у менеджера режимов торговли
        self.orchestrator._trading_mode_manager.set_mode.assert_called_once_with(TRADING_MODES["BACKTEST"])
        
        # Проверка результата выполнения команды
        self.assertEqual(result, {"success": True, "message": f"Установлен режим работы: {TRADING_MODES['BACKTEST']}"})
    
    def test_handle_set_model_command(self):
        """Тест обработчика команды "set_model"."""
        # Вызов обработчика команды "set_model" без аргументов
        result = self.orchestrator._handle_set_model_command([])
        
        # Проверка результата выполнения команды
        self.assertEqual(result, {"success": False, "message": "Не указана модель"})
        
        # Вызов обработчика команды "set_model" с аргументами
        result = self.orchestrator._handle_set_model_command(["lstm"])
        
        # Проверка, что был вызван метод set_active_model у менеджера интеграции с ML-моделями
        self.orchestrator._ml_integration_manager.set_active_model.assert_called_once_with("lstm")
        
        # Проверка результата выполнения команды
        self.assertEqual(result, {"success": True, "message": "Установлена активная модель: lstm"})


if __name__ == '__main__':
    unittest.main() 