"""
Модуль оркестратора для Leon Trading Bot.

Этот модуль предоставляет центральный компонент для управления и координации
всех подсистем Leon Trading Bot. Оркестратор отвечает за инициализацию компонентов,
управление жизненным циклом системы, переключение между режимами работы и обработку событий.
"""

import asyncio
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Callable, Set
from datetime import datetime
import random
import threading
import yaml

from core.config_manager import ConfigManager
from core.component_factory import (
    ExchangeFactory, NotificationFactory, TradingFactory, 
    VisualizationFactory, MLFactory
)
from core.exceptions import (
    OrchestratorError, InitializationError, OperationError, 
    InvalidModeError, CommandError, ModelLoadError, 
    PredictionError, EvaluationError
)
from core.constants import (
    EVENT_TYPES, TRADING_MODES, LOCALIZATION_KEYS, SYSTEM_STATUSES, TradingModes, EventTypes
)
from core.localization import LocalizationManager
from notification.telegram.bot import TelegramBot
from visualization.manager import VisualizationManager


class EventBus:
    """
    Шина событий для обмена сообщениями между компонентами системы.
    
    Позволяет компонентам регистрировать обработчики событий и генерировать события.
    """
    
    def __init__(self):
        """Инициализация шины событий."""
        self.handlers: Dict[str, Set[Callable]] = {}
        self.logger = logging.getLogger("EventBus")
    
    def register_handler(self, event_type: str, handler: Callable) -> None:
        """
        Регистрирует обработчик событий.
        
        Args:
            event_type: Тип события
            handler: Функция-обработчик события
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = set()
        
        self.handlers[event_type].add(handler)
        self.logger.debug(f"Зарегистрирован обработчик для события '{event_type}'")
    
    def unregister_handler(self, event_type: str, handler: Callable) -> bool:
        """
        Отменяет регистрацию обработчика событий.
        
        Args:
            event_type: Тип события
            handler: Функция-обработчик события
            
        Returns:
            bool: Успешность отмены регистрации
        """
        if event_type not in self.handlers or handler not in self.handlers[event_type]:
            self.logger.warning(f"Попытка отменить регистрацию несуществующего обработчика для события '{event_type}'")
            return False
        
        self.handlers[event_type].remove(handler)
        self.logger.debug(f"Отменена регистрация обработчика для события '{event_type}'")
        return True
    
    async def emit(self, event_type: str, data: Any = None) -> None:
        """
        Генерирует событие.
        
        Args:
            event_type: Тип события
            data: Данные события
        """
        if event_type not in self.handlers:
            self.logger.debug(f"Нет обработчиков для события '{event_type}'")
            return
        
        self.logger.debug(f"Генерация события '{event_type}'")
        
        # Если это событие успешной сделки, выводим случайную фразу
        if event_type == EVENT_TYPES["TRADE_COMPLETED"] and data and data.get("profit", 0) > 0:
            localization_manager = self.orchestrator.localization_manager if hasattr(self, 'orchestrator') else None
            if localization_manager:
                success_phrases = localization_manager.get_text(LOCALIZATION_KEYS["SUCCESS_PHRASES"])
                success_phrase = random.choice(success_phrases)
                print(f"\n🎉 {success_phrase}")
        
        # Создаем копию множества обработчиков, чтобы избежать изменения во время итерации
        handlers = self.handlers[event_type].copy()
        
        # Вызываем все обработчики асинхронно
        tasks = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(asyncio.create_task(handler(data)))
            else:
                try:
                    handler(data)
                except Exception as e:
                    self.logger.error(f"Ошибка в обработчике события '{event_type}': {str(e)}")
                    self.logger.debug(traceback.format_exc())
        
        # Ожидаем завершения всех асинхронных обработчиков
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class CommandProcessor:
    """
    Обработчик команд пользователя.
    
    Позволяет регистрировать обработчики команд и выполнять команды.
    """
    
    def __init__(self, orchestrator: 'LeonOrchestrator'):
        """
        Инициализация обработчика команд.
        
        Args:
            orchestrator: Экземпляр оркестратора
        """
        self.orchestrator = orchestrator
        self.commands: Dict[str, Callable] = {}
        self.logger = logging.getLogger("CommandProcessor")
    
    def register_command(self, command: str, handler: Callable) -> None:
        """
        Регистрирует обработчик команды.
        
        Args:
            command: Команда
            handler: Функция-обработчик команды
        """
        self.commands[command] = handler
        self.logger.debug(f"Зарегистрирован обработчик для команды '{command}'")
    
    def validate_command(self, command: str, *args, **kwargs) -> bool:
        """
        Проверяет допустимость команды и ее аргументов.
        
        Args:
            command: Команда для проверки
            *args, **kwargs: Аргументы команды
            
        Returns:
            bool: Допустимость команды
        """
        if command not in self.commands:
            self.logger.warning(f"Неизвестная команда: '{command}'")
            return False
        
        return True
    
    async def process_command(self, command: str, *args, **kwargs) -> Any:
        """
        Обрабатывает команду пользователя.
        
        Args:
            command: Команда для выполнения
            *args, **kwargs: Аргументы команды
            
        Returns:
            Any: Результат выполнения команды
            
        Raises:
            CommandError: При ошибке выполнения команды
        """
        if not self.validate_command(command, *args, **kwargs):
            raise CommandError(f"Неизвестная команда: '{command}'", command=command)
        
        handler = self.commands[command]
        
        try:
            self.logger.info(f"Выполнение команды: '{command}'")
            
            # Проверяем, есть ли в аргументах ответ "нет"
            has_no_answer = False
            for arg in args:
                if isinstance(arg, str) and arg.lower() in ["нет", "no", "n", "н"]:
                    has_no_answer = True
                    break
            
            for key, value in kwargs.items():
                if isinstance(value, str) and value.lower() in ["нет", "no", "n", "н"]:
                    has_no_answer = True
                    break
            
            # Если есть ответ "нет", выводим шуточное сообщение
            if has_no_answer and hasattr(self.orchestrator, 'localization_manager'):
                no_answer_text = self.orchestrator.localization_manager.get_text(LOCALIZATION_KEYS["NO_ANSWER"])
                print(f"\n😏 {no_answer_text}\n")
            
            if asyncio.iscoroutinefunction(handler):
                return await handler(*args, **kwargs)
            else:
                return handler(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Ошибка при выполнении команды '{command}': {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise CommandError(f"Ошибка при выполнении команды '{command}': {str(e)}", command=command) from e


class TradingModeManager:
    """
    Менеджер режимов торговли.
    
    Отвечает за переключение между режимами работы системы (dry, real, backtest).
    """
    
    def __init__(self, orchestrator: 'LeonOrchestrator'):
        """
        Инициализация менеджера режимов торговли.
        
        Args:
            orchestrator: Экземпляр оркестратора
        """
        self.orchestrator = orchestrator
        self.current_mode: Optional[str] = None
        self.available_modes = ["dry", "real", "backtest"]
        self.logger = logging.getLogger("TradingModeManager")
        
        # Компоненты для торговли
        self.trader = None
        self.position_monitor = None
    
    def validate_mode(self, mode: str) -> bool:
        """
        Проверяет допустимость указанного режима.
        
        Args:
            mode: Режим работы для проверки
            
        Returns:
            bool: Допустимость режима
        """
        if mode not in self.available_modes:
            self.logger.warning(f"Недопустимый режим работы: '{mode}'")
            return False
        
        return True
    
    def get_current_mode(self) -> str:
        """
        Возвращает текущий режим работы.
        
        Returns:
            str: Текущий режим работы
        """
        return self.current_mode
    
    async def switch_to_mode(self, mode: str) -> bool:
        """
        Переключает систему в указанный режим работы.
        
        Args:
            mode: Режим работы ("dry", "real", "backtest")
            
        Returns:
            bool: Успешность переключения
            
        Raises:
            InvalidModeError: При указании недопустимого режима
        """
        if not self.validate_mode(mode):
            raise InvalidModeError(f"Недопустимый режим работы: '{mode}'", mode=mode)
        
        if self.current_mode == mode:
            self.logger.info(f"Система уже работает в режиме '{mode}'")
            return True
        
        # Если система уже работает в другом режиме, останавливаем ее
        if self.current_mode is not None:
            self.logger.info(f"Остановка режима '{self.current_mode}'")
            await self._stop_current_mode()
        
        self.logger.info(f"Переключение в режим '{mode}'")
        
        # Запускаем новый режим
        await self._start_mode(mode)
        
        self.current_mode = mode
        
        # Генерируем событие о смене режима
        await self.orchestrator.event_bus.emit("mode_changed", {
            "mode": mode,
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    
    async def _stop_current_mode(self) -> None:
        """Останавливает текущий режим работы."""
        mode = self.current_mode
        
        # Выводим сообщение о завершении режима
        if mode:
            localization_manager = self.orchestrator.localization_manager
            exit_messages = localization_manager.get_text(LOCALIZATION_KEYS["MODE_EXIT_MESSAGES"])
            if mode in exit_messages:
                print(exit_messages[mode])
        
        # Останавливаем мониторинг позиций
        if self.position_monitor:
            try:
                await self.position_monitor.stop()
                self.logger.info("Мониторинг позиций остановлен")
            except Exception as e:
                self.logger.error(f"Ошибка при остановке мониторинга позиций: {str(e)}")
        
        # Останавливаем трейдер
        if self.trader:
            try:
                await self.trader.stop()
                self.logger.info("Трейдер остановлен")
            except Exception as e:
                self.logger.error(f"Ошибка при остановке трейдера: {str(e)}")
        
        # Останавливаем визуализацию
        if hasattr(self.orchestrator, 'visualization_manager'):
            await self.orchestrator.visualization_manager.stop_visualization()
    
    async def _start_mode(self, mode: str) -> None:
        """
        Запускает указанный режим работы.
        
        Args:
            mode: Режим работы ("dry", "real", "backtest")
        """
        config = self.orchestrator.config_manager.get_config()
        localization_manager = self.orchestrator.localization_manager
        
        # Получаем фабрики компонентов
        trading_factory = TradingFactory(config)
        
        # Получаем необходимые компоненты из оркестратора
        binance_client = self.orchestrator.binance_client
        telegram = self.orchestrator.telegram
        strategy = self.orchestrator.strategy
        risk_controller = self.orchestrator.risk_controller
        decision_maker = self.orchestrator.decision_maker
        visualizer = self.orchestrator.visualizer if hasattr(self.orchestrator, 'visualizer') else None
        
        # Параметры для трейдера
        symbol = config["general"]["symbol"]
        leverage = config["general"]["leverage"]
        
        # Отправляем уведомление в Telegram, если он настроен
        if telegram and hasattr(telegram, 'connected') and telegram.connected:
            try:
                trader_name = {"dry": "DryModeTrader", "real": "RealTrader", "backtest": "BacktestTrader"}[mode]
                await telegram.send_message(f"🚀 Запуск торгового бота LEON\n\nРежим: {trader_name}\nПара: {symbol}\nКредитное плечо: {leverage}x")
                self.logger.info("Отправлено уведомление о запуске в Telegram")
            except Exception as e:
                self.logger.error(f"Ошибка отправки уведомления в Telegram: {str(e)}")
            self.logger.debug(traceback.format_exc())
        
        # Базовые параметры для трейдера
        trader_params = {
            'symbol': symbol,
            'binance_client': binance_client,
            'strategy': strategy,
            'telegram': telegram,
            'risk_controller': risk_controller,
            'leverage': leverage,
            'decision_maker': decision_maker
        }
        
        # Добавляем дополнительные параметры в зависимости от режима
        if mode == "dry":
            trader_params.update({
                'initial_balance': config["general"]["initial_balance"],
                'visualizer': visualizer,
                'update_interval': config["general"].get("update_interval", 5)
            })
        elif mode == "backtest":
            trader_params.update({
                'interval': config["backtest"]["interval"],
                'days': config["backtest"]["days"],
                'commission': config["backtest"]["commission"]
            })
        
        # Создаем и запускаем трейдер
        self.trader = trading_factory.create_trader(mode, **trader_params)
        await self.trader.start()
        
        # Инициализируем мониторинг позиций
        from trading.position_monitor import PositionMonitor
        self.position_monitor = PositionMonitor(
            trader=self.trader,
            max_position_age_hours=config["safety"]["stuck_position_timeout"],
            check_interval_minutes=15,
            loss_threshold_percent=config["safety"].get("max_position_loss", 5.0),
            enable_cleanup=config["safety"]["cleanup_stuck_positions"]
        )
        await self.position_monitor.start()
        
        # Выводим приветственное сообщение в зависимости от режима
        welcome_messages = localization_manager.get_text(LOCALIZATION_KEYS["MODE_WELCOME_MESSAGES"])
        print(welcome_messages.get(mode, f"\n🚀 РЕЖИМ '{mode.upper()}' АКТИВИРОВАН!\n"))
        
        warning_messages = localization_manager.get_text(LOCALIZATION_KEYS["MODE_WARNING_MESSAGES"])
        if mode == TRADING_MODES["DRY"]:
            balance = config['general']['initial_balance']
            print(warning_messages["dry"].format(balance=balance))
        elif mode == TRADING_MODES["REAL"]:
            print(warning_messages["real"])
        elif mode == TRADING_MODES["BACKTEST"]:
            interval = config["backtest"]["interval"]
            days = config["backtest"]["days"]
            print(warning_messages["backtest"].format(interval=interval, days=days))
        
        # Если используется ML, выводим дополнительное сообщение
        use_ml = (config["general"].get("decision_mode") == "ml" or config["strategy"].get("use_ml", False))
        if use_ml:
            ml_phrases = localization_manager.get_text(LOCALIZATION_KEYS["ML_PHRASES"])
            ml_phrase = random.choice(ml_phrases)
            budget_killer_text = localization_manager.get_text(LOCALIZATION_KEYS["BUDGET_KILLER"])
            print(budget_killer_text.format(phrase=ml_phrase))
        
        press_ctrl_c = localization_manager.get_text(LOCALIZATION_KEYS["PRESS_CTRL_C"])
        print(press_ctrl_c)


class MLIntegrationManager:
    """
    Менеджер интеграции с машинным обучением.
    
    Отвечает за загрузку, обучение и использование моделей машинного обучения.
    """
    
    def __init__(self, orchestrator: 'LeonOrchestrator'):
        """
        Инициализация менеджера интеграции с машинным обучением.
        
        Args:
            orchestrator: Экземпляр оркестратора
        """
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = orchestrator.config if orchestrator else {}
        
        # Загруженные модели
        self.models = {}
        
        self.logger.info("Менеджер интеграции с машинным обучением инициализирован")
    
    async def load_model(self, model_name: str) -> bool:
        """
        Загружает ML-модель.
        
        Args:
            model_name: Имя модели для загрузки
            
        Returns:
            bool: Успешность загрузки
            
        Raises:
            ModelLoadError: При ошибке загрузки модели
        """
        try:
            self.logger.info(f"Загрузка модели '{model_name}'")
            
            # Здесь должна быть логика загрузки модели
            # ...
            
            # Временная заглушка
            self.models[model_name] = {
                "name": model_name,
                "loaded_at": datetime.now().isoformat()
            }
            
            self.current_model = model_name
            
            # Генерируем событие о загрузке модели
            await self.orchestrator.event_bus.emit("model_loaded", {
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            })
            
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели '{model_name}': {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise ModelLoadError(f"Ошибка при загрузке модели '{model_name}': {str(e)}", model_name=model_name) from e
    
    async def train_model(self) -> Dict[str, Any]:
        """
        Обучает ML-модель на исторических данных.
        
        Returns:
            Dict[str, Any]: Результат обучения с метриками
            
        Raises:
            ModelLoadError: При ошибке обучения модели
        """
        try:
            self.logger.info("Запуск обучения модели")
            
            # Получаем конфигурацию
            config = self.orchestrator.config_manager.get_config()
            model_config = config.get("ml", {})
            
            # Получаем параметры обучения
            model_name = model_config.get("model_name", "default")
            epochs = model_config.get("epochs", 10)
            batch_size = model_config.get("batch_size", 32)
            learning_rate = model_config.get("learning_rate", 0.001)
            
            # Логируем параметры обучения
            self.logger.info(f"Параметры обучения: model={model_name}, epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
            
            # Здесь должна быть логика обучения модели
            # ...
            
            # Временная заглушка для имитации обучения
            import random
            import time
            
            # Имитируем процесс обучения
            for epoch in range(epochs):
                self.logger.info(f"Эпоха {epoch+1}/{epochs}")
                time.sleep(0.5)  # Имитация времени обучения
            
            # Генерируем случайные метрики
            accuracy = 0.7 + random.random() * 0.2
            precision = 0.65 + random.random() * 0.25
            recall = 0.6 + random.random() * 0.3
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Сохраняем модель
            self.models[model_name] = {
                "name": model_name,
                "trained_at": datetime.now().isoformat(),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score
                }
            }
            
            self.current_model = model_name
            
            # Генерируем событие о завершении обучения
            await self.orchestrator.event_bus.emit("model_trained", {
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score
                }
            })
            
            return {
                "success": True,
                "model_name": model_name,
                "metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score
                }
            }
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_prediction(self, data: Any) -> Dict[str, Any]:
        """
        Получает предсказание от ML-модели.
        
        Args:
            data: Данные для предсказания
            
        Returns:
            Dict[str, Any]: Результат предсказания
            
        Raises:
            PredictionError: При ошибке предсказания
        """
        if self.current_model is None:
            raise PredictionError("Нет загруженной модели")
        
        try:
            self.logger.debug(f"Получение предсказания от модели '{self.current_model}'")
            
            # Здесь должна быть логика получения предсказания
            # ...
            
            # Временная заглушка
            prediction = {
                "model": self.current_model,
                "timestamp": datetime.now().isoformat(),
                "prediction": {
                    "direction": "BUY" if datetime.now().second % 2 == 0 else "SELL",
                    "confidence": 0.75,
                    "target_price": 50000.0
                }
            }
            
            # Генерируем событие о получении предсказания
            await self.orchestrator.event_bus.emit("prediction_received", prediction)
            
            return prediction
        except Exception as e:
            self.logger.error(f"Ошибка при получении предсказания: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise PredictionError(f"Ошибка при получении предсказания: {str(e)}") from e
    
    async def evaluate_model(self, model_name: str, test_data: Any) -> Dict[str, Any]:
        """
        Оценивает эффективность ML-модели.
        
        Args:
            model_name: Имя модели для оценки
            test_data: Тестовые данные
            
        Returns:
            Dict[str, Any]: Результаты оценки
            
        Raises:
            EvaluationError: При ошибке оценки
        """
        if model_name not in self.models:
            raise EvaluationError(f"Модель '{model_name}' не загружена", model_name=model_name)
        
        try:
            self.logger.info(f"Оценка модели '{model_name}'")
            
            # Здесь должна быть логика оценки модели
            # ...
            
            # Временная заглушка
            evaluation = {
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "accuracy": 0.85,
                    "precision": 0.82,
                    "recall": 0.79,
                    "f1_score": 0.80
                }
            }
            
            # Генерируем событие о результатах оценки
            await self.orchestrator.event_bus.emit("model_evaluated", evaluation)
            
            return evaluation
        except Exception as e:
            self.logger.error(f"Ошибка при оценке модели '{model_name}': {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise EvaluationError(f"Ошибка при оценке модели '{model_name}': {str(e)}", model_name=model_name) from e


class LeonOrchestrator:
    """
    Основной оркестратор Leon Trading Bot.
    
    Отвечает за координацию всех компонентов системы.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Инициализация оркестратора.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path
        self.config = {}
        
        # Загрузка конфигурации
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке конфигурации: {str(e)}")
            self.config = {}
        
        # Статус системы
        self.status = SYSTEM_STATUSES["INITIALIZING"]
        self.running = False
        
        # Инициализация коллекций данных для визуализации
        self._prices = []
        self._indicators = {}
        self._signals = []
        
        # Добавляем блокировку для потокобезопасного доступа к данным
        self._data_lock = threading.Lock()
        
        # Инициализация компонентов
        self.localization_manager = LocalizationManager()
        self.event_bus = EventBus()
        self.command_processor = CommandProcessor(self)
        self.trading_mode_manager = TradingModeManager(self)
        self.ml_integration_manager = MLIntegrationManager(self)
        self.visualization_manager = VisualizationManager(self)
        
        # Регистрация базовых команд
        self._register_base_commands()
    
    def _register_base_commands(self) -> None:
        """Регистрирует базовые команды."""
        self.command_processor.register_command("start", self.start)
        self.command_processor.register_command("stop", self.stop)
        self.command_processor.register_command("switch_mode", self.switch_mode)
        self.command_processor.register_command("get_status", self.get_status)
    
    async def start(self, mode: Optional[str] = None) -> bool:
        """
        Запускает систему в указанном режиме.
        
        Args:
            mode: Режим работы ("dry", "real", "backtest"). Если не указан, используется режим из конфигурации.
            
        Returns:
            bool: Успешность запуска
            
        Raises:
            OperationError: При ошибке запуска
        """
        if not self.initialized:
            raise OperationError("Оркестратор не инициализирован")
        
        if self.running:
            self.logger.warning("Оркестратор уже запущен")
            return True
        
        try:
            # Если режим не указан, используем режим из конфигурации
            if mode is None:
                mode = self.config_manager.get_value("general.mode", "dry")
            
            self.logger.info(f"Запуск оркестратора в режиме '{mode}'")
            
            # Выводим случайное приветственное сообщение
            welcome_phrases = self.localization_manager.get_text(LOCALIZATION_KEYS["WELCOME_PHRASES"])
            welcome_phrase = random.choice(welcome_phrases)
            print(f"\n{welcome_phrase}\n")
            
            # Запускаем визуализацию
            await self.visualization_manager.start_visualization()
            
            # Переключаемся в указанный режим
            await self.trading_mode_manager.switch_to_mode(mode)
            
            # Устанавливаем флаг работы
            self.running = True
            
            # Генерируем событие о запуске
            await self.event_bus.emit("orchestrator_started", {
                "mode": mode,
                "timestamp": datetime.now().isoformat()
            })
            
            # Отправляем уведомление о запуске через Telegram
            if hasattr(self, 'telegram_bot') and self.telegram_bot:
                try:
                    self.logger.info("Отправка уведомления о запуске в Telegram")
                    # Получаем параметры из конфигурации
                    config = self.config_manager.get_config()
                    symbol = config.get("general", {}).get("symbol", "BTCUSDT")
                    balance = config.get("general", {}).get("initial_balance", 1000.0)
                    leverage = config.get("general", {}).get("leverage", 10)
                    risk_per_trade = config.get("strategy", {}).get("params", {}).get("risk_per_trade", 1.0)
                    stop_loss = config.get("strategy", {}).get("params", {}).get("stop_loss", 2.0)
                    take_profit = config.get("strategy", {}).get("params", {}).get("take_profit", 3.0)
                    
                    # Запускаем Telegram бота, если он еще не запущен
                    if not self.telegram_bot.is_running:
                        await self.telegram_bot.start()
                    
                    # Отправляем статус
                    await self.telegram_bot.send_status_update(
                        symbol=symbol,
                        mode=mode,
                        balance=balance,
                        leverage=leverage,
                        risk_per_trade=risk_per_trade,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    self.logger.info("Отправлено уведомление о запуске в Telegram")
                except Exception as e:
                    self.logger.error(f"Ошибка отправки уведомления в Telegram: {str(e)}")
                    self.logger.debug(traceback.format_exc())
            else:
                self.logger.warning("Telegram бот не инициализирован, уведомление не отправлено")
            
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при запуске оркестратора: {str(e)}")
            self.logger.debug(traceback.format_exc())
            
            # Пытаемся остановить систему в случае ошибки
            try:
                await self.stop()
            except Exception as stop_error:
                self.logger.error(f"Ошибка при остановке системы после сбоя: {str(stop_error)}")
            
            raise OperationError(f"Ошибка при запуске оркестратора: {str(e)}", operation="start") from e
    
    async def stop(self):
        """Останавливает работу системы."""
        if not self.initialized:
            self.logger.warning("Попытка остановить неинициализированную систему")
            return
        
        self.logger.info("Останавливаем систему...")
        
        # Отправляем итоги торговой сессии через Telegram
        if hasattr(self, 'telegram_bot') and self.telegram_bot:
            try:
                await self._send_session_summary()
            except Exception as e:
                self.logger.error(f"Ошибка при отправке итогов сессии в Telegram: {e}")
        
        # Останавливаем визуализацию
        if hasattr(self, 'visualization_manager'):
            try:
                await self.visualization_manager.stop_visualization()
            except Exception as e:
                self.logger.error(f"Ошибка при остановке визуализации: {e}")
        
        # Останавливаем все компоненты
        try:
            # Останавливаем стратегию, если она поддерживает метод stop
            if hasattr(self, 'strategy') and self.strategy and hasattr(self.strategy, 'stop'):
                try:
                    await self.strategy.stop()
                    self.logger.info("Стратегия остановлена")
                except Exception as e:
                    self.logger.error(f"Ошибка при остановке стратегии: {str(e)}")
                    self.logger.debug(traceback.format_exc())
            
            # Останавливаем интеграцию с биржей
            if hasattr(self, 'exchange_integration') and self.exchange_integration:
                await self.exchange_integration.stop()
            
            # Закрываем соединение с Binance
            if hasattr(self, 'binance_client') and self.binance_client:
                try:
                    if hasattr(self.binance_client, 'close'):
                        await self.binance_client.close()
                    self.logger.info("Соединение с Binance закрыто")
                except Exception as e:
                    self.logger.error(f"Ошибка при закрытии соединения с Binance: {str(e)}")
                    self.logger.debug(traceback.format_exc())
            
            # Останавливаем Telegram бота
            if hasattr(self, 'telegram_bot') and self.telegram_bot:
                await self.telegram_bot.stop()
            
            # Сбрасываем флаг работы
            self.running = False
            
            self.logger.info("Система успешно остановлена")
        except Exception as e:
            self.logger.error(f"Ошибка при остановке системы: {e}")
            raise
    
    async def _send_telegram_stop_notification(self):
        """Отправляет уведомление об остановке системы в Telegram."""
        try:
            if not hasattr(self, 'telegram_bot') or self.telegram_bot is None:
                self.logger.warning("Telegram бот не инициализирован, уведомление не отправлено")
                return
                
            # Формируем сообщение
            message = f"🛑 *Торговый бот остановлен*\n\n"
            current_mode = self.trading_mode_manager.get_current_mode()
            message += f"Режим: {current_mode.upper() if current_mode else 'Не установлен'}\n"
            
            # Добавляем информацию о времени работы
            if hasattr(self, 'start_time'):
                import datetime
                duration = datetime.datetime.now() - self.start_time
                hours, remainder = divmod(duration.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                message += f"Время работы: {hours}ч {minutes}м {seconds}с\n"
            
            # Отправляем сообщение
            await self.telegram.send_message(message)
            self.logger.info("Отправлено уведомление об остановке в Telegram")
        except Exception as e:
            self.logger.error(f"Ошибка при отправке уведомления в Telegram: {str(e)}")
            self.logger.debug(traceback.format_exc())
    
    async def switch_mode(self, mode: str) -> bool:
        """
        Переключает систему в указанный режим работы.
        
        Args:
            mode: Режим работы ("dry", "real", "backtest")
            
        Returns:
            bool: Успешность переключения
            
        Raises:
            InvalidModeError: При указании недопустимого режима
        """
        if not self.initialized:
            raise OperationError("Оркестратор не инициализирован")
        
        return await self.trading_mode_manager.switch_to_mode(mode)
    
    async def process_command(self, command: str, *args, **kwargs) -> Any:
        """
        Обрабатывает команду пользователя.
        
        Args:
            command: Команда для выполнения
            *args, **kwargs: Аргументы команды
            
        Returns:
            Any: Результат выполнения команды
            
        Raises:
            CommandError: При ошибке выполнения команды
        """
        if not self.initialized:
            raise OperationError("Оркестратор не инициализирован")
        
        return await self.command_processor.process_command(command, *args, **kwargs)
    
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """
        Регистрирует обработчик событий.
        
        Args:
            event_type: Тип события
            handler: Функция-обработчик события
        """
        if not self.initialized:
            self.logger.warning("Оркестратор не инициализирован")
            return
        
        self.event_bus.register_handler(event_type, handler)
    
    async def emit_event(self, event_type: str, data: Any = None) -> None:
        """
        Генерирует событие.
        
        Args:
            event_type: Тип события
            data: Данные события
        """
        if not self.initialized:
            self.logger.warning("Оркестратор не инициализирован")
            return
        
        await self.event_bus.emit(event_type, data)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус системы.
        
        Returns:
            Dict[str, Any]: Словарь с информацией о статусе
        """
        status = {
            "is_running": self.running,
            "is_initialized": self.initialized,
            "current_mode": self.trading_mode_manager.get_current_mode() if self.initialized else None,
            "timestamp": datetime.now().isoformat()
        }
        
        return status
    
    async def _initialize_components(self, config: Dict[str, Any]) -> None:
        """
        Инициализирует компоненты системы на основе конфигурации.
        
        Args:
            config: Конфигурация системы
            
        Raises:
            InitializationError: При ошибке инициализации компонентов
        """
        try:
            # Создаем фабрики компонентов
            exchange_factory = ExchangeFactory(config)
            notification_factory = NotificationFactory(config)
            trading_factory = TradingFactory(config)
            visualization_factory = VisualizationFactory(config)
            ml_factory = MLFactory(config)
            
            # Получаем режим работы
            mode = config["general"]["mode"]
            
            # Инициализируем клиент Binance
            self.binance_client = exchange_factory.create_binance_client(mode)
            
            # Инициализируем интеграцию с Telegram
            self.telegram = await notification_factory.create_telegram_integration()
            
            # Логируем статус Telegram интеграции
            if self.telegram:
                self.logger.info("Telegram интеграция успешно инициализирована")
            else:
                self.logger.warning("Telegram интеграция не инициализирована")
            
            # Создаем стратегию
            self.strategy = trading_factory.create_strategy()
            
            # Создаем контроллер рисков
            self.risk_controller = trading_factory.create_risk_controller()
            
            # Проверяем, нужен ли ML-режим
            use_ml = (config["general"].get("decision_mode") == "ml" or 
                      config["strategy"].get("use_ml", False))
            
            # Создаем компонент принятия решений
            self.decision_maker = await ml_factory.create_decision_maker(
                use_ml, self.strategy, self.risk_controller
            )
            
            # Устанавливаем флаг инициализации
            self.initialized = True
            
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации компонентов: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise InitializationError(f"Ошибка при инициализации компонентов: {str(e)}") from e

    async def _display_menu(self) -> None:
        """
        Отображает главное меню бота.
        """
        from core.constants import LOCALIZATION_KEYS
        
        while self.running:
            print("\n" + "=" * 50)
            print("LEON TRADING BOT - ГЛАВНОЕ МЕНЮ")
            print("=" * 50)
            
            # Получаем локализованные названия режимов
            current_mode = self.trading_mode_manager.get_current_mode()
            mode_name = self.localization_manager.get_text(f"{LOCALIZATION_KEYS['MODE_NAMES']}.{current_mode}")
            
            print(f"Текущий режим: {mode_name}")
            print("\nВыберите опцию:")
            print("1. Изменить режим торговли")
            print("2. Управление моделями ML")
            print("3. Мониторинг и отчеты")
            print("4. Настройки")
            print("0. Выход")
            
            choice = input("\nВаш выбор: ")
            
            if choice == "0":
                await self.stop()
                break
            elif choice.lower() in ["нет", "no", "n", "н"]:
                # Используем локализованное сообщение для ответа "нет"
                no_answer_text = self.localization_manager.get_text(LOCALIZATION_KEYS["NO_ANSWER"])
                print(f"\n😏 {no_answer_text}\n")
                continue
            elif choice == "1":
                await self._display_mode_menu()
            elif choice == "2":
                await self._display_ml_menu()
            elif choice == "3":
                await self._display_monitoring_menu()
            elif choice == "4":
                await self._display_settings_menu()
            else:
                print("\nНеверный выбор. Пожалуйста, попробуйте снова.") 

    def _init_telegram_bot(self):
        """Инициализация Telegram бота."""
        try:
            # Получаем токен и chat_id из конфигурации
            config = self.config_manager.get_config()
            telegram_token = config.get("telegram", {}).get("bot_token", "")
            telegram_chat_id = config.get("telegram", {}).get("chat_id", "")
            telegram_enabled = config.get("telegram", {}).get("enabled", False)
            
            if not telegram_enabled:
                self.logger.info("Telegram интеграция отключена в конфигурации")
                self.telegram_bot = None
                return
                
            if not telegram_token or not telegram_chat_id:
                self.logger.warning("Не указаны токен или chat_id для Telegram бота")
                self.telegram_bot = None
                return
                
            # Инициализируем бота
            from notification.telegram.bot import TelegramBot
            self.telegram_bot = TelegramBot(self.config_manager, self.localization_manager)
            self.telegram_bot.token = telegram_token
            self.telegram_bot.chat_id = telegram_chat_id
            
            # Устанавливаем список разрешенных пользователей (можно добавить в конфигурацию)
            self.telegram_bot.allowed_users = [123456789]  # Замените на реальные ID пользователей
            
            # Устанавливаем ссылку на оркестратор
            self.telegram_bot.set_orchestrator(self)
            
            self.logger.info(f"Telegram бот инициализирован с токеном {telegram_token[:5]}... и chat_id {telegram_chat_id}")
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации Telegram бота: {e}")
            self.logger.debug(traceback.format_exc())
            self.telegram_bot = None

    async def start(self):
        """Запуск системы."""
        if self.running:
            self.logger.warning("Система уже запущена")
            return
        
        try:
            # Сохраняем время начала сессии
            import datetime
            self.start_time = datetime.datetime.now()
            
            # Инициализируем компоненты
            await self._init_components()
            
            # Запускаем компоненты
            await self._start_components()
            
            # Запускаем визуализацию
            await self.visualization_manager.start_visualization()
            
            # Устанавливаем флаг работы
            self.running = True
            
            # Отправляем уведомление в Telegram
            await self._send_telegram_notification()
            
            current_mode = self.trading_mode_manager.get_current_mode()
            self.logger.info(f"Система запущена в режиме: {current_mode}")
        except Exception as e:
            self.logger.error(f"Ошибка при запуске системы: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
    
    async def _send_telegram_notification(self):
        """Отправляет уведомление о запуске системы в Telegram."""
        try:
            if not hasattr(self, 'telegram_bot') or self.telegram_bot is None:
                self.logger.warning("Telegram бот не инициализирован, уведомление не отправлено")
                return
                
            # Получаем параметры из конфигурации
            config = self.config_manager.get_config()
            symbol = config.get("general", {}).get("symbol", "BTCUSDT")
            balance = config.get("general", {}).get("initial_balance", 1000.0)
            leverage = config.get("general", {}).get("leverage", 10)
            risk_per_trade = config.get("strategy", {}).get("params", {}).get("risk_per_trade", 1.0)
            stop_loss = config.get("strategy", {}).get("params", {}).get("stop_loss", 2.0)
            take_profit = config.get("strategy", {}).get("params", {}).get("take_profit", 3.0)
            
            # Формируем сообщение
            current_mode = self.trading_mode_manager.get_current_mode()
            mode_emoji = "🧪" if current_mode == "dry" else "🔥" if current_mode == "real" else "📊"
            message = f"📊 *Статус торгового бота*\n\n"
            message += f"{mode_emoji} Режим: {current_mode.upper() if current_mode else 'Не установлен'}\n"
            message += f"�� Пара: {symbol}\n"
            message += f"💰 Баланс: {balance:.2f} USDT\n"
            message += f"⚡ Плечо: {leverage}x\n"
            message += f"⚠️ Риск на сделку: {risk_per_trade}%\n"
            message += f"🛑 Стоп-лосс: {stop_loss}%\n"
            message += f"🎯 Тейк-профит: {take_profit}%\n"
            
            # Пробуем отправить сообщение напрямую для отладки
            try:
                self.logger.info("Пробуем отправить сообщение напрямую для отладки")
                direct_result = await self.telegram.send_message_direct(message)
                if direct_result:
                    self.logger.info("Сообщение успешно отправлено напрямую")
                else:
                    self.logger.warning("Не удалось отправить сообщение напрямую")
            except Exception as direct_error:
                self.logger.error(f"Ошибка при прямой отправке: {str(direct_error)}")
            
            # Отправляем сообщение обычным способом
            result = await self.telegram.send_message(message)
            if result:
                self.logger.info("Отправлено уведомление о запуске в Telegram")
            else:
                self.logger.warning("Не удалось отправить уведомление в Telegram")
                
        except Exception as e:
            self.logger.error(f"Ошибка при отправке уведомления в Telegram: {str(e)}")
            self.logger.debug(traceback.format_exc())

    async def stop(self):
        """Останавливает работу системы."""
        if not self.initialized:
            self.logger.warning("Попытка остановить неинициализированную систему")
            return
        
        self.logger.info("Останавливаем систему...")
        
        # Отправляем итоги торговой сессии через Telegram
        if hasattr(self, 'telegram_bot') and self.telegram_bot:
            try:
                await self._send_session_summary()
            except Exception as e:
                self.logger.error(f"Ошибка при отправке итогов сессии в Telegram: {e}")
        
        # Останавливаем визуализацию
        if hasattr(self, 'visualization_manager'):
            try:
                await self.visualization_manager.stop_visualization()
            except Exception as e:
                self.logger.error(f"Ошибка при остановке визуализации: {e}")
        
        # Останавливаем все компоненты
        try:
            # Останавливаем стратегию, если она поддерживает метод stop
            if hasattr(self, 'strategy') and self.strategy and hasattr(self.strategy, 'stop'):
                try:
                    await self.strategy.stop()
                    self.logger.info("Стратегия остановлена")
                except Exception as e:
                    self.logger.error(f"Ошибка при остановке стратегии: {str(e)}")
                    self.logger.debug(traceback.format_exc())
            
            # Останавливаем интеграцию с биржей
            if hasattr(self, 'exchange_integration') and self.exchange_integration:
                await self.exchange_integration.stop()
            
            # Закрываем соединение с Binance
            if hasattr(self, 'binance_client') and self.binance_client:
                try:
                    if hasattr(self.binance_client, 'close'):
                        await self.binance_client.close()
                    self.logger.info("Соединение с Binance закрыто")
                except Exception as e:
                    self.logger.error(f"Ошибка при закрытии соединения с Binance: {str(e)}")
                    self.logger.debug(traceback.format_exc())
            
            # Останавливаем Telegram бота
            if hasattr(self, 'telegram_bot') and self.telegram_bot:
                await self.telegram_bot.stop()
            
            # Сбрасываем флаг работы
            self.running = False
            
            self.logger.info("Система успешно остановлена")
        except Exception as e:
            self.logger.error(f"Ошибка при остановке системы: {e}")
            raise

    async def _send_session_summary(self):
        """Отправляет сводку по торговой сессии в Telegram."""
        try:
            if not hasattr(self, 'telegram') or self.telegram is None:
                self.logger.warning("Telegram интеграция не инициализирована, сводка не отправлена")
                return
                
            # Получаем параметры из конфигурации
            config = self.config_manager.get_config()
            symbol = config.get("general", {}).get("symbol", "BTCUSDT")
            
            # Формируем сообщение
            message = f"📊 *Сводка торговой сессии*\n\n"
            message += f"💱 Пара: {symbol}\n"
            
            # Добавляем информацию о времени работы
            if hasattr(self, 'start_time'):
                import datetime
                duration = datetime.datetime.now() - self.start_time
                hours, remainder = divmod(duration.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                message += f"⏱️ Время работы: {hours}ч {minutes}м {seconds}с\n"
            
            # Добавляем статистику торговли, если есть
            if hasattr(self, 'trader') and self.trader:
                try:
                    # Получаем статистику от трейдера
                    stats = await self.trader.get_performance_stats()
                    
                    # Добавляем статистику в сообщение
                    message += f"\n📈 *Статистика торговли:*\n"
                    message += f"💰 P&L: {stats.get('pnl', 0.0):.2f} USDT ({stats.get('pnl_percent', 0.0):.2f}%)\n"
                    message += f"🔄 Сделок: {stats.get('trades_count', 0)}\n"
                    message += f"✅ Успешных: {stats.get('winning_trades', 0)}\n"
                    message += f"❌ Убыточных: {stats.get('losing_trades', 0)}\n"
                    
                    # Добавляем винрейт, если есть сделки
                    if stats.get('trades_count', 0) > 0:
                        winrate = (stats.get('winning_trades', 0) / stats.get('trades_count', 0)) * 100
                        message += f"🎯 Винрейт: {winrate:.2f}%\n"
                except Exception as e:
                    self.logger.error(f"Ошибка при получении статистики торговли: {str(e)}")
                    message += "\n⚠️ Не удалось получить статистику торговли\n"
            else:
                message += "\n⚠️ Нет данных о торговле\n"
            
            # Отправляем сообщение напрямую для отладки
            try:
                self.logger.info("Отправка сводки по сессии напрямую")
                direct_result = await self.telegram.send_message_direct(message)
                if direct_result:
                    self.logger.info("Сводка по сессии успешно отправлена напрямую")
                else:
                    self.logger.warning("Не удалось отправить сводку по сессии напрямую")
            except Exception as direct_error:
                self.logger.error(f"Ошибка при прямой отправке сводки: {str(direct_error)}")
            
            # Отправляем сообщение обычным способом
            result = await self.telegram.send_message(message)
            if result:
                self.logger.info("Отправлена сводка по торговой сессии в Telegram")
            else:
                self.logger.warning("Не удалось отправить сводку по торговой сессии в Telegram")
                
        except Exception as e:
            self.logger.error(f"Ошибка при отправке сводки по сессии: {str(e)}")
            self.logger.debug(traceback.format_exc())

    async def run_forever(self):
        """
        Запускает бесконечный цикл для поддержания работы бота.
        Этот метод блокирует выполнение до получения сигнала остановки.
        """
        self.logger.info("Бот запущен. Нажмите Ctrl+C для завершения...")
        
        try:
            # Бесконечный цикл для поддержания работы бота
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.logger.info("Задача отменена")
            raise

    async def set_mode(self, mode: str) -> bool:
        """
        Изменение режима работы системы.
        
        Args:
            mode: Новый режим работы
            
        Returns:
            True, если режим успешно изменен, иначе False
        """
        try:
            # Проверка допустимости режима
            if not self.trading_mode_manager.validate_mode(mode):
                self.logger.error(f"Недопустимый режим работы: {mode}")
                return False
                
            # Остановка текущих компонентов
            await self.stop()
            
            # Установка нового режима
            self.trading_mode_manager.current_mode = mode
            self.logger.info(f"Режим работы изменен на: {mode}")
            
            # Запуск системы в новом режиме
            await self.start()
            
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при изменении режима работы: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return False

    async def get_status(self) -> str:
        """
        Получение текущего статуса системы.
        
        Returns:
            Текстовое представление статуса
        """
        status = []
        status.append(f"Статус: {'Запущена' if self.running else 'Остановлена'}")
        status.append(f"Режим: {self._get_mode_display()}")
        
        if self.trader:
            balance = await self.get_balance()
            status.append(f"Баланс: {balance:.2f} USDT")
            
            positions = await self.get_positions()
            status.append(f"Открытых позиций: {len(positions)}")
        
        return "\n".join(status)

    async def get_balance(self) -> float:
        """
        Получение текущего баланса.
        
        Returns:
            float: Текущий баланс
        """
        try:
            if hasattr(self, 'binance_client') and self.binance_client:
                return await self.binance_client.get_balance()
            return 0.0
        except Exception as e:
            self.logger.error(f"Ошибка при получении баланса: {e}")
            return 0.0

    async def get_positions(self) -> list:
        """
        Получение открытых позиций.
        
        Returns:
            Список открытых позиций
        """
        if not self.trader:
            self.logger.warning("Трейдер не инициализирован")
            return []
        
        try:
            positions = await self.trader.get_positions()
            return positions
        except Exception as e:
            self.logger.error(f"Ошибка при получении позиций: {e}")
            return []

    async def open_position(self, direction: str) -> dict:
        """
        Открытие позиции.
        
        Args:
            direction: Направление сделки (BUY/SELL)
            
        Returns:
            Результат операции
        """
        if not self.trader:
            self.logger.warning("Трейдер не инициализирован")
            return {"success": False, "error": "Трейдер не инициализирован"}
        
        try:
            # Получение текущего символа
            symbol = self.config_manager.get_value("TRADING_SYMBOL", "BTCUSDT")
            
            # Открытие позиции
            result = await self.trader.enter_position(symbol, direction)
            
            if result.get("success"):
                # Отправка уведомления
                if self.telegram_bot:
                    await self.telegram_bot.send_trade_notification(
                        symbol=symbol,
                        direction=direction,
                        price=result.get("price"),
                        size=result.get("size"),
                        is_open=True
                    )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка при открытии позиции: {e}")
            return {"success": False, "error": str(e)}

    async def close_all_positions(self) -> dict:
        """
        Закрытие всех позиций.
        
        Returns:
            Результат операции
        """
        if not self.trader:
            self.logger.warning("Трейдер не инициализирован")
            return {"success": False, "error": "Трейдер не инициализирован"}
        
        try:
            # Получение открытых позиций
            positions = await self.trader.get_positions()
            
            if not positions:
                return {"success": True, "count": 0, "total_pnl": 0.0}
            
            # Закрытие позиций
            total_pnl = 0.0
            for pos in positions:
                result = await self.trader.exit_position(
                    pos["symbol"],
                    pos["direction"]
                )
                
                if result.get("success"):
                    total_pnl += result.get("pnl", 0.0)
                    
                    # Отправка уведомления
                    if self.telegram_bot:
                        await self.telegram_bot.send_trade_notification(
                            symbol=pos["symbol"],
                            direction="SELL" if pos["direction"] == "BUY" else "BUY",
                            price=result.get("price"),
                            size=pos["size"],
                            pnl=result.get("pnl"),
                            is_open=False
                        )
            
            return {
                "success": True,
                "count": len(positions),
                "total_pnl": total_pnl
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при закрытии позиций: {e}")
            return {"success": False, "error": str(e)}

    def _get_mode_display(self) -> str:
        """
        Получение отображаемого имени режима.
        
        Returns:
            str: Отображаемое имя режима
        """
        # Этот метод заменен на get_formatted_mode
        return self.get_formatted_mode()

    async def _send_status_update(self):
        """Отправляет обновление статуса через Telegram."""
        if not hasattr(self, 'telegram_bot') or not self.telegram_bot:
            self.logger.warning("Невозможно отправить статус: Telegram бот не инициализирован")
            return
        
        try:
            config = self.config_manager.get_config()
            symbol = config["general"]["symbol"]
            mode = self.trading_mode_manager.get_current_mode()
            balance = config["general"]["initial_balance"]
            leverage = config["general"]["leverage"]
            risk_per_trade = config["risk"]["max_position_size"]
            stop_loss = config["risk"]["max_loss_percent"]
            take_profit = config["risk"].get("take_profit_multiplier", 2.0)
            
            # Получаем случайную юмористическую фразу
            humor_phrases = [
                "Ваш электронный финансовый самоубийца на связи!",
                "Продолжаем терять деньги с улыбкой!",
                "Кто сказал, что деньги не растут на деревьях? У нас они вообще испаряются!",
                "Инвестиции — это способ сохранить деньги... у брокера!"
            ]
            
            import random
            humor_phrase = random.choice(humor_phrases)
            
            # Формируем сообщение о статусе с текстовыми таблицами
            status_message = f"*StableTrade*\n🤖 *Leon Trading Bot*\n\n"
            
            # Добавляем юмористическую фразу
            status_message += f"_{humor_phrase}_\n\n"
            
            # Таблица: Основные параметры
            status_message += "```\n"
            status_message += "╔══════════════════════════════════════════════╗\n"
            status_message += f"║ [БАЛАНС] {balance:.2f} USDT │ [ПАРА] {symbol}".ljust(49) + "║\n"
            status_message += f"║ [РЕЖИМ] {self._get_mode_display()} │ [ПЛЕЧО] {leverage}x".ljust(49) + "║\n"
            status_message += f"║ [РИСК] {risk_per_trade} USDT │ [СТОП-ЛОСС] {stop_loss}%".ljust(49) + "║\n"
            status_message += f"║ [ТЕЙК-ПРОФИТ] {take_profit}x".ljust(49) + "║\n"
            status_message += "╚══════════════════════════════════════════════╝\n"
            status_message += "```\n\n"
            
            # Таблица: Последние цены
            recent_prices = self.get_recent_prices(limit=4)
            if recent_prices:
                import datetime
                now = datetime.datetime.now().strftime("%H:%M:%S")
                
                status_message += "```\n"
                status_message += "╔══════════════════════════════════════════════╗\n"
                status_message += f"║ ПОСЛЕДНИЕ ЦЕНЫ ([ОБНОВЛЕНО] {now})".ljust(49) + "║\n"
                status_message += "╠══════════════════════════════════════════════╣\n"
                
                for i in range(0, len(recent_prices), 2):
                    line = "║ "
                    for j in range(2):
                        if i + j < len(recent_prices):
                            price = recent_prices[i + j]
                            price_value = price.get("price", 0)
                            prev_price = recent_prices[i + j - 1].get("price", price_value) if i + j > 0 else price_value
                            
                            direction = "▲" if price_value >= prev_price else "▼"
                            color = "[ЗЕЛЕНЫЙ]" if price_value >= prev_price else "[КРАСНЫЙ]"
                            
                            line += f"{color} {price_value:.2f} {direction} │ "
                    
                    line = line.rstrip("│ ").ljust(48) + "║\n"
                    status_message += line
                
                status_message += "╚══════════════════════════════════════════════╝\n"
                status_message += "```\n\n"
            
            # Таблица: Результаты и позиции
            positions = []
            profit_loss = 0
            profit_loss_percent = 0
            
            try:
                if hasattr(self, 'trader') and self.trader:
                    positions = await self.trader.get_positions()
                    stats = await self.trader.get_performance_stats()
                    if stats:
                        profit_loss = stats.get("profit_loss", 0)
                        profit_loss_percent = stats.get("profit_loss_percent", 0)
            except Exception as e:
                self.logger.error(f"Ошибка при получении позиций или статистики: {e}")
            
            color = "[ЗЕЛЕНЫЙ]" if profit_loss >= 0 else "[КРАСНЫЙ]"
            
            status_message += "```\n"
            status_message += "╔══════════════════════════════════════════════╗\n"
            status_message += f"║ [P&L] {color} {profit_loss:.2f} USDT ({profit_loss_percent:.2f}%)".ljust(49) + "║\n"
            status_message += "╠══════════════════════════════════════════════╣\n"
            
            if positions:
                status_message += "║ ОТКРЫТЫЕ ПОЗИЦИИ:".ljust(49) + "║\n"
                for pos in positions:
                    direction = pos.get("direction", "UNKNOWN")
                    symbol = pos.get("symbol", "UNKNOWN")
                    size = pos.get("size", 0)
                    entry_price = pos.get("entry_price", 0)
                    current_price = pos.get("current_price", 0)
                    pos_pnl = pos.get("pnl", 0)
                    pos_pnl_percent = pos.get("pnl_percent", 0)
                    
                    dir_color = "[ЗЕЛЕНЫЙ]" if direction == "LONG" else "[КРАСНЫЙ]"
                    pnl_color = "[ЗЕЛЕНЫЙ]" if pos_pnl >= 0 else "[КРАСНЫЙ]"
                    
                    status_message += f"║ {dir_color} {direction} {symbol} | Размер: {size:.2f}".ljust(49) + "║\n"
                    status_message += f"║ Вход: {entry_price:.2f} | Текущая: {current_price:.2f}".ljust(49) + "║\n"
                    status_message += f"║ P&L: {pnl_color} {pos_pnl:.2f} USDT ({pos_pnl_percent:.2f}%)".ljust(49) + "║\n"
            else:
                status_message += "║ ОТКРЫТЫЕ ПОЗИЦИИ: [НЕТ]".ljust(49) + "║\n"
            
            status_message += "╚══════════════════════════════════════════════╝\n"
            status_message += "```\n\n"
            
            # Таблица: Сигналы
            signals = self.get_signals(limit=1)
            if signals:
                status_message += "```\n"
                status_message += "╔══════════════════════════════════════════════╗\n"
                
                signal = signals[0]
                action = signal.get("action", "UNKNOWN")
                confidence = signal.get("confidence", 0)
                
                action_color = "[ЗЕЛЕНЫЙ]" if action == "BUY" else "[КРАСНЫЙ]"
                
                status_message += f"║ СИГНАЛЫ: {action_color} {action}".ljust(49) + "║\n"
                status_message += "╠══════════════════════════════════════════════╣\n"
                status_message += f"║ Уверенность: {int(confidence * 100)}% ({confidence:.2f})".ljust(49) + "║\n"
                status_message += "╚══════════════════════════════════════════════╝\n"
                status_message += "```\n\n"
            
            # Таблица: Индикаторы
            indicators = self.get_indicators()
            if indicators:
                status_message += "```\n"
                status_message += "╔══════════════════════════════════════════════╗\n"
                status_message += "║ ИНДИКАТОРЫ:".ljust(49) + "║\n"
                status_message += "╠══════════════════════════════════════════════╣\n"
                
                if "rsi" in indicators:
                    rsi = indicators["rsi"]
                    rsi_status = "[ПЕРЕКУПЛЕННОСТЬ]" if rsi > 70 else "[ПЕРЕПРОДАННОСТЬ]" if rsi < 30 else ""
                    status_message += f"║ RSI: {rsi:.2f} {rsi_status}".ljust(49) + "║\n"
                
                if "macd" in indicators:
                    macd = indicators["macd"]
                    macd_signal = indicators.get("macd_signal", 0)
                    macd_status = "[БЫЧИЙ]" if macd > macd_signal else "[МЕДВЕЖИЙ]"
                    status_message += f"║ MACD: {macd:.2f} {macd_status}".ljust(49) + "║\n"
                
                if "bb_upper" in indicators and "bb_middle" in indicators and "bb_lower" in indicators:
                    bb_upper = indicators["bb_upper"]
                    bb_middle = indicators["bb_middle"]
                    bb_lower = indicators["bb_lower"]
                    
                    status_message += "║ Bollinger Bands:".ljust(49) + "║\n"
                    status_message += f"║ Нижняя: {bb_lower:.2f} │ Средняя: {bb_middle:.2f}".ljust(49) + "║\n"
                    status_message += f"║ Верхняя: {bb_upper:.2f}".ljust(49) + "║\n"
                
                status_message += "╚══════════════════════════════════════════════╝\n"
                status_message += "```\n\n"
            
            # Таблица: Управление
            status_message += "```\n"
            status_message += "╔══════════════════════════════════════════════╗\n"
            status_message += "║ [ОБНОВЛЕНИЕ] каждые 5 сек.".ljust(49) + "║\n"
            status_message += "║ [УПРАВЛЕНИЕ] через кнопки ниже".ljust(49) + "║\n"
            status_message += "╚══════════════════════════════════════════════╝\n"
            status_message += "```\n"
            
            # Отправляем сообщение с кнопками управления
            keyboard = [
                [
                    {"text": "📊 Статус", "callback_data": "status"},
                    {"text": "📈 Торговля", "callback_data": "trade"}
                ],
                [
                    {"text": "⏸️ Пауза", "callback_data": "pause_bot"},
                    {"text": "▶️ Продолжить", "callback_data": "resume_bot"}
                ],
                [
                    {"text": "⚙️ Настройки", "callback_data": "settings"},
                    {"text": "❌ Остановить", "callback_data": "stop_bot"}
                ]
            ]
            
            # Отправляем сообщение с клавиатурой
            await self.telegram_bot.send_message_with_keyboard(status_message, keyboard)
            
            self.logger.debug("Статус системы отправлен в Telegram")
        except Exception as e:
            self.logger.error(f"Ошибка при отправке статуса в Telegram: {e}")
            self.logger.debug(traceback.format_exc())

    async def _init_components(self) -> None:
        """Инициализирует компоненты системы."""
        config = self.config
        
        # Компоненты инициализируются в конструкторе класса
        pass

    async def _start_components(self) -> None:
        """Запуск компонентов системы."""
        # Запуск компонентов в зависимости от текущего режима
        current_mode = self.trading_mode_manager.get_current_mode()
        if current_mode:
            await self._start_components_for_mode(current_mode)
        else:
            self.logger.warning("Не удалось запустить компоненты: режим работы не установлен")
            
    async def _start_components_for_mode(self, mode: str) -> None:
        """Запуск компонентов для указанного режима."""
        if mode == 'dry':
            await self._start_dry_mode()
        elif mode == 'backtest':
            await self._start_backtest_mode()
        elif mode == 'real':
            await self._start_real_mode()

    async def _start_dry_mode(self) -> None:
        """Запуск режима сухого тестирования."""
        # Реализация запуска режима сухого тестирования
        pass

    async def _start_backtest_mode(self) -> None:
        """Запуск режима бэктестирования."""
        # Реализация запуска режима бэктестирования
        pass

    async def _start_real_mode(self) -> None:
        """Запуск режима реальной торговли."""
        # Реализация запуска режима реальной торговли
        pass

    async def _stop_components(self) -> None:
        """Останавливает компоненты системы."""
        try:
            # Останавливаем стратегию, если она поддерживает метод stop
            if hasattr(self, 'strategy') and self.strategy and hasattr(self.strategy, 'stop'):
                try:
                    await self.strategy.stop()
                    self.logger.info("Стратегия остановлена")
                except Exception as e:
                    self.logger.error(f"Ошибка при остановке стратегии: {str(e)}")
                    self.logger.debug(traceback.format_exc())
            
            # Останавливаем интеграцию с биржей
            if hasattr(self, 'exchange_integration') and self.exchange_integration:
                await self.exchange_integration.stop()
            
            # Закрываем соединение с Binance
            if hasattr(self, 'binance_client') and self.binance_client:
                try:
                    if hasattr(self.binance_client, 'close'):
                        await self.binance_client.close()
                    self.logger.info("Соединение с Binance закрыто")
                except Exception as e:
                    self.logger.error(f"Ошибка при закрытии соединения с Binance: {str(e)}")
                    self.logger.debug(traceback.format_exc())
            
            # Останавливаем Telegram бота
            if hasattr(self, 'telegram_bot') and self.telegram_bot:
                try:
                    if hasattr(self.telegram_bot, 'stop'):
                        await self.telegram_bot.stop()
                    self.logger.info("Telegram бот остановлен")
                except Exception as e:
                    self.logger.error(f"Ошибка при остановке Telegram бота: {str(e)}")
                    self.logger.debug(traceback.format_exc())
            
            # Отправляем сводку по сессии
            await self._send_session_summary()
            
        except Exception as e:
            self.logger.error(f"Ошибка при остановке компонентов: {str(e)}")
            self.logger.debug(traceback.format_exc())

    async def pause(self) -> bool:
        """
        Приостанавливает работу бота без полной остановки.
        
        Returns:
            bool: Успешность приостановки
        """
        if not self.running:
            self.logger.warning("Оркестратор не запущен")
            return False
            
        if self.paused:
            self.logger.warning("Оркестратор уже приостановлен")
            return True
            
        try:
            self.logger.info("Приостановка работы оркестратора")
            
            # Приостанавливаем торговлю
            if hasattr(self, 'trader') and self.trader:
                await self.trader.pause()
                
            # Устанавливаем флаг паузы
            self.paused = True
            
            # Генерируем событие о приостановке
            await self.event_bus.emit("orchestrator_paused", {
                "timestamp": datetime.now().isoformat()
            })
            
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при приостановке оркестратора: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return False
    
    async def resume(self) -> bool:
        """
        Возобновляет работу бота после паузы.
        
        Returns:
            bool: Успешность возобновления
        """
        if not self.running:
            self.logger.warning("Оркестратор не запущен")
            return False
            
        if not self.paused:
            self.logger.warning("Оркестратор не приостановлен")
            return True
            
        try:
            self.logger.info("Возобновление работы оркестратора")
            
            # Возобновляем торговлю
            if hasattr(self, 'trader') and self.trader:
                await self.trader.resume()
                
            # Сбрасываем флаг паузы
            self.paused = False
            
            # Генерируем событие о возобновлении
            await self.event_bus.emit("orchestrator_resumed", {
                "timestamp": datetime.now().isoformat()
            })
            
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при возобновлении работы оркестратора: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return False 

    def get_visualizer(self):
        """
        Получить визуализатор.
        
        Returns:
            Визуализатор или None, если он не инициализирован
        """
        if hasattr(self, 'visualization_manager') and self.visualization_manager:
            # Возвращаем основной визуализатор
            return self.visualization_manager.get_visualizer('console')
        return None
    
    def get_trader(self) -> Optional[Any]:
        """
        Получает экземпляр трейдера.
        
        Returns:
            Экземпляр трейдера или None, если трейдер не инициализирован
        """
        if hasattr(self, 'trader') and self.trader:
            return self.trader
        return None
    
    async def add_signal(self, signal: Dict[str, Any]):
        """
        Добавляет новый торговый сигнал.
        
        Args:
            signal: Информация о сигнале (должна содержать ключи 'action', 'confidence', 'timestamp')
        """
        with self._data_lock:
            if not hasattr(self, '_signals'):
                self._signals = []
                
            # Добавляем сигнал в начало списка
            self._signals.insert(0, signal)
            
            # Ограничиваем размер списка
            max_signals = 10
            if len(self._signals) > max_signals:
                self._signals = self._signals[:max_signals]
                
        self.logger.debug(f"Добавлен новый сигнал: {signal}")
        
        # Обновляем визуализацию, если доступна
        if hasattr(self, 'visualization_manager') and self.visualization_manager:
            await self.visualization_manager.update()
    
    async def update_indicators(self, new_indicators: Dict[str, Any]):
        """
        Обновляет значения индикаторов.
        
        Args:
            new_indicators: Словарь с новыми значениями индикаторов
        """
        with self._data_lock:
            if not hasattr(self, '_indicators'):
                self._indicators = {
                    "rsi": 50.0,
                    "macd": 0.0,
                    "macd_signal": 0.0,
                    "bb_upper": 0.0,
                    "bb_middle": 0.0,
                    "bb_lower": 0.0
                }
                
            self._indicators.update(new_indicators)
        self.logger.debug(f"Индикаторы обновлены: {new_indicators}")
        
        # Обновляем визуализацию, если доступна
        if hasattr(self, 'visualization_manager') and self.visualization_manager:
            await self.visualization_manager.update()
    
    async def update_price(self, symbol: str, price: float):
        """
        Обновляет текущую цену.
        
        Args:
            symbol: Символ торговой пары
            price: Новая цена
        """
        if not hasattr(self, '_prices'):
            self._prices = []
            
        # Добавляем новую цену в начало списка
        self._prices.insert(0, price)
        
        # Ограничиваем размер списка
        max_prices = 20
        if len(self._prices) > max_prices:
            self._prices = self._prices[:max_prices]
            
        self.logger.debug(f"Цена обновлена: {symbol} = {price}")
        
        # Обновляем визуализацию, если доступна
        if hasattr(self, 'visualization_manager') and self.visualization_manager:
            await self.visualization_manager.update()

    def is_running(self):
        """
        Проверить, запущена ли система.
        
        Returns:
            bool: True, если система запущена, иначе False
        """
        return self.running
    
    def get_recent_prices(self, limit: int = 5):
        """
        Получает последние цены.
        
        Args:
            limit: Максимальное количество цен для возврата
            
        Returns:
            Список последних цен
        """
        if not hasattr(self, '_prices') or not self._prices:
            self._prices = []
            
            # Добавляем тестовые данные, если список пуст
            if not self._prices:
                import random
                base_price = 3000.0
                for i in range(10):
                    price = base_price + random.uniform(-50, 50)
                    self._prices.append(price)
            
        # Возвращаем последние цены
        return self._prices[:limit]
    
    def get_indicators(self) -> Dict[str, Any]:
        """
        Получает значения индикаторов для визуализации.
        
        Returns:
            Словарь с индикаторами
        """
        try:
            with self._data_lock:
                # Получаем индикаторы из стратегии
                if hasattr(self, 'strategy') and self.strategy:
                    indicators = self.strategy.get_indicators()
                    return indicators.copy() if isinstance(indicators, dict) else {}
                    
                # Если стратегия не инициализирована или индикаторы пусты
                if not hasattr(self, '_indicators') or not self._indicators:
                    # Делегируем инициализацию данных в VisualizationManager
                    if hasattr(self, 'visualization_manager') and self.visualization_manager:
                        self.visualization_manager._ensure_visualization_data()
                
                # Возвращаем копию словаря
                return self._indicators.copy()
        except Exception as e:
            self.logger.warning(f"Ошибка при получении индикаторов: {str(e)}")
            return {}
    
    def get_signals(self) -> List[Dict[str, Any]]:
        """
        Получает сигналы для визуализации.
        
        Returns:
            Список сигналов
        """
        try:
            with self._data_lock:
                # Получаем сигналы из стратегии
                if hasattr(self, 'strategy') and self.strategy:
                    signals = self.strategy.get_signals()
                    return signals.copy() if isinstance(signals, list) else []
                    
                # Если стратегия не инициализирована или сигналы пусты
                if not hasattr(self, '_signals') or not self._signals:
                    # Делегируем инициализацию данных в VisualizationManager
                    if hasattr(self, 'visualization_manager') and self.visualization_manager:
                        self.visualization_manager._ensure_visualization_data()
                
                # Возвращаем копию списка
                return self._signals.copy()
        except Exception as e:
            self.logger.warning(f"Ошибка при получении сигналов: {str(e)}")
            return []

    async def change_mode(self, mode: str) -> bool:
        """
        Изменение режима работы системы.
        
        Args:
            mode: Новый режим работы
            
        Returns:
            bool: Успешность изменения режима
        """
        try:
            # Проверка допустимости режима
            if not self.trading_mode_manager.validate_mode(mode):
                self.logger.error(f"Недопустимый режим работы: {mode}")
                return False
                
            # Остановка текущих компонентов
            await self.stop()
            
            # Установка нового режима
            self.trading_mode_manager.current_mode = mode
            self.logger.info(f"Режим работы изменен на: {mode}")
            
            # Запуск системы в новом режиме
            await self.start()
            
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при изменении режима работы: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return False

    def get_formatted_mode(self) -> str:
        """
        Получение форматированного режима работы.
        
        Returns:
            str: Форматированный режим работы
        """
        mode_map = {
            TradingModes.DRY: "Симуляция (Dry Mode)",
            TradingModes.BACKTEST: "Бэктестинг",
            TradingModes.REAL: "Реальная торговля"
        }
        current_mode = self.trading_mode_manager.get_current_mode()
        return mode_map.get(current_mode, current_mode)

    def get_formatted_trading_mode(self) -> str:
        """
        Получение форматированного режима торговли.
        
        Returns:
            str: Форматированный режим торговли
        """
        # Этот метод заменен на get_formatted_mode
        return self.get_formatted_mode()