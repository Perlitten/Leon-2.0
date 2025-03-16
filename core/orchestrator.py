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
    Менеджер интеграции с ML-моделями.
    
    Отвечает за загрузку, управление и использование ML-моделей.
    """
    
    def __init__(self, orchestrator: 'LeonOrchestrator'):
        """
        Инициализация менеджера интеграции с ML-моделями.
        
        Args:
            orchestrator: Экземпляр оркестратора
        """
        self.orchestrator = orchestrator
        self.models = {}
        self.current_model = None
        self.logger = logging.getLogger("MLIntegrationManager")
    
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


class VisualizationManager:
    """
    Менеджер визуализации.
    
    Отвечает за управление визуализацией данных.
    """
    
    def __init__(self, orchestrator: 'LeonOrchestrator'):
        """
        Инициализация менеджера визуализации.
        
        Args:
            orchestrator: Экземпляр оркестратора
        """
        self.orchestrator = orchestrator
        self.visualizer = None
        self.visualization_task = None
        self.logger = logging.getLogger("VisualizationManager")
    
    async def start_visualization(self) -> None:
        """Запускает визуализацию."""
        config = self.orchestrator.config_manager.get_config()
        
        # Проверяем, включена ли визуализация
        if not config["visualization"]["enabled"]:
            self.logger.info("Визуализация отключена в конфигурации")
            return
        
        # Создаем визуализатор, если он еще не создан
        if not self.visualizer:
            visualization_factory = VisualizationFactory(config)
            self.visualizer = visualization_factory.create_visualizer()
        
        # Запускаем задачу визуализации
        self.logger.info("Запуск визуализации")
        self.visualization_task = asyncio.create_task(self._run_visualization())
    
    async def stop_visualization(self) -> None:
        """Останавливает визуализацию."""
        if self.visualizer:
            self.logger.info("Остановка консольной визуализации")
            
            # Останавливаем визуализатор
            if hasattr(self.visualizer, 'stop_visualization'):
                self.visualizer.stop_visualization()
            
            # Отменяем задачу, если она запущена
            if self.visualization_task and not self.visualization_task.done():
                self.visualization_task.cancel()
                try:
                    await self.visualization_task
                except asyncio.CancelledError:
                    pass
    
    async def _run_visualization(self) -> None:
        """Запускает цикл визуализации."""
        if not self.visualizer:
            return
        
        config = self.orchestrator.config_manager.get_config()
        update_interval = config["visualization"].get("update_interval", 1.0)
        
        try:
            self.logger.info(f"Запуск цикла визуализации с интервалом {update_interval} сек")
            
            # Инициализация визуализатора
            self.visualizer.start_visualization()
            
            # Основной цикл визуализации
            while True:
                # Обновляем визуализацию
                self.visualizer.update()
                
                # Ждем следующего обновления
                await asyncio.sleep(update_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Задача визуализации отменена")
        except Exception as e:
            self.logger.error(f"Ошибка визуализации: {str(e)}")
            self.logger.debug(traceback.format_exc())
        finally:
            # Убеждаемся, что визуализация остановлена
            if self.visualizer:
                self.visualizer.stop_visualization()


class LeonOrchestrator:
    """
    Центральный класс для управления всеми подсистемами Leon Trading Bot.
    
    Отвечает за координацию работы различных компонентов системы,
    управление жизненным циклом, обработку событий и переключение режимов.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", env_file: str = ".env"):
        """
        Инициализирует оркестратор.
        
        Args:
            config_path: Путь к файлу конфигурации
            env_file: Путь к файлу с переменными окружения
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.env_file = env_file
        self.config = {}
        self.running = False
        self.initialized = False
        
        # Инициализация компонентов
        self.event_bus = EventBus()
        self.command_processor = CommandProcessor(self)
        self.trading_mode_manager = TradingModeManager(self)
        self.ml_integration_manager = MLIntegrationManager(self)
        self.visualization_manager = VisualizationManager(self)
        
        # Инициализация менеджера локализации
        from core.localization import get_localization_manager
        self.localization_manager = get_localization_manager()
        
        # Регистрация базовых команд
        self._register_base_commands()
        
        self.logger.info("Оркестратор создан")
        
        # Инициализация Telegram бота
        self._init_telegram_bot()
    
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
            
            # Запуск Telegram бота
            if self.telegram_bot:
                await self.telegram_bot.start()
            
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при запуске оркестратора: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise OperationError(f"Ошибка при запуске оркестратора: {str(e)}", operation="start") from e
    
    async def stop(self) -> bool:
        """
        Останавливает систему и освобождает ресурсы.
        
        Returns:
            bool: Успешность остановки
            
        Raises:
            OperationError: При ошибке остановки
        """
        if not self.running:
            self.logger.warning("Оркестратор не запущен")
            return True
        
        try:
            self.logger.info("Остановка оркестратора")
            
            # Останавливаем текущий режим
            await self.trading_mode_manager._stop_current_mode()
            
            # Останавливаем визуализацию
            await self.visualization_manager.stop_visualization()
            
            # Отправляем уведомление в Telegram о завершении работы
            if self.telegram_bot and hasattr(self.telegram_bot, 'connected') and self.telegram_bot.connected:
                try:
                    await self.telegram_bot.stop()
                    self.logger.info("Отправлено уведомление о завершении работы в Telegram")
                except Exception as e:
                    self.logger.error(f"Ошибка отправки уведомления в Telegram: {str(e)}")
            
            # Закрываем соединение с Binance
            if self.binance_client:
                try:
                    await self.binance_client.close()
                    self.logger.info("Соединение с Binance закрыто")
                except Exception as e:
                    self.logger.error(f"Ошибка при закрытии соединения с Binance: {str(e)}")
            
            # Сбрасываем флаг работы
            self.running = False
            
            # Генерируем событие об остановке
            await self.event_bus.emit("orchestrator_stopped", {
                "timestamp": datetime.now().isoformat()
            })
            
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при остановке оркестратора: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise OperationError(f"Ошибка при остановке оркестратора: {str(e)}", operation="stop") from e
    
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
        Инициализирует компоненты системы с использованием фабрик.
        
        Args:
            config: Конфигурация системы
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
            self.telegram = notification_factory.create_telegram_integration()
            
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
            
            # Создаем визуализатор, если визуализация включена
            if config["visualization"]["enabled"]:
                self.visualizer = visualization_factory.create_visualizer()
                self.visualization_manager.visualizer = self.visualizer 
        except Exception as e:
            error_phrases = self.localization_manager.get_text(LOCALIZATION_KEYS["ERROR_PHRASES"])
            error_phrase = random.choice(error_phrases)
            self.logger.error(f"Ошибка при инициализации компонентов: {str(e)}")
            self.logger.debug(traceback.format_exc())
            print(f"\n⚠️ {error_phrase}")
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
            self.telegram_bot = TelegramBot(self.config, self.localization_manager)
            self.telegram_bot.set_orchestrator(self)
            self.logger.info("Telegram бот инициализирован")
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации Telegram бота: {e}")
            self.telegram_bot = None

    async def start(self):
        """Запуск системы."""
        if self.running:
            logger.warning("Система уже запущена")
            return
        
        try:
            # Инициализация компонентов в зависимости от режима
            await self._init_components()
            
            # Запуск компонентов
            await self._start_components()
            
            # Запуск Telegram бота
            if self.telegram_bot:
                await self.telegram_bot.start()
            
            self.running = True
            logger.info(f"Система запущена в режиме: {self.current_mode}")
            
            # Отправка уведомления о запуске
            if self.telegram_bot:
                await self._send_status_update()
            
        except Exception as e:
            logger.error(f"Ошибка при запуске системы: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Остановка системы."""
        if not self.running:
            logger.warning("Система не запущена")
            return
        
        try:
            # Остановка Telegram бота
            if self.telegram_bot:
                await self.telegram_bot.stop()
            
            # Остановка компонентов
            await self._stop_components()
            
            self.running = False
            logger.info("Система остановлена")
            
        except Exception as e:
            logger.error(f"Ошибка при остановке системы: {e}")
            raise

    async def set_mode(self, mode: str) -> dict:
        """
        Изменение режима работы.
        
        Args:
            mode: Новый режим работы
            
        Returns:
            Результат операции
        """
        if mode not in [TradingModes.DRY, TradingModes.BACKTEST, TradingModes.REAL]:
            self.logger.error(f"Недопустимый режим работы: {mode}")
            return {"success": False, "error": f"Недопустимый режим работы: {mode}"}
        
        if self.current_mode == mode:
            self.logger.info(f"Система уже работает в режиме: {mode}")
            return {"success": True, "mode": mode}
        
        try:
            # Если система запущена, останавливаем ее
            was_running = self.running
            if was_running:
                await self.stop()
            
            # Изменение режима
            self.current_mode = mode
            self.config.set("TRADING_MODE", mode)
            
            # Если система была запущена, запускаем ее снова
            if was_running:
                await self.start()
            
            self.logger.info(f"Режим работы изменен на: {mode}")
            return {"success": True, "mode": mode}
            
        except Exception as e:
            self.logger.error(f"Ошибка при изменении режима работы: {e}")
            return {"success": False, "error": str(e)}

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
            Текущий баланс
        """
        if not self.trader:
            logger.warning("Трейдер не инициализирован")
            return 0.0
        
        try:
            balance = await self.trader.get_balance()
            return balance
        except Exception as e:
            logger.error(f"Ошибка при получении баланса: {e}")
            return 0.0

    async def get_positions(self) -> list:
        """
        Получение открытых позиций.
        
        Returns:
            Список открытых позиций
        """
        if not self.trader:
            logger.warning("Трейдер не инициализирован")
            return []
        
        try:
            positions = await self.trader.get_positions()
            return positions
        except Exception as e:
            logger.error(f"Ошибка при получении позиций: {e}")
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
            logger.warning("Трейдер не инициализирован")
            return {"success": False, "error": "Трейдер не инициализирован"}
        
        try:
            # Получение текущего символа
            symbol = self.config.get("TRADING_SYMBOL", "BTCUSDT")
            
            # Открытие позиции
            result = await self.trader.enter_position(symbol, direction)
            
            if result.get("success"):
                # Отправка уведомления
                if self.telegram_bot:
                    await self.telegram_bot.send_trade_notification(
                        symbol=symbol,
                        direction=direction,
                        price=result.get("price"),
                        size=result.get("size")
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при открытии позиции: {e}")
            return {"success": False, "error": str(e)}

    async def close_all_positions(self) -> dict:
        """
        Закрытие всех позиций.
        
        Returns:
            Результат операции
        """
        if not self.trader:
            logger.warning("Трейдер не инициализирован")
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
                            pnl=result.get("pnl")
                        )
            
            return {
                "success": True,
                "count": len(positions),
                "total_pnl": total_pnl
            }
            
        except Exception as e:
            logger.error(f"Ошибка при закрытии позиций: {e}")
            return {"success": False, "error": str(e)}

    def _get_mode_display(self) -> str:
        """
        Получение отображаемого названия режима.
        
        Returns:
            Отображаемое название режима
        """
        mode_map = {
            TradingModes.DRY: "Симуляция (Dry Mode)",
            TradingModes.BACKTEST: "Бэктестинг",
            TradingModes.REAL: "Реальная торговля"
        }
        return mode_map.get(self.current_mode, self.current_mode)

    async def _send_status_update(self):
        """Отправка обновления статуса через Telegram."""
        if not self.telegram_bot:
            return
        
        try:
            # Получение параметров
            symbol = self.config.get("TRADING_SYMBOL", "BTCUSDT")
            balance = await self.get_balance()
            leverage = int(self.config.get("LEVERAGE", 20))
            risk_per_trade = float(self.config.get("RISK_PER_TRADE", 2.0))
            stop_loss = float(self.config.get("STOP_LOSS_PERCENT", 2.0))
            take_profit = float(self.config.get("TAKE_PROFIT_PERCENT", 3.0))
            
            # Отправка обновления
            await self.telegram_bot.send_status_update(
                symbol=symbol,
                mode=self.current_mode,
                balance=balance,
                leverage=leverage,
                risk_per_trade=risk_per_trade,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка при отправке обновления статуса: {e}")

    async def _init_components(self) -> None:
        """Инициализация компонентов системы."""
        await self._initialize_components(self.config)

    async def _start_components(self) -> None:
        """Запуск компонентов системы."""
        # Запуск компонентов в зависимости от текущего режима
        await self._start_components_for_mode(self.current_mode)

    async def _start_components_for_mode(self, mode: str) -> None:
        """Запуск компонентов для указанного режима."""
        if mode == TRADING_MODES.DRY:
            await self._start_dry_mode()
        elif mode == TRADING_MODES.BACKTEST:
            await self._start_backtest_mode()
        elif mode == TRADING_MODES.REAL:
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
        """Остановка компонентов системы."""
        # Остановка компонентов в зависимости от текущего режима
        await self._stop_components_for_mode(self.current_mode)

    async def _stop_components_for_mode(self, mode: str) -> None:
        """Остановка компонентов для указанного режима."""
        if mode == TRADING_MODES.DRY:
            await self._stop_dry_mode()
        elif mode == TRADING_MODES.BACKTEST:
            await self._stop_backtest_mode()
        elif mode == TRADING_MODES.REAL:
            await self._stop_real_mode()

    async def _stop_dry_mode(self) -> None:
        """Остановка режима сухого тестирования."""
        # Реализация остановки режима сухого тестирования
        pass

    async def _stop_backtest_mode(self) -> None:
        """Остановка режима бэктестирования."""
        # Реализация остановки режима бэктестирования
        pass

    async def _stop_real_mode(self) -> None:
        """Остановка режима реальной торговли."""
        # Реализация остановки режима реальной торговли
        pass 