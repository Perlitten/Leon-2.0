"""
Модуль фабрики компонентов для Leon Trading Bot.

Этот модуль предоставляет классы для создания различных компонентов системы,
таких как клиенты бирж, интеграции с Telegram, стратегии, контроллеры рисков и т.д.
"""

import logging
import traceback
from typing import Dict, Any, Optional
from datetime import datetime

from core.exceptions import InitializationError
from core.utils import mask_sensitive_data


class ComponentFactory:
    """
    Базовый класс фабрики компонентов.
    
    Предоставляет общие методы для создания компонентов системы.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация фабрики компонентов.
        
        Args:
            config: Конфигурация системы
        """
        self.config = config
        self.logger = logging.getLogger("ComponentFactory")


class ExchangeFactory(ComponentFactory):
    """
    Фабрика для создания клиентов бирж.
    """
    
    def create_binance_client(self, mode: str) -> Any:
        """
        Создает и инициализирует клиент Binance API.
        
        Args:
            mode: Режим работы системы ("dry", "real", "backtest")
            
        Returns:
            BinanceIntegration: Инициализированный клиент
            
        Raises:
            InitializationError: При ошибке инициализации клиента
        """
        try:
            api_key = self.config["binance"]["api_key"]
            api_secret = self.config["binance"]["api_secret"]
            testnet = self.config["binance"]["testnet"]
            
            # Для режимов симуляции и бэктестинга можно использовать тестовую сеть или публичный API
            if mode in ["dry", "backtest"] and not api_key:
                # Упрощенная инициализация для режимов, не требующих API ключей
                self.logger.info("Инициализация клиента Binance для неторговых операций")
                from exchange.binance.client import BinanceIntegration
                return BinanceIntegration("", "", testnet=testnet)
            
            if not api_key or not api_secret:
                raise ValueError("Не предоставлены ключи Binance API")
            
            # Логирование с маскированным ключом API для безопасности
            self.logger.info(f"Инициализация клиента Binance (testnet: {testnet}) с ключом: {mask_sensitive_data(api_key)}")
            from exchange.binance.client import BinanceIntegration
            return BinanceIntegration(api_key, api_secret, testnet=testnet)
        except Exception as e:
            self.logger.error(f"Ошибка при создании клиента Binance: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise InitializationError(f"Ошибка при создании клиента Binance: {str(e)}", component="BinanceClient") from e


class NotificationFactory(ComponentFactory):
    """
    Фабрика для создания компонентов уведомлений.
    """
    
    def create_telegram_integration(self) -> Optional[Any]:
        """
        Создает интеграцию с Telegram, если предоставлены необходимые параметры.
        
        Returns:
            TelegramIntegration или None: Интеграция или None, если не настроена
            
        Raises:
            InitializationError: При ошибке инициализации интеграции
        """
        try:
            bot_token = self.config["telegram"]["bot_token"]
            chat_id = self.config["telegram"]["chat_id"]
            
            if not bot_token or not chat_id:
                self.logger.info("Telegram интеграция не настроена, уведомления отключены")
                return None
            
            # Логирование с маскированным токеном для безопасности
            self.logger.info(f"Инициализация Telegram интеграции с токеном: {mask_sensitive_data(bot_token)}")
            
            from notification.telegram_bot import TelegramIntegration
            telegram = TelegramIntegration(bot_token, chat_id)
            
            # Проверяем, что интеграция успешно подключилась
            if not telegram.connected:
                self.logger.error("Не удалось подключиться к Telegram API. Проверьте токен и подключение к интернету.")
                return None
                
            return telegram
        except Exception as e:
            self.logger.error(f"Ошибка при создании Telegram интеграции: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise InitializationError(f"Ошибка при создании Telegram интеграции: {str(e)}", component="TelegramIntegration") from e


class TradingFactory(ComponentFactory):
    """
    Фабрика для создания торговых компонентов.
    """
    
    def create_strategy(self) -> Any:
        """
        Создает экземпляр торговой стратегии с параметрами из конфигурации.
        
        Returns:
            object: Экземпляр стратегии
            
        Raises:
            InitializationError: При ошибке создания стратегии
        """
        try:
            strategy_config = self.config["strategy"]
            
            self.logger.info("Инициализация торговой стратегии")
            
            # Создаем конфигурацию для стратегии
            from trading.strategies.base import StrategyConfig
            config = StrategyConfig(
                stop_loss=strategy_config["stop_loss"],
                take_profit=strategy_config["take_profit"],
                risk_per_trade=strategy_config["risk_per_trade"],
                use_trailing_stop=strategy_config["use_trailing_stop"],
                trailing_stop_activation=strategy_config["trailing_stop_activation"]
            )
            
            # Создаем стратегию с правильными аргументами
            from trading.strategies.scalping import ScalpingStrategy
            strategy = ScalpingStrategy(
                symbol=self.config["general"]["symbol"],
                config=config
            )
            
            return strategy
        except Exception as e:
            self.logger.error(f"Ошибка при создании торговой стратегии: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise InitializationError(f"Ошибка при создании торговой стратегии: {str(e)}", component="TradingStrategy") from e
    
    def create_risk_controller(self) -> Any:
        """
        Создает контроллер управления рисками.
        
        Returns:
            RiskController: Экземпляр контроллера рисков
            
        Raises:
            InitializationError: При ошибке создания контроллера рисков
        """
        try:
            self.logger.info("Инициализация контроллера рисков")
            
            # Получаем параметры риска из конфигурации
            risk_params = {
                "max_daily_loss": self.config["safety"]["max_daily_loss"],
                "max_daily_trades": self.config["safety"]["max_daily_trades"]
            }
            
            # Создаем и возвращаем контроллер рисков
            from trading.risk.risk_manager import create_risk_controller
            return create_risk_controller(**risk_params)
        except Exception as e:
            self.logger.error(f"Ошибка при создании контроллера рисков: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise InitializationError(f"Ошибка при создании контроллера рисков: {str(e)}", component="RiskController") from e
    
    def create_trader(self, mode: str, **kwargs) -> Any:
        """
        Создает трейдер в соответствии с указанным режимом.
        
        Args:
            mode: Режим работы ("dry", "real", "backtest")
            **kwargs: Дополнительные параметры для трейдера
            
        Returns:
            object: Экземпляр трейдера
            
        Raises:
            InitializationError: При ошибке создания трейдера
        """
        try:
            self.logger.info(f"Создание трейдера для режима: {mode}")
            
            if mode == "dry":
                from trading.traders.dry_trader import DryModeTrader
                trader_class = DryModeTrader
            elif mode == "real":
                from trading.traders.real_trader import RealTrader
                trader_class = RealTrader
            elif mode == "backtest":
                from trading.traders.backtest_trader import BacktestTrader
                trader_class = BacktestTrader
            else:
                raise ValueError(f"Неизвестный режим работы: {mode}")
            
            # Создаем трейдер с соответствующими параметрами
            trader = trader_class(**kwargs)
            
            return trader
        except Exception as e:
            self.logger.error(f"Ошибка при создании трейдера для режима {mode}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise InitializationError(f"Ошибка при создании трейдера: {str(e)}", component=f"{mode.capitalize()}Trader") from e


class VisualizationFactory(ComponentFactory):
    """
    Фабрика для создания компонентов визуализации.
    """
    
    def create_visualizer(self) -> Any:
        """
        Создает консольный визуализатор.
        
        Returns:
            ConsoleVisualizer: Экземпляр визуализатора
            
        Raises:
            InitializationError: При ошибке создания визуализатора
        """
        try:
            self.logger.info("Инициализация консольной визуализации")
            
            symbol = self.config["general"]["symbol"]
            initial_balance = self.config["general"]["initial_balance"]
            leverage = self.config["general"]["leverage"]
            risk_per_trade = self.config["strategy"]["risk_per_trade"]
            stop_loss = self.config["strategy"]["stop_loss"]
            take_profit = self.config["strategy"]["take_profit"]
            
            from visualization.console_ui import ConsoleVisualizer
            visualizer = ConsoleVisualizer(
                symbol=symbol,
                initial_balance=initial_balance,
                leverage=leverage,
                risk_per_trade=risk_per_trade,
                stop_loss=stop_loss, 
                take_profit=take_profit
            )
            
            # Добавляем дополнительные параметры из конфигурации
            if "visualization" in self.config:
                viz_config = self.config["visualization"]
                visualizer.use_ascii = viz_config.get("use_ascii", True)
                
                # Устанавливаем язык, если указан в конфигурации
                if "language" in viz_config:
                    language = viz_config.get("language", "en")
                    self.logger.info(f"Установка языка визуализации: {language}")
                    visualizer.set_language(language)
                
                # Добавляем конфигурацию в визуализатор
                visualizer.config = viz_config
            
            # Инициализируем дополнительные атрибуты для ML
            visualizer.ml_predictions = None
            visualizer.ml_metrics = None
            
            return visualizer
        except Exception as e:
            self.logger.error(f"Ошибка при создании визуализатора: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise InitializationError(f"Ошибка при создании визуализатора: {str(e)}", component="ConsoleVisualizer") from e


class MLFactory(ComponentFactory):
    """
    Фабрика для создания компонентов машинного обучения.
    """
    
    async def create_decision_maker(self, use_ml: bool, strategy, risk_controller) -> Any:
        """
        Создает компонент принятия решений.
        
        Args:
            use_ml: Использовать ли ML для принятия решений
            strategy: Экземпляр торговой стратегии
            risk_controller: Экземпляр контроллера рисков
            
        Returns:
            object: Экземпляр компонента принятия решений
            
        Raises:
            InitializationError: При ошибке создания компонента
        """
        try:
            if use_ml:
                self.logger.info("Инициализация ML-компонента принятия решений")
                
                # Здесь должна быть логика создания ML-компонента
                # ...
                
                # Временная заглушка
                from ml.decision_maker import MLDecisionMaker
                decision_maker = MLDecisionMaker(strategy, risk_controller)
                
                # Загрузка модели
                model_path = self.config["ml"].get("model_path", "models/trading_model")
                await decision_maker.load_model(model_path)
                
                return decision_maker
            else:
                self.logger.info("Инициализация стандартного компонента принятия решений")
                
                from ml.decision_maker import DecisionMaker
                return DecisionMaker(strategy, risk_controller)
        except Exception as e:
            self.logger.error(f"Ошибка при создании компонента принятия решений: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise InitializationError(f"Ошибка при создании компонента принятия решений: {str(e)}", component="DecisionMaker") from e 