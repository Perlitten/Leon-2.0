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
    
    async def create_telegram_integration(self) -> Any:
        """
        Создает и инициализирует интеграцию с Telegram.
        
        Returns:
            TelegramIntegration: Инициализированная интеграция с Telegram
            
        Raises:
            InitializationError: При ошибке инициализации интеграции
        """
        try:
            # Проверяем, включена ли Telegram интеграция
            if not self.config.get("telegram", {}).get("enabled", False):
                self.logger.info("Telegram интеграция отключена в конфигурации")
                return None
            
            # Получаем токен и chat_id из конфигурации
            bot_token = self.config.get("telegram", {}).get("bot_token", "")
            chat_id = self.config.get("telegram", {}).get("chat_id", "")
            
            if not bot_token or not chat_id:
                self.logger.warning("Не указаны токен или chat_id для Telegram интеграции")
                return None
            
            # Маскируем токен для логирования
            masked_token = mask_sensitive_data(bot_token)
            self.logger.info(f"Инициализация Telegram интеграции с токеном: {masked_token}")
            
            # Создаем интеграцию
            from notification.telegram_bot import TelegramIntegration
            telegram = TelegramIntegration(bot_token, chat_id, None)
            
            # Инициализируем интеграцию
            success = await telegram.initialize()
            
            if not success:
                self.logger.warning("Не удалось инициализировать Telegram интеграцию")
                return None
                
            # Проверяем, что интеграция успешно подключена
            if not telegram.connected:
                self.logger.warning("Telegram интеграция не подключена")
                return None
                
            # Отправляем тестовое сообщение
            test_message = "🤖 *Leon Trading Bot* запускается..."
            await telegram.send_message(test_message)
            self.logger.info("Отправлено тестовое сообщение в Telegram")
            
            return telegram
        except Exception as e:
            self.logger.error(f"Ошибка при создании Telegram интеграции: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None


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
            strategy_params = strategy_config.get("params", {})
            
            self.logger.info("Инициализация торговой стратегии")
            
            # Создаем конфигурацию для стратегии с параметрами по умолчанию, если они не указаны
            from trading.strategies.base import StrategyConfig
            config = StrategyConfig(
                stop_loss=strategy_params.get("stop_loss", 2.0),
                take_profit=strategy_params.get("take_profit", 3.0),
                risk_per_trade=self.config["risk"].get("max_loss_percent", 1.0),
                use_trailing_stop=strategy_params.get("use_trailing_stop", False),
                trailing_stop_activation=strategy_params.get("trailing_stop_activation", 0.5)
            )
            
            # Создаем стратегию с правильными аргументами
            from trading.strategies.simple_ma import SimpleMAStrategy
            strategy = SimpleMAStrategy(
                symbol=self.config["general"]["symbol"],
                timeframe=self.config["general"]["kline_interval"],
                params=strategy_params
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
            max_daily_loss = self.config["safety"].get("max_daily_loss", 5.0)
            max_daily_trades = self.config["safety"].get("max_daily_trades", 10)
            
            # Создаем и возвращаем контроллер рисков
            from trading.risk import create_risk_controller
            risk_controller = create_risk_controller(
                max_daily_loss=max_daily_loss,
                max_daily_trades=max_daily_trades
            )
            
            # Логируем успешное создание
            self.logger.info(f"Контроллер рисков создан успешно: max_daily_loss={max_daily_loss}%, max_daily_trades={max_daily_trades}")
            
            return risk_controller
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
        Создает и возвращает визуализатор для отображения информации о торговле.
        
        Returns:
            ConsoleVisualizer: Экземпляр визуализатора.
        """
        try:
            self.logger.info("Инициализация консольной визуализации")
            
            # Создаем базовую конфигурацию для визуализатора
            viz_config = {}
            if "visualization" in self.config:
                viz_config = self.config["visualization"]
            
            from visualization.console_ui import ConsoleVisualizer
            visualizer = ConsoleVisualizer(name="console", config=viz_config)
            
            # Добавляем дополнительные параметры из конфигурации
            if "visualization" in self.config:
                # Устанавливаем язык, если указан в конфигурации
                if "language" in viz_config:
                    language = viz_config.get("language", "en")
                    self.logger.info(f"Установка языка визуализации: {language}")
                    if hasattr(visualizer, 'set_language'):
                        visualizer.set_language(language)
            
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