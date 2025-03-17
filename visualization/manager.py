"""
Менеджер визуализаторов для Leon Trading Bot.

Предоставляет централизованное управление всеми визуализаторами в системе.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Type
from threading import Thread
import traceback
from datetime import datetime

from visualization.base import BaseVisualizer
from visualization.trading_dashboard import TradingDashboard
from visualization.candle_visualizer import CandleVisualizer
from visualization.web_dashboard import WebDashboard


class VisualizationManager:
    """
    Менеджер визуализаторов.
    
    Отвечает за создание, настройку и управление всеми визуализаторами в системе.
    """
    
    def __init__(self, orchestrator: 'LeonOrchestrator'):
        """
        Инициализация менеджера визуализации.
        
        Args:
            orchestrator: Экземпляр оркестратора
        """
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = orchestrator.config if orchestrator else {}
        
        # Визуализатор
        self.visualizer = None
        
        # Задачи для асинхронного обновления
        self.visualization_task = None
        self.update_task = None
        
        # Словарь доступных визуализаторов
        self.available_visualizers = {
            "trading_dashboard": TradingDashboard,
            "candle_visualizer": CandleVisualizer,
            "web_dashboard": WebDashboard
        }
        
        # Словарь активных визуализаторов
        self.active_visualizers = {}
        
        # Фоновые потоки для визуализаторов
        self.background_threads = {}
        
        # Проверяем наличие блокировки для потокобезопасного доступа к данным
        if not hasattr(self.orchestrator, '_data_lock'):
            import threading
            self._data_lock = threading.Lock()
            self.logger.warning("В оркестраторе отсутствует _data_lock, создан локальный")
        else:
            self._data_lock = self.orchestrator._data_lock
        
        self.logger.info("Менеджер визуализаторов инициализирован")
    
    def create_visualizer(self, visualizer_type: str, name: Optional[str] = None, 
                         config: Optional[Dict[str, Any]] = None) -> Optional[BaseVisualizer]:
        """
        Создание визуализатора.
        
        Args:
            visualizer_type: Тип визуализатора
            name: Имя визуализатора (если None, будет использовано имя по умолчанию)
            config: Конфигурация визуализатора
            
        Returns:
            Созданный визуализатор или None, если тип не найден
        """
        if visualizer_type not in self.available_visualizers:
            self.logger.error(f"Неизвестный тип визуализатора: {visualizer_type}")
            return None
        
        # Получение класса визуализатора
        visualizer_class = self.available_visualizers[visualizer_type]
        
        # Создание визуализатора
        visualizer = visualizer_class(name=name or visualizer_type, config=config)
        
        self.logger.info(f"Создан визуализатор {visualizer.name} типа {visualizer_type}")
        return visualizer
    
    def add_visualizer(self, visualizer: BaseVisualizer) -> bool:
        """
        Добавление визуализатора в список активных.
        
        Args:
            visualizer: Визуализатор
            
        Returns:
            True, если визуализатор успешно добавлен, иначе False
        """
        if visualizer.name in self.active_visualizers:
            self.logger.warning(f"Визуализатор с именем {visualizer.name} уже существует")
            return False
        
        self.active_visualizers[visualizer.name] = visualizer
        self.logger.info(f"Визуализатор {visualizer.name} добавлен в список активных")
        return True
    
    def remove_visualizer(self, name: str) -> bool:
        """
        Удаление визуализатора из списка активных.
        
        Args:
            name: Имя визуализатора
            
        Returns:
            True, если визуализатор успешно удален, иначе False
        """
        if name not in self.active_visualizers:
            self.logger.warning(f"Визуализатор с именем {name} не найден")
            return False
        
        # Остановка визуализатора, если он запущен
        if self.active_visualizers[name].is_running:
            self.stop_visualizer(name)
        
        # Удаление визуализатора из списка активных
        del self.active_visualizers[name]
        
        self.logger.info(f"Визуализатор {name} удален из списка активных")
        return True
    
    def get_visualizer(self, name: str) -> Optional[BaseVisualizer]:
        """
        Получение визуализатора по имени.
        
        Args:
            name: Имя визуализатора
            
        Returns:
            Визуализатор или None, если не найден
        """
        return self.active_visualizers.get(name)
    
    def start_visualizer(self, name: str, background: bool = False) -> bool:
        """
        Запуск визуализатора.
        
        Args:
            name: Имя визуализатора
            background: Запустить в фоновом режиме
            
        Returns:
            True, если визуализатор успешно запущен, иначе False
        """
        visualizer = self.get_visualizer(name)
        if not visualizer:
            self.logger.error(f"Визуализатор с именем {name} не найден")
            return False
        
        # Проверка, запущен ли визуализатор
        if visualizer.is_running:
            self.logger.warning(f"Визуализатор {name} уже запущен")
            return False
        
        # Запуск визуализатора
        if background:
            # Запуск в фоновом режиме
            thread = Thread(target=self._run_visualizer_in_background, args=(visualizer,))
            thread.daemon = True
            thread.start()
            
            self.background_threads[name] = thread
            self.logger.info(f"Визуализатор {name} запущен в фоновом режиме")
        else:
            # Запуск в текущем потоке
            visualizer.start()
            self.logger.info(f"Визуализатор {name} запущен")
        
        return True
    
    def stop_visualizer(self, name: str) -> bool:
        """
        Остановка визуализатора.
        
        Args:
            name: Имя визуализатора
            
        Returns:
            True, если визуализатор успешно остановлен, иначе False
        """
        visualizer = self.get_visualizer(name)
        if not visualizer:
            self.logger.error(f"Визуализатор с именем {name} не найден")
            return False
        
        # Проверка, запущен ли визуализатор
        if not visualizer.is_running:
            self.logger.warning(f"Визуализатор {name} не запущен")
            return False
        
        # Остановка визуализатора
        visualizer.stop()
        
        # Удаление фонового потока, если есть
        if name in self.background_threads:
            del self.background_threads[name]
        
        self.logger.info(f"Визуализатор {name} остановлен")
        return True
    
    def update_visualizer(self, name: str, data: Dict[str, Any]) -> bool:
        """
        Обновление данных визуализатора.
        
        Args:
            name: Имя визуализатора
            data: Данные для обновления
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        visualizer = self.get_visualizer(name)
        if not visualizer:
            self.logger.error(f"Визуализатор с именем {name} не найден")
            return False
        
        # Обновление данных визуализатора
        return visualizer.update(data)
    
    def update_all_visualizers(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """
        Обновление данных всех активных визуализаторов.
        
        Args:
            data: Данные для обновления
            
        Returns:
            Словарь с результатами обновления для каждого визуализатора
        """
        results = {}
        
        for name, visualizer in self.active_visualizers.items():
            results[name] = visualizer.update(data)
        
        return results
    
    def start_all_visualizers(self, background: bool = True) -> Dict[str, bool]:
        """
        Запуск всех активных визуализаторов.
        
        Args:
            background: Запустить в фоновом режиме
            
        Returns:
            Словарь с результатами запуска для каждого визуализатора
        """
        results = {}
        
        for name in self.active_visualizers:
            results[name] = self.start_visualizer(name, background)
        
        return results
    
    def stop_all_visualizers(self) -> Dict[str, bool]:
        """
        Остановка всех активных визуализаторов.
        
        Returns:
            Словарь с результатами остановки для каждого визуализатора
        """
        results = {}
        
        for name in list(self.active_visualizers.keys()):
            results[name] = self.stop_visualizer(name)
        
        return results
    
    def get_active_visualizers(self) -> Dict[str, BaseVisualizer]:
        """
        Получение списка активных визуализаторов.
        
        Returns:
            Словарь активных визуализаторов
        """
        return self.active_visualizers
    
    def get_available_visualizer_types(self) -> List[str]:
        """
        Получение списка доступных типов визуализаторов.
        
        Returns:
            Список доступных типов визуализаторов
        """
        return list(self.available_visualizers.keys())
    
    def register_visualizer_type(self, name: str, visualizer_class: Type[BaseVisualizer]) -> bool:
        """
        Регистрация нового типа визуализатора.
        
        Args:
            name: Имя типа визуализатора
            visualizer_class: Класс визуализатора
            
        Returns:
            True, если тип успешно зарегистрирован, иначе False
        """
        if name in self.available_visualizers:
            self.logger.warning(f"Тип визуализатора {name} уже зарегистрирован")
            return False
        
        self.available_visualizers[name] = visualizer_class
        self.logger.info(f"Тип визуализатора {name} зарегистрирован")
        return True
    
    def _run_visualizer_in_background(self, visualizer: BaseVisualizer) -> None:
        """
        Запуск визуализатора в фоновом режиме.
        
        Args:
            visualizer: Визуализатор
        """
        visualizer.start()
        
        # Для консольных визуализаторов запускаем метод run
        if hasattr(visualizer, 'run'):
            try:
                visualizer.run()
            except KeyboardInterrupt:
                self.logger.info(f"Визуализатор {visualizer.name} остановлен пользователем")
            except Exception as e:
                self.logger.error(f"Ошибка при работе визуализатора {visualizer.name}: {e}")
            finally:
                visualizer.stop()
    
    def create_and_start_visualizer(self, visualizer_type: str, name: Optional[str] = None,
                                  config: Optional[Dict[str, Any]] = None,
                                  background: bool = True) -> Optional[BaseVisualizer]:
        """
        Создание и запуск визуализатора.
        
        Args:
            visualizer_type: Тип визуализатора
            name: Имя визуализатора
            config: Конфигурация визуализатора
            background: Запустить в фоновом режиме
            
        Returns:
            Созданный визуализатор или None, если произошла ошибка
        """
        # Создание визуализатора
        visualizer = self.create_visualizer(visualizer_type, name, config)
        if not visualizer:
            return None
        
        # Добавление визуализатора в список активных
        if not self.add_visualizer(visualizer):
            return None
        
        # Запуск визуализатора
        if not self.start_visualizer(visualizer.name, background):
            self.remove_visualizer(visualizer.name)
            return None
        
        return visualizer
    
    async def start_visualization(self) -> None:
        """Запускает визуализацию."""
        try:
            config = self.orchestrator.config
            
            # Проверяем наличие раздела visualization в конфигурации
            if not config or "visualization" not in config:
                self.logger.warning("Раздел visualization отсутствует в конфигурации")
                return
                
            # Проверяем, включена ли визуализация
            if not config["visualization"].get("enabled", False):
                self.logger.info("Визуализация отключена в конфигурации")
                return
            
            # Создаем визуализатор, если он еще не создан
            if not self.visualizer:
                from visualization.console_ui import ConsoleVisualizer
                self.visualizer = ConsoleVisualizer(
                    name="console",
                    config=config["visualization"],
                    localization=self.orchestrator.localization_manager
                )
            
            # Запускаем визуализатор
            if hasattr(self.visualizer, 'start'):
                self.visualizer.start()
                current_mode = self.orchestrator.trading_mode_manager.get_current_mode()
                self.logger.info(f"Запущен консольный визуализатор в режиме {current_mode}")
            
            # Запускаем задачу обновления данных
            self.update_task = asyncio.create_task(self._periodic_visualization_update())
        except Exception as e:
            self.logger.error(f"Ошибка при запуске визуализации: {str(e)}")
            self.logger.debug(traceback.format_exc())
    
    async def stop_visualization(self) -> None:
        """Останавливает визуализацию."""
        if self.visualizer:
            self.logger.info("Остановка консольной визуализации")
            
            # Останавливаем визуализатор
            if hasattr(self.visualizer, 'stop'):
                self.visualizer.stop()
            
            # Отменяем задачу обновления, если она запущена
            if self.update_task and not self.update_task.done():
                self.update_task.cancel()
                try:
                    await self.update_task
                except asyncio.CancelledError:
                    pass
    
    async def _periodic_visualization_update(self) -> None:
        """Периодическое обновление данных визуализатора."""
        while True:
            try:
                if not self.orchestrator.running:
                    await asyncio.sleep(1)
                    continue
                    
                await self._update_visualization()
                await asyncio.sleep(1)  # Обновляем каждую секунду
            except asyncio.CancelledError:
                self.logger.info("Задача обновления визуализации отменена")
                break
            except Exception as e:
                self.logger.error(f"Ошибка в цикле обновления визуализации: {str(e)}")
                self.logger.debug(traceback.format_exc())
                await asyncio.sleep(1)  # Пауза перед следующей попыткой

    async def _update_visualization(self) -> None:
        """Обновляет данные визуализации."""
        if not self.visualizer:
            return
            
        # Гарантируем наличие базовых данных для визуализации
        self._ensure_visualization_data()
        
        try:
            # Формируем единый словарь данных для визуализации
            visualization_data = {
                "mode": self.orchestrator.trading_mode_manager.get_current_mode(),
                "symbol": self.orchestrator.config.get("general", {}).get("symbol", ""),
                "current_price": self._get_safe_price(),
                "balance": await self._get_safe_balance(),
                "positions": await self._get_safe_positions(),
                "signals": self._get_safe_signals(),
                "indicators": self._get_safe_indicators(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Обновляем визуализатор единым вызовом
            if hasattr(self.visualizer, 'update'):
                self.visualizer.update(visualization_data)
            else:
                # Если метод update не найден, используем отдельные методы
                await self._update_visualization_by_parts(visualization_data)
                
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении визуализации: {str(e)}")
            self.logger.debug(traceback.format_exc())
    
    async def _update_visualization_by_parts(self, data: Dict[str, Any]) -> None:
        """Обновляет визуализатор по частям, если единый метод update не доступен."""
        # Обновляем режим работы
        if hasattr(self.visualizer, 'update_mode'):
            self.visualizer.update_mode(data["mode"])
            
        # Обновляем торговую пару
        if hasattr(self.visualizer, 'update_trading_pair'):
            symbol = data["symbol"]
            interval = self.orchestrator.config.get("general", {}).get("interval", "1h")
            self.visualizer.update_trading_pair(symbol, interval)
            
        # Обновляем баланс
        if hasattr(self.visualizer, 'update_balance'):
            self.visualizer.update_balance(data["balance"])
            
        # Обновляем цены
        if hasattr(self.visualizer, 'update_price') and data["current_price"]:
            self.visualizer.update_price(data["current_price"])
            
        # Обновляем индикаторы
        if hasattr(self.visualizer, 'update_indicators'):
            self.visualizer.update_indicators(data["indicators"])
            
        # Обновляем сигналы
        if hasattr(self.visualizer, 'update_signals'):
            self.visualizer.update_signals(data["signals"])
            
        # Обновляем позиции
        if hasattr(self.visualizer, 'update_positions'):
            self.visualizer.update_positions(data["positions"])
    
    def _get_safe_price(self) -> Optional[float]:
        """Безопасно получает текущую цену."""
        try:
            with self._data_lock:
                if hasattr(self.orchestrator, '_prices') and self.orchestrator._prices:
                    return self.orchestrator._prices[0]
                return None
        except Exception as e:
            self.logger.warning(f"Ошибка при получении цены: {str(e)}")
            return None
    
    async def _get_safe_balance(self) -> Optional[float]:
        """Безопасно получает текущий баланс."""
        try:
            with self._data_lock:
                if hasattr(self.orchestrator, 'get_balance'):
                    # Проверяем, является ли метод асинхронным
                    import inspect
                    if inspect.iscoroutinefunction(self.orchestrator.get_balance):
                        balance = await self.orchestrator.get_balance()
                    else:
                        balance = self.orchestrator.get_balance()
                    return balance if balance is not None else 0.0
                return 0.0
        except Exception as e:
            self.logger.warning(f"Ошибка при получении баланса: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return 0.0
    
    async def _get_safe_positions(self) -> List[Dict[str, Any]]:
        """Безопасно получает открытые позиции."""
        try:
            # Проверяем наличие метода get_trader
            if not hasattr(self.orchestrator, 'get_trader'):
                return []
                
            # Получаем трейдер
            trader = self.orchestrator.get_trader()
            if not trader:
                return []
                
            # Проверяем наличие метода get_open_positions
            if not hasattr(trader, 'get_open_positions'):
                return []
                
            # Проверяем, является ли метод асинхронным
            import inspect
            if inspect.iscoroutinefunction(trader.get_open_positions):
                positions = await trader.get_open_positions()
            else:
                positions = trader.get_open_positions()
                
            return positions if positions else []
        except Exception as e:
            self.logger.warning(f"Ошибка при получении позиций: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return []
    
    def _get_safe_signals(self) -> List[Dict[str, Any]]:
        """Безопасно получает сигналы."""
        try:
            with self._data_lock:
                if hasattr(self.orchestrator, 'get_signals'):
                    signals = self.orchestrator.get_signals()
                    return signals if signals else []
                return []
        except Exception as e:
            self.logger.warning(f"Ошибка при получении сигналов: {str(e)}")
            return []
    
    def _get_safe_indicators(self) -> Dict[str, Any]:
        """Безопасно получает индикаторы."""
        try:
            with self._data_lock:
                if hasattr(self.orchestrator, 'get_indicators'):
                    indicators = self.orchestrator.get_indicators()
                    return indicators if indicators else {}
                return {}
        except Exception as e:
            self.logger.warning(f"Ошибка при получении индикаторов: {str(e)}")
            return {}
    
    def _ensure_visualization_data(self):
        """Гарантирует наличие базовых данных для визуализации в оркестраторе."""
        if not self.orchestrator:
            self.logger.warning("Оркестратор не инициализирован, невозможно обеспечить данные для визуализации")
            return
            
        with self._data_lock:
            # Проверяем наличие цен
            if not hasattr(self.orchestrator, '_prices') or self.orchestrator._prices is None:
                self.orchestrator._prices = []
                # Добавляем тестовые данные для начального отображения
                import random
                base_price = 50000.0
                for i in range(10):
                    self.orchestrator._prices.append(base_price + random.uniform(-100, 100))
            
            # Проверяем наличие индикаторов
            if not hasattr(self.orchestrator, '_indicators') or self.orchestrator._indicators is None:
                self.orchestrator._indicators = {
                    "rsi": 50.0,
                    "macd": 0.0,
                    "macd_signal": 0.0,
                    "bb_upper": 0.0,
                    "bb_middle": 0.0,
                    "bb_lower": 0.0
                }
            
            # Проверяем наличие сигналов
            if not hasattr(self.orchestrator, '_signals') or self.orchestrator._signals is None:
                self.orchestrator._signals = []
                
        self.logger.debug("Данные для визуализации инициализированы")

    async def update(self):
        """Обновляет все активные визуализаторы с актуальными данными."""
        try:
            # Получаем все необходимые данные для визуализации
            visualization_data = await self._prepare_visualization_data()
            
            # Обновляем основной визуализатор
            if self.visualizer:
                if hasattr(self.visualizer, 'update'):
                    self.visualizer.update(visualization_data)
            
            # Обновляем все остальные активные визуализаторы
            for name, visualizer in self.active_visualizers.items():
                if visualizer != self.visualizer and hasattr(visualizer, 'update'):
                    visualizer.update(visualization_data)
                    
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении визуализаторов: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return False
            
    async def _prepare_visualization_data(self) -> Dict[str, Any]:
        """Подготавливает данные для визуализации."""
        # Убеждаемся, что данные инициализированы
        self._ensure_visualization_data()
        
        # Собираем все необходимые данные
        data = {
            "mode": self.orchestrator.trading_mode_manager.get_current_mode(),
            "symbol": getattr(self.orchestrator, 'symbol', 'BTC/USDT'),
            "interval": getattr(self.orchestrator, 'interval', '1h'),
            "current_price": self._get_safe_price(),
            "balance": await self._get_safe_balance(),
            "indicators": self._get_safe_indicators(),
            "signals": self._get_safe_signals(),
            "positions": await self._get_safe_positions()
        }
        
        return data 