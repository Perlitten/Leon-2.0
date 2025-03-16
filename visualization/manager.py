"""
Менеджер визуализаторов для Leon Trading Bot.

Предоставляет централизованное управление всеми визуализаторами в системе.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Type
from threading import Thread

from visualization.base import BaseVisualizer
from visualization.trading_dashboard import TradingDashboard
from visualization.candle_visualizer import CandleVisualizer
from visualization.web_dashboard import WebDashboard


class VisualizationManager:
    """
    Менеджер визуализаторов.
    
    Отвечает за создание, настройку и управление всеми визуализаторами в системе.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Инициализация менеджера визуализаторов.
        
        Args:
            config: Конфигурация визуализаторов
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        
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