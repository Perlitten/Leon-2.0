"""
Базовый модуль визуализации для Leon Trading Bot.

Предоставляет базовые классы и интерфейсы для визуализации данных.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union


class BaseVisualizer(ABC):
    """
    Базовый абстрактный класс для всех визуализаторов.
    
    Определяет общий интерфейс для всех визуализаторов в системе.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Инициализация базового визуализатора.
        
        Args:
            name: Имя визуализатора
            config: Конфигурация визуализатора
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"visualizer.{name}")
        self.is_running = False
        self.logger.info(f"Визуализатор {name} инициализирован")
    
    @abstractmethod
    def start(self) -> bool:
        """
        Запуск визуализатора.
        
        Returns:
            True, если визуализатор успешно запущен, иначе False
        """
        self.is_running = True
        self.logger.info(f"Визуализатор {self.name} запущен")
        return True
    
    @abstractmethod
    def stop(self) -> bool:
        """
        Остановка визуализатора.
        
        Returns:
            True, если визуализатор успешно остановлен, иначе False
        """
        self.is_running = False
        self.logger.info(f"Визуализатор {self.name} остановлен")
        return True
    
    @abstractmethod
    def update(self, data: Dict[str, Any]) -> bool:
        """
        Обновление данных визуализатора.
        
        Args:
            data: Данные для обновления
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        self.logger.debug(f"Обновление данных визуализатора {self.name}")
        return True
    
    @abstractmethod
    def render(self) -> Any:
        """
        Отрисовка визуализации.
        
        Returns:
            Результат отрисовки (зависит от конкретной реализации)
        """
        self.logger.debug(f"Отрисовка визуализатора {self.name}")
        pass


class ConsoleVisualizer(BaseVisualizer):
    """
    Базовый класс для визуализаторов, работающих в консоли.
    
    Предоставляет общую функциональность для консольных визуализаторов.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Инициализация консольного визуализатора.
        
        Args:
            name: Имя визуализатора
            config: Конфигурация визуализатора
        """
        super().__init__(name, config)
        self.refresh_rate = self.config.get("refresh_rate", 1.0)  # Частота обновления в секундах
        self.width = self.config.get("width", 80)  # Ширина консоли
        self.height = self.config.get("height", 24)  # Высота консоли
        self.clear_screen = self.config.get("clear_screen", True)  # Очищать экран перед отрисовкой
        self.logger.debug(f"Консольный визуализатор {name} инициализирован с параметрами: "
                         f"refresh_rate={self.refresh_rate}, width={self.width}, height={self.height}")
    
    def clear(self) -> None:
        """Очистка экрана."""
        if self.clear_screen:
            print("\033c", end="")  # Очистка экрана
    
    def start(self) -> bool:
        """
        Запуск консольного визуализатора.
        
        Returns:
            True, если визуализатор успешно запущен, иначе False
        """
        return super().start()
    
    def stop(self) -> bool:
        """
        Остановка консольного визуализатора.
        
        Returns:
            True, если визуализатор успешно остановлен, иначе False
        """
        return super().stop()
    
    def update(self, data: Dict[str, Any]) -> bool:
        """
        Обновление данных консольного визуализатора.
        
        Args:
            data: Данные для обновления
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        return super().update(data)
    
    def render(self) -> str:
        """
        Отрисовка консольного визуализатора.
        
        Returns:
            Строка для вывода в консоль
        """
        return ""


class WebVisualizer(BaseVisualizer):
    """
    Базовый класс для визуализаторов, работающих через веб-интерфейс.
    
    Предоставляет общую функциональность для веб-визуализаторов.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Инициализация веб-визуализатора.
        
        Args:
            name: Имя визуализатора
            config: Конфигурация визуализатора
        """
        super().__init__(name, config)
        self.host = self.config.get("host", "127.0.0.1")  # Хост для веб-сервера
        self.port = self.config.get("port", 8080)  # Порт для веб-сервера
        self.debug = self.config.get("debug", False)  # Режим отладки
        self.logger.debug(f"Веб-визуализатор {name} инициализирован с параметрами: "
                         f"host={self.host}, port={self.port}, debug={self.debug}")
    
    def start(self) -> bool:
        """
        Запуск веб-визуализатора.
        
        Returns:
            True, если визуализатор успешно запущен, иначе False
        """
        return super().start()
    
    def stop(self) -> bool:
        """
        Остановка веб-визуализатора.
        
        Returns:
            True, если визуализатор успешно остановлен, иначе False
        """
        return super().stop()
    
    def update(self, data: Dict[str, Any]) -> bool:
        """
        Обновление данных веб-визуализатора.
        
        Args:
            data: Данные для обновления
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        return super().update(data)
    
    def render(self) -> Dict[str, Any]:
        """
        Отрисовка веб-визуализатора.
        
        Returns:
            Данные для отправки клиенту
        """
        return {} 