"""
Базовый класс для моделей машинного обучения.

Предоставляет интерфейс для всех моделей ML в системе.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
import os
import pickle
import json
from datetime import datetime


class BaseModel(ABC):
    """
    Абстрактный базовый класс для всех моделей машинного обучения.
    
    Определяет общий интерфейс, который должны реализовать все модели.
    """
    
    def __init__(self, name: str, version: str = "1.0.0", **kwargs):
        """
        Инициализация базовой модели.
        
        Args:
            name: Название модели
            version: Версия модели
            **kwargs: Дополнительные параметры модели
        """
        self.name = name
        self.version = version
        self.params = kwargs
        self.is_trained = False
        self.model = None
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{name}")
        self.model_type = self.__class__.__name__
        self.metadata = {
            "name": name,
            "type": self.model_type,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "features": [],
            "target": None,
            "metrics": {},
            "parameters": {},
            "description": ""
        }
        self.logger.info(f"Инициализирована модель {name} (тип: {self.model_type}, версия: {version})")
    
    @abstractmethod
    def train(self, X: Union[np.ndarray, pd.DataFrame], 
              y: Union[np.ndarray, pd.Series], **kwargs) -> Dict[str, Any]:
        """
        Обучение модели на данных.
        
        Args:
            X: Признаки для обучения
            y: Целевые значения
            **kwargs: Дополнительные параметры обучения
            
        Returns:
            Dict[str, Any]: Метрики обучения
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Получение предсказаний модели.
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            np.ndarray: Предсказанные значения
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Оценка производительности модели на данных.
        
        Args:
            X: Признаки для оценки
            y: Истинные значения
            
        Returns:
            Dict[str, float]: Метрики оценки
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """
        Получение параметров модели.
        
        Returns:
            Dict[str, Any]: Параметры модели
        """
        return self.params
    
    def set_params(self, **params) -> None:
        """
        Установка параметров модели.
        
        Args:
            **params: Новые параметры модели
        """
        self.params.update(params)
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Получение метаданных модели.
        
        Returns:
            Dict[str, Any]: Метаданные модели
        """
        return self.metadata
    
    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Обновление метаданных модели.
        
        Args:
            metadata: Новые метаданные
        """
        self.metadata.update(metadata)
        self.metadata["updated_at"] = datetime.now().isoformat()
        self.logger.debug(f"Метаданные модели обновлены")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Получение важности признаков.
        
        Returns:
            Словарь с важностью признаков
        """
        raise NotImplementedError("Метод get_feature_importance должен быть реализован в дочернем классе")
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Получение параметров модели.
        
        Returns:
            Параметры модели
        """
        return self.metadata.get("parameters", {})
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Установка параметров модели.
        
        Args:
            parameters: Параметры модели
        """
        self.metadata["parameters"] = parameters
        self.logger.debug(f"Параметры модели установлены: {parameters}")
    
    def get_description(self) -> str:
        """
        Получение описания модели.
        
        Returns:
            Описание модели
        """
        return self.metadata.get("description", "")
    
    def set_description(self, description: str) -> None:
        """
        Установка описания модели.
        
        Args:
            description: Описание модели
        """
        self.metadata["description"] = description
        self.logger.debug(f"Описание модели установлено")
    
    def __str__(self) -> str:
        """
        Строковое представление модели.
        
        Returns:
            str: Строковое представление
        """
        return f"{self.__class__.__name__}(name={self.name}, version={self.version}, trained={self.is_trained})"
    
    def save(self, path: str) -> str:
        """
        Сохранение модели.
        
        Args:
            path: Путь для сохранения модели
            
        Returns:
            Полный путь к сохраненной модели
        """
        if self.model is None:
            self.logger.error("Невозможно сохранить модель: модель не обучена")
            return ""
        
        # Создание директории, если она не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Обновление метаданных
        self.metadata["updated_at"] = datetime.now().isoformat()
        
        # Сохранение модели
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Сохранение метаданных
        metadata_path = f"{os.path.splitext(path)[0]}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        
        self.logger.info(f"Модель сохранена в {path}")
        return path
    
    def load(self, path: str) -> bool:
        """
        Загрузка модели.
        
        Args:
            path: Путь к сохраненной модели
            
        Returns:
            True, если загрузка успешна, иначе False
        """
        if not os.path.exists(path):
            self.logger.error(f"Невозможно загрузить модель: файл {path} не существует")
            return False
        
        try:
            # Загрузка модели
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Загрузка метаданных
            metadata_path = f"{os.path.splitext(path)[0]}_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            self.logger.info(f"Модель загружена из {path}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {str(e)}")
            return False 