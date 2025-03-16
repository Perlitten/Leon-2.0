"""
Модуль управления моделями машинного обучения.

Предоставляет классы и функции для управления жизненным циклом моделей,
включая сохранение, загрузку, версионирование и отслеживание метаданных.
"""

import os
import json
import pickle
import shutil
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime
import pandas as pd
import numpy as np
import glob
import re

from .models import BaseModel


class ModelManager:
    """
    Класс для управления жизненным циклом моделей машинного обучения.
    
    Предоставляет функциональность для сохранения, загрузки, версионирования
    и отслеживания метаданных моделей.
    
    Attributes:
        models_dir (str): Директория для хранения моделей
        registry (Dict[str, Dict[str, Any]]): Реестр моделей
        config (Dict[str, Any]): Конфигурация менеджера моделей
    """
    
    def __init__(self, models_dir: str = "models", 
                config: Optional[Dict[str, Any]] = None):
        """
        Инициализация менеджера моделей.
        
        Args:
            models_dir: Директория для хранения моделей
            config: Конфигурация менеджера моделей
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.models_dir = models_dir
        self.config = config or {}
        
        # Создание директории для моделей, если она не существует
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Инициализация реестра моделей
        self.registry = self._load_registry()
        
        self.logger.info(f"Инициализирован менеджер моделей с {len(self.registry)} моделями")
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Загрузка реестра моделей из файла.
        
        Returns:
            Dict[str, Dict[str, Any]]: Реестр моделей
        """
        registry_path = os.path.join(self.models_dir, "registry.json")
        
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
                self.logger.info(f"Загружен реестр моделей из {registry_path}")
                return registry
            except Exception as e:
                self.logger.error(f"Ошибка при загрузке реестра моделей: {e}")
                return {}
        else:
            self.logger.info(f"Реестр моделей не найден, создан новый реестр")
            return {}
    
    def _save_registry(self) -> None:
        """
        Сохранение реестра моделей в файл.
        """
        registry_path = os.path.join(self.models_dir, "registry.json")
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.registry, f, indent=4)
            self.logger.info(f"Реестр моделей сохранен в {registry_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении реестра моделей: {e}")
    
    def save_model(self, model: BaseModel, 
                  version: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None,
                  overwrite: bool = False) -> str:
        """
        Сохранение модели.
        
        Args:
            model: Модель для сохранения
            version: Версия модели (если не указана, будет сгенерирована автоматически)
            metadata: Дополнительные метаданные модели
            overwrite: Перезаписать модель, если она уже существует
            
        Returns:
            str: Идентификатор сохраненной модели
            
        Raises:
            ValueError: Если модель с указанной версией уже существует и overwrite=False
        """
        # Генерация версии, если не указана
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Формирование идентификатора модели
        model_id = f"{model.name}_{version}"
        
        # Проверка существования модели с указанной версией
        if model_id in self.registry and not overwrite:
            raise ValueError(f"Модель с идентификатором {model_id} уже существует. Используйте overwrite=True для перезаписи.")
        
        # Создание директории для модели
        model_dir = os.path.join(self.models_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Путь к файлу модели
        model_path = os.path.join(model_dir, f"{model_id}.pkl")
        
        # Объединение метаданных
        combined_metadata = model.get_metadata().copy()
        if metadata:
            combined_metadata.update(metadata)
        
        # Добавление информации о версии и времени сохранения
        combined_metadata.update({
            "model_id": model_id,
            "version": version,
            "saved_at": datetime.now().isoformat(),
            "model_name": model.name,
            "is_fitted": model.is_fitted
        })
        
        # Сохранение модели
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Сохранение метаданных
        metadata_path = os.path.join(model_dir, f"{model_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(combined_metadata, f, indent=4)
        
        # Обновление реестра моделей
        self.registry[model_id] = {
            "model_path": model_path,
            "metadata_path": metadata_path,
            "metadata": combined_metadata
        }
        
        # Сохранение реестра
        self._save_registry()
        
        self.logger.info(f"Модель {model_id} сохранена в {model_path}")
        
        return model_id
    
    def load_model(self, model_id: str) -> BaseModel:
        """
        Загрузка модели.
        
        Args:
            model_id: Идентификатор модели
            
        Returns:
            BaseModel: Загруженная модель
            
        Raises:
            ValueError: Если модель с указанным идентификатором не найдена
        """
        # Проверка существования модели в реестре
        if model_id not in self.registry:
            raise ValueError(f"Модель с идентификатором {model_id} не найдена в реестре")
        
        # Получение пути к файлу модели
        model_path = self.registry[model_id]["model_path"]
        
        # Проверка существования файла модели
        if not os.path.exists(model_path):
            raise ValueError(f"Файл модели {model_path} не найден")
        
        # Загрузка модели
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        self.logger.info(f"Модель {model_id} загружена из {model_path}")
        
        return model
    
    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Получение метаданных модели.
        
        Args:
            model_id: Идентификатор модели
            
        Returns:
            Dict[str, Any]: Метаданные модели
            
        Raises:
            ValueError: Если модель с указанным идентификатором не найдена
        """
        # Проверка существования модели в реестре
        if model_id not in self.registry:
            raise ValueError(f"Модель с идентификатором {model_id} не найдена в реестре")
        
        # Получение метаданных из реестра
        metadata = self.registry[model_id]["metadata"]
        
        return metadata.copy()
    
    def delete_model(self, model_id: str) -> None:
        """
        Удаление модели.
        
        Args:
            model_id: Идентификатор модели
            
        Raises:
            ValueError: Если модель с указанным идентификатором не найдена
        """
        # Проверка существования модели в реестре
        if model_id not in self.registry:
            raise ValueError(f"Модель с идентификатором {model_id} не найдена в реестре")
        
        # Получение пути к директории модели
        model_dir = os.path.dirname(self.registry[model_id]["model_path"])
        
        # Удаление директории модели
        shutil.rmtree(model_dir)
        
        # Удаление модели из реестра
        del self.registry[model_id]
        
        # Сохранение реестра
        self._save_registry()
        
        self.logger.info(f"Модель {model_id} удалена")
    
    def list_models(self, name: Optional[str] = None, 
                   task_type: Optional[str] = None,
                   is_fitted: Optional[bool] = None,
                   sort_by: str = "saved_at",
                   ascending: bool = False) -> List[Dict[str, Any]]:
        """
        Получение списка моделей.
        
        Args:
            name: Фильтр по имени модели
            task_type: Фильтр по типу задачи
            is_fitted: Фильтр по статусу обучения
            sort_by: Поле для сортировки
            ascending: Порядок сортировки (по возрастанию или убыванию)
            
        Returns:
            List[Dict[str, Any]]: Список моделей
        """
        # Фильтрация моделей
        filtered_models = []
        
        for model_id, model_info in self.registry.items():
            metadata = model_info["metadata"]
            
            # Фильтрация по имени
            if name is not None and metadata.get("model_name") != name:
                continue
            
            # Фильтрация по типу задачи
            if task_type is not None and metadata.get("task_type") != task_type:
                continue
            
            # Фильтрация по статусу обучения
            if is_fitted is not None and metadata.get("is_fitted") != is_fitted:
                continue
            
            # Добавление модели в результат
            filtered_models.append({
                "model_id": model_id,
                "model_name": metadata.get("model_name"),
                "version": metadata.get("version"),
                "task_type": metadata.get("task_type"),
                "is_fitted": metadata.get("is_fitted"),
                "saved_at": metadata.get("saved_at"),
                "metrics": metadata.get("metrics", {})
            })
        
        # Сортировка моделей
        if sort_by in ["model_id", "model_name", "version", "task_type", "is_fitted", "saved_at"]:
            filtered_models.sort(key=lambda x: x.get(sort_by, ""), reverse=not ascending)
        
        return filtered_models
    
    def get_best_model(self, name: Optional[str] = None, 
                      task_type: Optional[str] = None,
                      metric: str = "accuracy",
                      higher_is_better: bool = True) -> Optional[str]:
        """
        Получение лучшей модели по указанной метрике.
        
        Args:
            name: Фильтр по имени модели
            task_type: Фильтр по типу задачи
            metric: Метрика для сравнения моделей
            higher_is_better: Флаг, указывающий, что большее значение метрики лучше
            
        Returns:
            Optional[str]: Идентификатор лучшей модели или None, если модели не найдены
        """
        # Получение списка моделей
        models = self.list_models(name=name, task_type=task_type, is_fitted=True)
        
        if not models:
            return None
        
        # Фильтрация моделей, у которых есть указанная метрика
        models_with_metric = [
            model for model in models 
            if model.get("metrics") and metric in model.get("metrics", {})
        ]
        
        if not models_with_metric:
            return None
        
        # Сортировка моделей по метрике
        models_with_metric.sort(
            key=lambda x: x.get("metrics", {}).get(metric, 0),
            reverse=higher_is_better
        )
        
        # Возвращение идентификатора лучшей модели
        return models_with_metric[0]["model_id"]
    
    def export_model(self, model_id: str, export_dir: str) -> str:
        """
        Экспорт модели в указанную директорию.
        
        Args:
            model_id: Идентификатор модели
            export_dir: Директория для экспорта
            
        Returns:
            str: Путь к экспортированной модели
            
        Raises:
            ValueError: Если модель с указанным идентификатором не найдена
        """
        # Проверка существования модели в реестре
        if model_id not in self.registry:
            raise ValueError(f"Модель с идентификатором {model_id} не найдена в реестре")
        
        # Создание директории для экспорта
        os.makedirs(export_dir, exist_ok=True)
        
        # Получение путей к файлам модели
        model_path = self.registry[model_id]["model_path"]
        metadata_path = self.registry[model_id]["metadata_path"]
        
        # Пути к экспортированным файлам
        export_model_path = os.path.join(export_dir, os.path.basename(model_path))
        export_metadata_path = os.path.join(export_dir, os.path.basename(metadata_path))
        
        # Копирование файлов
        shutil.copy2(model_path, export_model_path)
        shutil.copy2(metadata_path, export_metadata_path)
        
        self.logger.info(f"Модель {model_id} экспортирована в {export_dir}")
        
        return export_model_path
    
    def import_model(self, model_path: str, 
                    metadata_path: Optional[str] = None,
                    new_version: Optional[str] = None,
                    overwrite: bool = False) -> str:
        """
        Импорт модели из указанного пути.
        
        Args:
            model_path: Путь к файлу модели
            metadata_path: Путь к файлу метаданных (если не указан, будет сгенерирован автоматически)
            new_version: Новая версия модели (если не указана, будет сохранена исходная версия)
            overwrite: Перезаписать модель, если она уже существует
            
        Returns:
            str: Идентификатор импортированной модели
            
        Raises:
            ValueError: Если файл модели не найден
        """
        # Проверка существования файла модели
        if not os.path.exists(model_path):
            raise ValueError(f"Файл модели {model_path} не найден")
        
        # Загрузка модели
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Загрузка метаданных
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Сохранение модели
        model_id = self.save_model(model, version=new_version, metadata=metadata, overwrite=overwrite)
        
        self.logger.info(f"Модель импортирована из {model_path} с идентификатором {model_id}")
        
        return model_id
    
    def create_model_report(self, model_id: str, 
                           report_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Создание отчета о модели.
        
        Args:
            model_id: Идентификатор модели
            report_path: Путь для сохранения отчета (если не указан, отчет не будет сохранен)
            
        Returns:
            Dict[str, Any]: Отчет о модели
            
        Raises:
            ValueError: Если модель с указанным идентификатором не найдена
        """
        # Проверка существования модели в реестре
        if model_id not in self.registry:
            raise ValueError(f"Модель с идентификатором {model_id} не найдена в реестре")
        
        # Получение метаданных модели
        metadata = self.get_model_metadata(model_id)
        
        # Создание отчета
        report = {
            "model_id": model_id,
            "model_name": metadata.get("model_name"),
            "version": metadata.get("version"),
            "task_type": metadata.get("task_type"),
            "is_fitted": metadata.get("is_fitted"),
            "saved_at": metadata.get("saved_at"),
            "metrics": metadata.get("metrics", {}),
            "params": metadata.get("params", {}),
            "feature_names": metadata.get("feature_names", []),
            "target_name": metadata.get("target_name", ""),
            "description": metadata.get("description", "")
        }
        
        # Сохранение отчета, если указан путь
        if report_path:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            self.logger.info(f"Отчет о модели {model_id} сохранен в {report_path}")
        
        return report
    
    def compare_models(self, model_ids: List[str], 
                      metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Сравнение моделей по метрикам.
        
        Args:
            model_ids: Список идентификаторов моделей
            metrics: Список метрик для сравнения (если не указан, будут использованы все доступные метрики)
            
        Returns:
            pd.DataFrame: Таблица сравнения моделей
            
        Raises:
            ValueError: Если одна из моделей не найдена
        """
        # Проверка существования моделей в реестре
        for model_id in model_ids:
            if model_id not in self.registry:
                raise ValueError(f"Модель с идентификатором {model_id} не найдена в реестре")
        
        # Получение метаданных моделей
        models_metadata = [self.get_model_metadata(model_id) for model_id in model_ids]
        
        # Определение списка метрик для сравнения
        if metrics is None:
            # Сбор всех доступных метрик
            all_metrics = set()
            for metadata in models_metadata:
                all_metrics.update(metadata.get("metrics", {}).keys())
            metrics = list(all_metrics)
        
        # Создание таблицы сравнения
        comparison_data = []
        
        for model_id, metadata in zip(model_ids, models_metadata):
            model_metrics = metadata.get("metrics", {})
            
            row = {
                "model_id": model_id,
                "model_name": metadata.get("model_name"),
                "version": metadata.get("version"),
                "task_type": metadata.get("task_type"),
                "is_fitted": metadata.get("is_fitted"),
                "saved_at": metadata.get("saved_at")
            }
            
            # Добавление метрик
            for metric in metrics:
                row[metric] = model_metrics.get(metric, None)
            
            comparison_data.append(row)
        
        # Создание DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        return comparison_df
    
    def get_model_versions(self, name: str) -> List[Dict[str, Any]]:
        """
        Получение списка версий модели.
        
        Args:
            name: Имя модели
            
        Returns:
            List[Dict[str, Any]]: Список версий модели
        """
        # Получение списка моделей с указанным именем
        models = self.list_models(name=name, sort_by="saved_at", ascending=False)
        
        return models
    
    def get_latest_version(self, name: str) -> Optional[str]:
        """
        Получение последней версии модели.
        
        Args:
            name: Имя модели
            
        Returns:
            Optional[str]: Идентификатор последней версии модели или None, если модели не найдены
        """
        # Получение списка версий модели
        versions = self.get_model_versions(name)
        
        if not versions:
            return None
        
        # Возвращение идентификатора последней версии
        return versions[0]["model_id"]
    
    def rollback_to_version(self, model_id: str) -> str:
        """
        Откат к указанной версии модели.
        
        Args:
            model_id: Идентификатор версии модели
            
        Returns:
            str: Идентификатор новой версии модели
            
        Raises:
            ValueError: Если модель с указанным идентификатором не найдена
        """
        # Загрузка модели
        model = self.load_model(model_id)
        
        # Получение метаданных модели
        metadata = self.get_model_metadata(model_id)
        
        # Генерация новой версии
        new_version = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_rollback_from_{metadata.get('version')}"
        
        # Сохранение модели с новой версией
        new_model_id = self.save_model(model, version=new_version, metadata=metadata)
        
        self.logger.info(f"Выполнен откат к версии {model_id}, создана новая версия {new_model_id}")
        
        return new_model_id 