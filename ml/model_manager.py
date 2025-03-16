"""
Модуль управления моделями машинного обучения.

Предоставляет класс для загрузки, сохранения и управления моделями.
"""

import os
import logging
import json
import pickle
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

class ModelManager:
    """
    Класс для управления моделями машинного обучения.
    
    Отвечает за:
    - Загрузку моделей из файлов
    - Сохранение моделей в файлы
    - Управление версиями моделей
    - Предоставление моделей для использования
    """
    
    def __init__(self, models_dir: str = "ml/models"):
        """
        Инициализация менеджера моделей.
        
        Args:
            models_dir: Директория с моделями
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models_dir = models_dir
        self.models = {}
        self.model_metadata = {}
        
        # Создание директории для моделей, если она не существует
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Загрузка списка доступных моделей
        self._load_model_list()
    
    def _load_model_list(self) -> None:
        """
        Загрузка списка доступных моделей из директории.
        """
        self.model_metadata = {}
        
        # Поиск файлов метаданных моделей
        for filename in os.listdir(self.models_dir):
            if filename.endswith(".json"):
                model_id = filename.replace(".json", "")
                metadata_path = os.path.join(self.models_dir, filename)
                
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    
                    self.model_metadata[model_id] = metadata
                    self.logger.info(f"Найдена модель: {model_id}")
                except Exception as e:
                    self.logger.error(f"Ошибка при загрузке метаданных модели {model_id}: {e}")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Получение списка доступных моделей.
        
        Returns:
            Список метаданных доступных моделей
        """
        return [
            {
                "id": model_id,
                **metadata
            }
            for model_id, metadata in self.model_metadata.items()
        ]
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """
        Загрузка модели по ID.
        
        Args:
            model_id: ID модели
            
        Returns:
            Загруженная модель или None, если модель не найдена
        """
        # Проверка наличия модели в кэше
        if model_id in self.models:
            self.logger.info(f"Модель {model_id} загружена из кэша")
            return self.models[model_id]
        
        # Проверка наличия метаданных модели
        if model_id not in self.model_metadata:
            self.logger.error(f"Модель {model_id} не найдена")
            return None
        
        # Загрузка модели из файла
        model_path = os.path.join(self.models_dir, f"{model_id}.pkl")
        
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            # Сохранение модели в кэше
            self.models[model_id] = model
            
            self.logger.info(f"Модель {model_id} успешно загружена")
            return model
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели {model_id}: {e}")
            return None
    
    def save_model(self, model: Any, model_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Сохранение модели.
        
        Args:
            model: Модель для сохранения
            model_id: ID модели
            metadata: Метаданные модели
            
        Returns:
            Успешность сохранения
        """
        # Добавление временной метки
        metadata["timestamp"] = datetime.now().isoformat()
        
        # Сохранение модели в файл
        model_path = os.path.join(self.models_dir, f"{model_id}.pkl")
        metadata_path = os.path.join(self.models_dir, f"{model_id}.json")
        
        try:
            # Сохранение модели
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            
            # Сохранение метаданных
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            
            # Обновление кэша и метаданных
            self.models[model_id] = model
            self.model_metadata[model_id] = metadata
            
            self.logger.info(f"Модель {model_id} успешно сохранена")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении модели {model_id}: {e}")
            return False
    
    def delete_model(self, model_id: str) -> bool:
        """
        Удаление модели.
        
        Args:
            model_id: ID модели
            
        Returns:
            Успешность удаления
        """
        # Проверка наличия модели
        if model_id not in self.model_metadata:
            self.logger.error(f"Модель {model_id} не найдена")
            return False
        
        # Удаление файлов модели
        model_path = os.path.join(self.models_dir, f"{model_id}.pkl")
        metadata_path = os.path.join(self.models_dir, f"{model_id}.json")
        
        try:
            # Удаление файлов
            if os.path.exists(model_path):
                os.remove(model_path)
            
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # Удаление из кэша и метаданных
            if model_id in self.models:
                del self.models[model_id]
            
            del self.model_metadata[model_id]
            
            self.logger.info(f"Модель {model_id} успешно удалена")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при удалении модели {model_id}: {e}")
            return False
    
    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение метаданных модели.
        
        Args:
            model_id: ID модели
            
        Returns:
            Метаданные модели или None, если модель не найдена
        """
        if model_id not in self.model_metadata:
            self.logger.error(f"Модель {model_id} не найдена")
            return None
        
        return self.model_metadata[model_id]
    
    def update_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Обновление метаданных модели.
        
        Args:
            model_id: ID модели
            metadata: Новые метаданные
            
        Returns:
            Успешность обновления
        """
        # Проверка наличия модели
        if model_id not in self.model_metadata:
            self.logger.error(f"Модель {model_id} не найдена")
            return False
        
        # Обновление метаданных
        metadata_path = os.path.join(self.models_dir, f"{model_id}.json")
        
        try:
            # Объединение существующих и новых метаданных
            updated_metadata = {**self.model_metadata[model_id], **metadata}
            
            # Сохранение метаданных
            with open(metadata_path, "w") as f:
                json.dump(updated_metadata, f, indent=4)
            
            # Обновление кэша метаданных
            self.model_metadata[model_id] = updated_metadata
            
            self.logger.info(f"Метаданные модели {model_id} успешно обновлены")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении метаданных модели {model_id}: {e}")
            return False
    
    def clear_cache(self) -> None:
        """
        Очистка кэша моделей.
        """
        self.models = {}
        self.logger.info("Кэш моделей очищен")
    
    def reload_model_list(self) -> None:
        """
        Перезагрузка списка моделей.
        """
        self._load_model_list()
        self.logger.info("Список моделей перезагружен") 