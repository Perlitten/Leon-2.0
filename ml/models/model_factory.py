"""
Модуль фабрики моделей машинного обучения.

Предоставляет классы и функции для создания моделей различных типов
на основе конфигурации и параметров.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple, Type, Callable
import logging
import os
import json
from datetime import datetime
import importlib

# Импорты для моделей scikit-learn
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, SGDClassifier, SGDRegressor
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Импорты для моделей XGBoost (если доступны)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Импорты для моделей LightGBM (если доступны)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Импорты для моделей CatBoost (если доступны)
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Импорты для моделей TensorFlow/Keras (если доступны)
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Импорты для моделей PyTorch (если доступны)
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from .models import (
    BaseModel, 
    RegressionModel, 
    ClassificationModel, 
    EnsembleModel,
    TimeSeriesModel
)


class ModelFactory:
    """
    Фабрика для создания моделей машинного обучения различных типов.
    
    Предоставляет методы для создания моделей на основе конфигурации
    и параметров, а также для регистрации пользовательских типов моделей.
    
    Attributes:
        model_registry (Dict[str, Type[BaseModel]]): Реестр типов моделей
        config (Dict[str, Any]): Конфигурация фабрики моделей
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация фабрики моделей.
        
        Args:
            config: Конфигурация фабрики моделей
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.config = config or {}
        
        # Инициализация реестра типов моделей
        self.model_registry = {
            'regression': RegressionModel,
            'classification': ClassificationModel,
            'ensemble': EnsembleModel,
            'time_series': TimeSeriesModel
        }
        
        self.logger.info(f"Инициализирована фабрика моделей с {len(self.model_registry)} типами моделей")
    
    def register_model_type(self, model_type: str, model_class: Type[BaseModel]) -> None:
        """
        Регистрация нового типа модели.
        
        Args:
            model_type: Тип модели
            model_class: Класс модели
            
        Raises:
            ValueError: Если тип модели уже зарегистрирован
            TypeError: Если класс модели не является подклассом BaseModel
        """
        # Проверка, что тип модели не зарегистрирован
        if model_type in self.model_registry:
            raise ValueError(f"Тип модели '{model_type}' уже зарегистрирован")
        
        # Проверка, что класс модели является подклассом BaseModel
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"Класс модели должен быть подклассом BaseModel")
        
        # Регистрация типа модели
        self.model_registry[model_type] = model_class
        
        self.logger.info(f"Зарегистрирован новый тип модели: {model_type}")
    
    def create_model(self, model_type: str, 
                    model_name: str,
                    model_class: Optional[str] = None,
                    params: Optional[Dict[str, Any]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> BaseModel:
        """
        Создание модели указанного типа.
        
        Args:
            model_type: Тип модели
            model_name: Имя модели
            model_class: Класс модели (полное имя класса или объект)
            params: Параметры модели
            metadata: Метаданные модели
            
        Returns:
            BaseModel: Созданная модель
            
        Raises:
            ValueError: Если тип модели не зарегистрирован
            ImportError: Если не удалось импортировать класс модели
        """
        # Проверка, что тип модели зарегистрирован
        if model_type not in self.model_registry:
            raise ValueError(f"Тип модели '{model_type}' не зарегистрирован")
        
        # Получение класса модели
        model_class_obj = None
        
        if model_class is not None:
            # Если model_class - строка, импортируем класс
            if isinstance(model_class, str):
                try:
                    module_path, class_name = model_class.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    model_class_obj = getattr(module, class_name)
                except (ImportError, AttributeError) as e:
                    raise ImportError(f"Не удалось импортировать класс модели '{model_class}': {e}")
            else:
                model_class_obj = model_class
        
        # Инициализация параметров модели
        model_params = params or {}
        model_metadata = metadata or {}
        
        # Создание модели
        model_class_type = self.model_registry[model_type]
        model = model_class_type(
            name=model_name,
            model=model_class_obj,
            params=model_params,
            metadata=model_metadata
        )
        
        self.logger.info(f"Создана модель типа '{model_type}' с именем '{model_name}'")
        
        return model
    
    def create_regression_model(self, model_name: str,
                               model_class: Optional[str] = None,
                               params: Optional[Dict[str, Any]] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> RegressionModel:
        """
        Создание регрессионной модели.
        
        Args:
            model_name: Имя модели
            model_class: Класс модели (полное имя класса или объект)
            params: Параметры модели
            metadata: Метаданные модели
            
        Returns:
            RegressionModel: Созданная регрессионная модель
        """
        return self.create_model(
            model_type='regression',
            model_name=model_name,
            model_class=model_class,
            params=params,
            metadata=metadata
        )
    
    def create_classification_model(self, model_name: str,
                                  model_class: Optional[str] = None,
                                  params: Optional[Dict[str, Any]] = None,
                                  metadata: Optional[Dict[str, Any]] = None) -> ClassificationModel:
        """
        Создание классификационной модели.
        
        Args:
            model_name: Имя модели
            model_class: Класс модели (полное имя класса или объект)
            params: Параметры модели
            metadata: Метаданные модели
            
        Returns:
            ClassificationModel: Созданная классификационная модель
        """
        return self.create_model(
            model_type='classification',
            model_name=model_name,
            model_class=model_class,
            params=params,
            metadata=metadata
        )
    
    def create_ensemble_model(self, model_name: str,
                            model_class: Optional[str] = None,
                            params: Optional[Dict[str, Any]] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> EnsembleModel:
        """
        Создание ансамблевой модели.
        
        Args:
            model_name: Имя модели
            model_class: Класс модели (полное имя класса или объект)
            params: Параметры модели
            metadata: Метаданные модели
            
        Returns:
            EnsembleModel: Созданная ансамблевая модель
        """
        return self.create_model(
            model_type='ensemble',
            model_name=model_name,
            model_class=model_class,
            params=params,
            metadata=metadata
        )
    
    def create_time_series_model(self, model_name: str,
                               model_class: Optional[str] = None,
                               params: Optional[Dict[str, Any]] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> TimeSeriesModel:
        """
        Создание модели временных рядов.
        
        Args:
            model_name: Имя модели
            model_class: Класс модели (полное имя класса или объект)
            params: Параметры модели
            metadata: Метаданные модели
            
        Returns:
            TimeSeriesModel: Созданная модель временных рядов
        """
        return self.create_model(
            model_type='time_series',
            model_name=model_name,
            model_class=model_class,
            params=params,
            metadata=metadata
        )
    
    def create_model_from_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Создание модели на основе конфигурации.
        
        Args:
            config: Конфигурация модели
            
        Returns:
            BaseModel: Созданная модель
            
        Raises:
            ValueError: Если конфигурация не содержит обязательных полей
        """
        # Проверка наличия обязательных полей
        required_fields = ['model_type', 'model_name']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Конфигурация модели должна содержать поле '{field}'")
        
        # Извлечение параметров из конфигурации
        model_type = config['model_type']
        model_name = config['model_name']
        model_class = config.get('model_class')
        params = config.get('params', {})
        metadata = config.get('metadata', {})
        
        # Создание модели
        return self.create_model(
            model_type=model_type,
            model_name=model_name,
            model_class=model_class,
            params=params,
            metadata=metadata
        )
    
    def get_available_model_types(self) -> List[str]:
        """
        Получение списка доступных типов моделей.
        
        Returns:
            List[str]: Список доступных типов моделей
        """
        return list(self.model_registry.keys())
    
    def _initialize_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Инициализация словаря доступных моделей.
        
        Returns:
            Dict[str, Dict[str, Any]]: Словарь доступных моделей
        """
        models = {
            # Линейные модели
            "linear_regression": {
                "class": LinearRegression,
                "task_type": "regression",
                "library": "sklearn",
                "default_params": {}
            },
            "ridge": {
                "class": Ridge,
                "task_type": "regression",
                "library": "sklearn",
                "default_params": {"alpha": 1.0}
            },
            "lasso": {
                "class": Lasso,
                "task_type": "regression",
                "library": "sklearn",
                "default_params": {"alpha": 1.0}
            },
            "elastic_net": {
                "class": ElasticNet,
                "task_type": "regression",
                "library": "sklearn",
                "default_params": {"alpha": 1.0, "l1_ratio": 0.5}
            },
            "logistic_regression": {
                "class": LogisticRegression,
                "task_type": "classification",
                "library": "sklearn",
                "default_params": {"C": 1.0, "random_state": 42}
            },
            "sgd_classifier": {
                "class": SGDClassifier,
                "task_type": "classification",
                "library": "sklearn",
                "default_params": {"random_state": 42}
            },
            "sgd_regressor": {
                "class": SGDRegressor,
                "task_type": "regression",
                "library": "sklearn",
                "default_params": {"random_state": 42}
            },
            
            # Деревья решений
            "decision_tree_classifier": {
                "class": DecisionTreeClassifier,
                "task_type": "classification",
                "library": "sklearn",
                "default_params": {"random_state": 42}
            },
            "decision_tree_regressor": {
                "class": DecisionTreeRegressor,
                "task_type": "regression",
                "library": "sklearn",
                "default_params": {"random_state": 42}
            },
            
            # Ансамблевые модели
            "random_forest_classifier": {
                "class": RandomForestClassifier,
                "task_type": "classification",
                "library": "sklearn",
                "default_params": {"n_estimators": 100, "random_state": 42}
            },
            "random_forest_regressor": {
                "class": RandomForestRegressor,
                "task_type": "regression",
                "library": "sklearn",
                "default_params": {"n_estimators": 100, "random_state": 42}
            },
            "gradient_boosting_classifier": {
                "class": GradientBoostingClassifier,
                "task_type": "classification",
                "library": "sklearn",
                "default_params": {"n_estimators": 100, "random_state": 42}
            },
            "gradient_boosting_regressor": {
                "class": GradientBoostingRegressor,
                "task_type": "regression",
                "library": "sklearn",
                "default_params": {"n_estimators": 100, "random_state": 42}
            },
            "adaboost_classifier": {
                "class": AdaBoostClassifier,
                "task_type": "classification",
                "library": "sklearn",
                "default_params": {"n_estimators": 50, "random_state": 42}
            },
            "adaboost_regressor": {
                "class": AdaBoostRegressor,
                "task_type": "regression",
                "library": "sklearn",
                "default_params": {"n_estimators": 50, "random_state": 42}
            },
            
            # SVM
            "svc": {
                "class": SVC,
                "task_type": "classification",
                "library": "sklearn",
                "default_params": {"probability": True, "random_state": 42}
            },
            "svr": {
                "class": SVR,
                "task_type": "regression",
                "library": "sklearn",
                "default_params": {}
            },
            
            # k-ближайших соседей
            "knn_classifier": {
                "class": KNeighborsClassifier,
                "task_type": "classification",
                "library": "sklearn",
                "default_params": {"n_neighbors": 5}
            },
            "knn_regressor": {
                "class": KNeighborsRegressor,
                "task_type": "regression",
                "library": "sklearn",
                "default_params": {"n_neighbors": 5}
            },
            
            # Наивный Байес
            "gaussian_nb": {
                "class": GaussianNB,
                "task_type": "classification",
                "library": "sklearn",
                "default_params": {}
            },
            
            # Нейронные сети (scikit-learn)
            "mlp_classifier": {
                "class": MLPClassifier,
                "task_type": "classification",
                "library": "sklearn",
                "default_params": {"hidden_layer_sizes": (100,), "random_state": 42}
            },
            "mlp_regressor": {
                "class": MLPRegressor,
                "task_type": "regression",
                "library": "sklearn",
                "default_params": {"hidden_layer_sizes": (100,), "random_state": 42}
            }
        }
        
        # Добавление моделей XGBoost, если доступны
        if XGBOOST_AVAILABLE:
            models.update({
                "xgboost_classifier": {
                    "class": xgb.XGBClassifier,
                    "task_type": "classification",
                    "library": "xgboost",
                    "default_params": {"n_estimators": 100, "random_state": 42}
                },
                "xgboost_regressor": {
                    "class": xgb.XGBRegressor,
                    "task_type": "regression",
                    "library": "xgboost",
                    "default_params": {"n_estimators": 100, "random_state": 42}
                }
            })
        
        # Добавление моделей LightGBM, если доступны
        if LIGHTGBM_AVAILABLE:
            models.update({
                "lightgbm_classifier": {
                    "class": lgb.LGBMClassifier,
                    "task_type": "classification",
                    "library": "lightgbm",
                    "default_params": {"n_estimators": 100, "random_state": 42}
                },
                "lightgbm_regressor": {
                    "class": lgb.LGBMRegressor,
                    "task_type": "regression",
                    "library": "lightgbm",
                    "default_params": {"n_estimators": 100, "random_state": 42}
                }
            })
        
        # Добавление моделей CatBoost, если доступны
        if CATBOOST_AVAILABLE:
            models.update({
                "catboost_classifier": {
                    "class": cb.CatBoostClassifier,
                    "task_type": "classification",
                    "library": "catboost",
                    "default_params": {"iterations": 100, "random_seed": 42, "verbose": False}
                },
                "catboost_regressor": {
                    "class": cb.CatBoostRegressor,
                    "task_type": "regression",
                    "library": "catboost",
                    "default_params": {"iterations": 100, "random_seed": 42, "verbose": False}
                }
            })
        
        # Добавление моделей TensorFlow/Keras, если доступны
        if TENSORFLOW_AVAILABLE:
            models.update({
                "keras_sequential": {
                    "class": self._create_keras_sequential,
                    "task_type": "both",
                    "library": "tensorflow",
                    "default_params": {}
                }
            })
        
        # Добавление моделей PyTorch, если доступны
        if PYTORCH_AVAILABLE:
            models.update({
                "pytorch_model": {
                    "class": self._create_pytorch_model,
                    "task_type": "both",
                    "library": "pytorch",
                    "default_params": {}
                }
            })
        
        return models
    
    def _create_keras_sequential(self, task_type: str, input_dim: int, **kwargs) -> 'keras.Sequential':
        """
        Создание модели Keras Sequential.
        
        Args:
            task_type: Тип задачи ('classification' или 'regression')
            input_dim: Размерность входных данных
            **kwargs: Дополнительные параметры
            
        Returns:
            keras.Sequential: Модель Keras Sequential
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras не установлен")
        
        # Получение параметров из kwargs
        hidden_layers = kwargs.get('hidden_layers', [64, 32])
        activation = kwargs.get('activation', 'relu')
        output_activation = kwargs.get('output_activation', 'sigmoid' if task_type == 'classification' else 'linear')
        output_units = kwargs.get('output_units', 1)
        
        # Создание модели
        model = keras.Sequential()
        
        # Добавление входного слоя
        model.add(keras.layers.Dense(hidden_layers[0], activation=activation, input_dim=input_dim))
        
        # Добавление скрытых слоев
        for units in hidden_layers[1:]:
            model.add(keras.layers.Dense(units, activation=activation))
        
        # Добавление выходного слоя
        model.add(keras.layers.Dense(output_units, activation=output_activation))
        
        # Компиляция модели
        loss = kwargs.get('loss', 'binary_crossentropy' if task_type == 'classification' else 'mse')
        optimizer = kwargs.get('optimizer', 'adam')
        metrics = kwargs.get('metrics', ['accuracy'] if task_type == 'classification' else ['mae'])
        
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
        return model
    
    def _create_pytorch_model(self, task_type: str, input_dim: int, **kwargs) -> 'nn.Module':
        """
        Создание модели PyTorch.
        
        Args:
            task_type: Тип задачи ('classification' или 'regression')
            input_dim: Размерность входных данных
            **kwargs: Дополнительные параметры
            
        Returns:
            nn.Module: Модель PyTorch
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch не установлен")
        
        # Получение параметров из kwargs
        hidden_layers = kwargs.get('hidden_layers', [64, 32])
        activation = kwargs.get('activation', nn.ReLU())
        output_activation = kwargs.get('output_activation', nn.Sigmoid() if task_type == 'classification' else None)
        output_units = kwargs.get('output_units', 1)
        
        # Создание класса модели
        class PyTorchModel(nn.Module):
            def __init__(self):
                super(PyTorchModel, self).__init__()
                
                # Создание списка слоев
                layers = []
                
                # Добавление входного слоя
                layers.append(nn.Linear(input_dim, hidden_layers[0]))
                layers.append(activation)
                
                # Добавление скрытых слоев
                for i in range(len(hidden_layers) - 1):
                    layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
                    layers.append(activation)
                
                # Добавление выходного слоя
                layers.append(nn.Linear(hidden_layers[-1], output_units))
                if output_activation is not None:
                    layers.append(output_activation)
                
                # Создание последовательности слоев
                self.model = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.model(x)
        
        # Создание экземпляра модели
        model = PyTorchModel()
        
        return model
    
    def register_custom_model(self, model_name: str, model_class: Callable, 
                             task_type: str, library: str, 
                             default_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Регистрация пользовательской модели.
        
        Args:
            model_name: Имя модели
            model_class: Класс модели
            task_type: Тип задачи ('classification', 'regression' или 'both')
            library: Название библиотеки
            default_params: Параметры по умолчанию
        """
        if model_name in self.available_models:
            self.logger.warning(f"Модель с именем {model_name} уже существует и будет перезаписана")
        
        self.available_models[model_name] = {
            "class": model_class,
            "task_type": task_type,
            "library": library,
            "default_params": default_params or {}
        }
        
        self.custom_models[model_name] = model_class
        
        self.logger.info(f"Зарегистрирована пользовательская модель {model_name}")
    
    def create_model(self, model_type: str, task_type: Optional[str] = None, 
                    model_params: Optional[Dict[str, Any]] = None,
                    input_dim: Optional[int] = None) -> Any:
        """
        Создание модели указанного типа.
        
        Args:
            model_type: Тип модели
            task_type: Тип задачи ('classification' или 'regression')
            model_params: Параметры модели
            input_dim: Размерность входных данных (для нейронных сетей)
            
        Returns:
            Any: Созданная модель
            
        Raises:
            ValueError: Если модель указанного типа не найдена
            ValueError: Если тип задачи не указан для моделей, поддерживающих оба типа задач
        """
        # Проверка наличия модели указанного типа
        if model_type not in self.available_models:
            raise ValueError(f"Модель типа {model_type} не найдена. Доступные типы: {list(self.available_models.keys())}")
        
        # Получение информации о модели
        model_info = self.available_models[model_type]
        model_class = model_info["class"]
        model_task_type = model_info["task_type"]
        default_params = model_info["default_params"].copy()
        
        # Проверка соответствия типа задачи
        if model_task_type == "both" and task_type is None:
            raise ValueError(f"Для модели {model_type} необходимо указать тип задачи (classification или regression)")
        
        if model_task_type != "both" and task_type is not None and model_task_type != task_type:
            self.logger.warning(f"Модель {model_type} предназначена для задач типа {model_task_type}, но указан тип {task_type}")
        
        # Объединение параметров по умолчанию и переданных параметров
        params = default_params.copy()
        if model_params:
            params.update(model_params)
        
        # Создание модели
        if model_type in ["keras_sequential", "pytorch_model"]:
            if input_dim is None:
                raise ValueError(f"Для модели {model_type} необходимо указать размерность входных данных (input_dim)")
            
            model = model_class(task_type=task_type or model_task_type, input_dim=input_dim, **params)
        else:
            model = model_class(**params)
        
        self.logger.info(f"Создана модель типа {model_type} для задачи {task_type or model_task_type}")
        
        return model
    
    def get_available_models(self, task_type: Optional[str] = None, 
                            library: Optional[str] = None) -> List[str]:
        """
        Получение списка доступных моделей.
        
        Args:
            task_type: Тип задачи ('classification' или 'regression')
            library: Название библиотеки
            
        Returns:
            List[str]: Список доступных моделей
        """
        models = []
        
        for model_name, model_info in self.available_models.items():
            # Фильтрация по типу задачи
            if task_type and model_info["task_type"] != "both" and model_info["task_type"] != task_type:
                continue
            
            # Фильтрация по библиотеке
            if library and model_info["library"] != library:
                continue
            
            models.append(model_name)
        
        return models
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        Получение информации о модели.
        
        Args:
            model_type: Тип модели
            
        Returns:
            Dict[str, Any]: Информация о модели
            
        Raises:
            ValueError: Если модель указанного типа не найдена
        """
        if model_type not in self.available_models:
            raise ValueError(f"Модель типа {model_type} не найдена. Доступные типы: {list(self.available_models.keys())}")
        
        return self.available_models[model_type].copy()
    
    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """
        Получение параметров модели по умолчанию.
        
        Args:
            model_type: Тип модели
            
        Returns:
            Dict[str, Any]: Параметры модели по умолчанию
            
        Raises:
            ValueError: Если модель указанного типа не найдена
        """
        if model_type not in self.available_models:
            raise ValueError(f"Модель типа {model_type} не найдена. Доступные типы: {list(self.available_models.keys())}")
        
        return self.available_models[model_type]["default_params"].copy()
    
    def save_model(self, model: Any, model_path: str, 
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Сохранение модели.
        
        Args:
            model: Модель для сохранения
            model_path: Путь для сохранения модели
            metadata: Метаданные модели
            
        Returns:
            str: Путь к сохраненной модели
        """
        import pickle
        
        # Создание директории, если она не существует
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Сохранение модели
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Сохранение метаданных, если они переданы
        if metadata:
            metadata_path = f"{os.path.splitext(model_path)[0]}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
        
        self.logger.info(f"Модель сохранена в {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str) -> Any:
        """
        Загрузка модели.
        
        Args:
            model_path: Путь к модели
            
        Returns:
            Any: Загруженная модель
            
        Raises:
            FileNotFoundError: Если файл модели не найден
        """
        import pickle
        
        # Проверка существования файла
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели {model_path} не найден")
        
        # Загрузка модели
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        self.logger.info(f"Модель загружена из {model_path}")
        
        return model
    
    def load_model_metadata(self, model_path: str) -> Dict[str, Any]:
        """
        Загрузка метаданных модели.
        
        Args:
            model_path: Путь к модели
            
        Returns:
            Dict[str, Any]: Метаданные модели
            
        Raises:
            FileNotFoundError: Если файл метаданных не найден
        """
        metadata_path = f"{os.path.splitext(model_path)[0]}_metadata.json"
        
        # Проверка существования файла
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Файл метаданных {metadata_path} не найден")
        
        # Загрузка метаданных
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.logger.info(f"Метаданные модели загружены из {metadata_path}")
        
        return metadata 