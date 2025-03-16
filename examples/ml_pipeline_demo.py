#!/usr/bin/env python
"""
Демонстрационный пример использования модуля машинного обучения.

Этот скрипт демонстрирует полный цикл работы с моделями машинного обучения:
1. Загрузка и подготовка данных
2. Предобработка данных
3. Выбор признаков
4. Создание и обучение моделей
5. Валидация моделей
6. Визуализация результатов
7. Сохранение и загрузка моделей
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

# Добавление корневой директории проекта в путь для импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импорт модулей машинного обучения
from ml.preprocessing import DataPreprocessor, FeatureSelector, FeatureEngineer
from ml.models import (
    BaseModel, RegressionModel, ClassificationModel, 
    ModelFactory, ModelManager
)
from ml.validation import ModelValidator, ModelVisualizer
from ml.training import ModelTrainer


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data():
    """
    Загрузка данных для демонстрации.
    
    Returns:
        tuple: Данные для регрессии и классификации
    """
    logger.info("Загрузка данных...")
    
    # Загрузка данных для регрессии
    try:
        housing = fetch_california_housing()
        X_reg = pd.DataFrame(housing.data, columns=housing.feature_names)
        y_reg = pd.Series(housing.target, name="price")
        
        logger.info(f"Загружены данные для регрессии: {X_reg.shape[0]} образцов, {X_reg.shape[1]} признаков")
    except Exception as e:
        logger.warning(f"Не удалось загрузить данные California Housing: {e}")
        logger.info("Генерация синтетических данных для регрессии...")
        
        # Генерация синтетических данных для регрессии
        np.random.seed(42)
        n_samples = 500
        n_features = 8
        
        X_reg = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        y_reg = pd.Series(
            np.random.randn(n_samples) * 2 + X_reg.sum(axis=1) * 0.5,
            name="target"
        )
        
        logger.info(f"Сгенерированы синтетические данные для регрессии: {X_reg.shape[0]} образцов, {X_reg.shape[1]} признаков")
    
    # Загрузка данных для классификации
    try:
        cancer = load_breast_cancer()
        X_clf = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        y_clf = pd.Series(cancer.target, name="target")
        
        logger.info(f"Загружены данные для классификации: {X_clf.shape[0]} образцов, {X_clf.shape[1]} признаков")
    except Exception as e:
        logger.warning(f"Не удалось загрузить данные Breast Cancer: {e}")
        logger.info("Генерация синтетических данных для классификации...")
        
        # Генерация синтетических данных для классификации
        np.random.seed(42)
        n_samples = 500
        n_features = 10
        
        X_clf = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        y_clf = pd.Series(
            (X_clf.sum(axis=1) > 0).astype(int),
            name="target"
        )
        
        logger.info(f"Сгенерированы синтетические данные для классификации: {X_clf.shape[0]} образцов, {X_clf.shape[1]} признаков")
    
    return (X_reg, y_reg), (X_clf, y_clf)


def regression_pipeline(X, y):
    """
    Демонстрация пайплайна для регрессии.
    
    Args:
        X: Матрица признаков
        y: Целевая переменная
    """
    logger.info("Запуск пайплайна для регрессии...")
    
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Размер обучающей выборки: {X_train.shape[0]} образцов")
    logger.info(f"Размер тестовой выборки: {X_test.shape[0]} образцов")
    
    # Создание директорий для сохранения результатов
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # 1. Предобработка данных
    logger.info("Предобработка данных...")
    
    preprocessor = DataPreprocessor()
    preprocessor.add_scaler("standard_scaler", scaler_type="standard")
    preprocessor.add_imputer("simple_imputer", imputer_type="simple", strategy="mean")
    
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    logger.info(f"Размер обработанных данных: {X_train_processed.shape[1]} признаков")
    
    # 2. Выбор признаков
    logger.info("Выбор признаков...")
    
    feature_selector = FeatureSelector()
    feature_selector.add_from_model(
        "rf_selector", 
        task_type="regression", 
        max_features=5
    )
    
    X_train_selected = feature_selector.fit_transform(X_train_processed, y_train)
    X_test_selected = feature_selector.transform(X_test_processed)
    
    selected_features = feature_selector.get_selected_features()
    logger.info(f"Выбрано {len(selected_features)} признаков: {selected_features}")
    
    # 3. Инженерия признаков
    logger.info("Инженерия признаков...")
    
    feature_engineer = FeatureEngineer()
    feature_engineer.add_polynomial_features(degree=2, interaction_only=True)
    
    X_train_engineered = feature_engineer.fit_transform(X_train_selected, y_train)
    X_test_engineered = feature_engineer.transform(X_test_selected)
    
    logger.info(f"Размер данных после инженерии признаков: {X_train_engineered.shape[1]} признаков")
    
    # 4. Создание и обучение моделей
    logger.info("Создание и обучение моделей...")
    
    # Создание фабрики моделей
    model_factory = ModelFactory()
    
    # Создание менеджера моделей
    model_manager = ModelManager(models_dir="models")
    
    # Создание моделей
    linear_model = model_factory.create_regression_model(
        model_name="linear_regression",
        model_class=LinearRegression,
        params={},
        metadata={"description": "Линейная регрессия для демонстрации"}
    )
    
    rf_model = model_factory.create_regression_model(
        model_name="random_forest",
        model_class=RandomForestRegressor,
        params={"n_estimators": 100, "random_state": 42},
        metadata={"description": "Случайный лес для регрессии"}
    )
    
    # Создание тренера моделей
    trainer = ModelTrainer(
        config={
            "cv": 5,
            "scoring": "neg_mean_squared_error",
            "n_jobs": -1
        }
    )
    
    # Обучение моделей
    linear_model = trainer.train(
        model=linear_model,
        X=X_train_engineered,
        y=y_train,
        X_val=X_test_engineered,
        y_val=y_test
    )
    
    rf_model = trainer.train(
        model=rf_model,
        X=X_train_engineered,
        y=y_train,
        X_val=X_test_engineered,
        y_val=y_test
    )
    
    # 5. Валидация моделей
    logger.info("Валидация моделей...")
    
    validator = ModelValidator(
        config={
            "report_dir": "reports",
            "visualizer_config": {
                "plots_dir": "reports/plots"
            }
        }
    )
    
    linear_report = validator.validate(
        model=linear_model,
        X=X_test_engineered,
        y=y_test,
        report_name="linear_regression_report"
    )
    
    rf_report = validator.validate(
        model=rf_model,
        X=X_test_engineered,
        y=y_test,
        report_name="random_forest_report"
    )
    
    # Сравнение моделей
    comparison = validator.compare_models(
        models=[linear_model, rf_model],
        X=X_test_engineered,
        y=y_test,
        report_name="regression_models_comparison"
    )
    
    # 6. Сохранение моделей
    logger.info("Сохранение моделей...")
    
    linear_model_id = model_manager.save_model(linear_model)
    rf_model_id = model_manager.save_model(rf_model)
    
    logger.info(f"Модель линейной регрессии сохранена с ID: {linear_model_id}")
    logger.info(f"Модель случайного леса сохранена с ID: {rf_model_id}")
    
    # 7. Загрузка моделей
    logger.info("Загрузка моделей...")
    
    loaded_linear_model = model_manager.load_model(linear_model_id)
    loaded_rf_model = model_manager.load_model(rf_model_id)
    
    logger.info("Модели успешно загружены")
    
    # 8. Получение лучшей модели
    best_model_id = model_manager.get_best_model(
        metric="r2_score",
        higher_is_better=True
    )
    
    if best_model_id:
        best_model = model_manager.load_model(best_model_id)
        logger.info(f"Лучшая модель: {best_model.name} (ID: {best_model_id})")
        
        # Создание отчета о лучшей модели
        model_manager.create_model_report(
            best_model_id,
            report_path="reports/best_regression_model_report.json"
        )
    else:
        logger.warning("Не удалось определить лучшую модель")
    
    logger.info("Пайплайн для регрессии завершен")


def classification_pipeline(X, y):
    """
    Демонстрация пайплайна для классификации.
    
    Args:
        X: Матрица признаков
        y: Целевая переменная
    """
    logger.info("Запуск пайплайна для классификации...")
    
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Размер обучающей выборки: {X_train.shape[0]} образцов")
    logger.info(f"Размер тестовой выборки: {X_test.shape[0]} образцов")
    
    # Создание директорий для сохранения результатов
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # 1. Предобработка данных
    logger.info("Предобработка данных...")
    
    preprocessor = DataPreprocessor()
    preprocessor.add_scaler("standard_scaler", scaler_type="standard")
    preprocessor.add_imputer("simple_imputer", imputer_type="simple", strategy="mean")
    
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    logger.info(f"Размер обработанных данных: {X_train_processed.shape[1]} признаков")
    
    # 2. Выбор признаков
    logger.info("Выбор признаков...")
    
    feature_selector = FeatureSelector()
    feature_selector.add_from_model(
        "rf_selector", 
        task_type="classification", 
        max_features=5
    )
    
    X_train_selected = feature_selector.fit_transform(X_train_processed, y_train)
    X_test_selected = feature_selector.transform(X_test_processed)
    
    selected_features = feature_selector.get_selected_features()
    logger.info(f"Выбрано {len(selected_features)} признаков: {selected_features}")
    
    # 3. Создание и обучение моделей
    logger.info("Создание и обучение моделей...")
    
    # Создание фабрики моделей
    model_factory = ModelFactory()
    
    # Создание менеджера моделей
    model_manager = ModelManager(models_dir="models")
    
    # Создание моделей
    logistic_model = model_factory.create_classification_model(
        model_name="logistic_regression",
        model_class=LogisticRegression,
        params={"max_iter": 1000, "random_state": 42},
        metadata={"description": "Логистическая регрессия для демонстрации"}
    )
    
    rf_model = model_factory.create_classification_model(
        model_name="random_forest",
        model_class=RandomForestClassifier,
        params={"n_estimators": 100, "random_state": 42},
        metadata={"description": "Случайный лес для классификации"}
    )
    
    # Создание тренера моделей
    trainer = ModelTrainer(
        config={
            "cv": 5,
            "scoring": "accuracy",
            "n_jobs": -1
        }
    )
    
    # Обучение моделей
    logistic_model = trainer.train(
        model=logistic_model,
        X=X_train_selected,
        y=y_train,
        X_val=X_test_selected,
        y_val=y_test
    )
    
    rf_model = trainer.train(
        model=rf_model,
        X=X_train_selected,
        y=y_train,
        X_val=X_test_selected,
        y_val=y_test
    )
    
    # 4. Валидация моделей
    logger.info("Валидация моделей...")
    
    validator = ModelValidator(
        config={
            "report_dir": "reports",
            "visualizer_config": {
                "plots_dir": "reports/plots"
            }
        }
    )
    
    logistic_report = validator.validate(
        model=logistic_model,
        X=X_test_selected,
        y=y_test,
        report_name="logistic_regression_report"
    )
    
    rf_report = validator.validate(
        model=rf_model,
        X=X_test_selected,
        y=y_test,
        report_name="random_forest_classification_report"
    )
    
    # Сравнение моделей
    comparison = validator.compare_models(
        models=[logistic_model, rf_model],
        X=X_test_selected,
        y=y_test,
        report_name="classification_models_comparison"
    )
    
    # 5. Сохранение моделей
    logger.info("Сохранение моделей...")
    
    logistic_model_id = model_manager.save_model(logistic_model)
    rf_model_id = model_manager.save_model(rf_model)
    
    logger.info(f"Модель логистической регрессии сохранена с ID: {logistic_model_id}")
    logger.info(f"Модель случайного леса сохранена с ID: {rf_model_id}")
    
    # 6. Загрузка моделей
    logger.info("Загрузка моделей...")
    
    loaded_logistic_model = model_manager.load_model(logistic_model_id)
    loaded_rf_model = model_manager.load_model(rf_model_id)
    
    logger.info("Модели успешно загружены")
    
    # 7. Получение лучшей модели
    best_model_id = model_manager.get_best_model(
        metric="accuracy",
        higher_is_better=True
    )
    
    if best_model_id:
        best_model = model_manager.load_model(best_model_id)
        logger.info(f"Лучшая модель: {best_model.name} (ID: {best_model_id})")
        
        # Создание отчета о лучшей модели
        model_manager.create_model_report(
            best_model_id,
            report_path="reports/best_classification_model_report.json"
        )
    else:
        logger.warning("Не удалось определить лучшую модель")
    
    logger.info("Пайплайн для классификации завершен")


def main():
    """
    Основная функция демонстрации.
    """
    logger.info("Запуск демонстрации модуля машинного обучения...")
    
    # Загрузка данных
    (X_reg, y_reg), (X_clf, y_clf) = load_data()
    
    # Запуск пайплайна для регрессии
    regression_pipeline(X_reg, y_reg)
    
    # Запуск пайплайна для классификации
    classification_pipeline(X_clf, y_clf)
    
    logger.info("Демонстрация завершена")


if __name__ == "__main__":
    main() 