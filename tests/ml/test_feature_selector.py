"""
Тесты для модуля выбора признаков (FeatureSelector).
"""

import unittest
import numpy as np
import pandas as pd
import os
import tempfile
from sklearn.datasets import make_classification, make_regression
from ml.models.feature_selector import FeatureSelector


class TestFeatureSelector(unittest.TestCase):
    """Тесты для класса FeatureSelector."""

    def setUp(self):
        """Подготовка данных для тестов."""
        # Создание синтетических данных для классификации
        X_class, y_class = make_classification(
            n_samples=100, n_features=20, n_informative=10, n_redundant=5,
            n_classes=2, random_state=42
        )
        self.X_class = pd.DataFrame(
            X_class, columns=[f'feature_{i}' for i in range(X_class.shape[1])]
        )
        self.y_class = pd.Series(y_class, name='target')

        # Создание синтетических данных для регрессии
        X_reg, y_reg = make_regression(
            n_samples=100, n_features=20, n_informative=10, noise=0.1,
            random_state=42
        )
        self.X_reg = pd.DataFrame(
            X_reg, columns=[f'feature_{i}' for i in range(X_reg.shape[1])]
        )
        self.y_reg = pd.Series(y_reg, name='target')

    def test_init(self):
        """Тест инициализации селектора признаков."""
        # Тест с параметрами по умолчанию
        selector = FeatureSelector()
        self.assertEqual(selector.method, 'k_best_anova')
        self.assertEqual(selector.n_features, 10)
        self.assertEqual(selector.percentile, 20)
        self.assertEqual(selector.threshold, 0.01)
        self.assertIsNone(selector.estimator)
        self.assertIsNone(selector.feature_names)

        # Тест с пользовательскими параметрами
        selector = FeatureSelector(
            method='random_forest',
            n_features=5,
            percentile=10,
            threshold=0.05,
            feature_names=['a', 'b', 'c']
        )
        self.assertEqual(selector.method, 'random_forest')
        self.assertEqual(selector.n_features, 5)
        self.assertEqual(selector.percentile, 10)
        self.assertEqual(selector.threshold, 0.05)
        self.assertEqual(selector.feature_names, ['a', 'b', 'c'])

        # Тест с неправильным методом
        with self.assertRaises(ValueError):
            FeatureSelector(method='invalid_method')

    def test_fit_transform_classification(self):
        """Тест обучения и преобразования для задачи классификации."""
        # Тест для метода k_best_anova
        selector = FeatureSelector(method='k_best_anova', n_features=5)
        X_transformed = selector.fit_transform(self.X_class, self.y_class)
        
        # Проверка размерности преобразованных данных
        self.assertEqual(X_transformed.shape[0], self.X_class.shape[0])
        self.assertEqual(X_transformed.shape[1], 5)
        
        # Проверка, что выбраны признаки
        self.assertEqual(len(selector.selected_features), 5)
        
        # Проверка, что оценки важности признаков рассчитаны
        self.assertIsNotNone(selector.feature_scores)
        self.assertEqual(len(selector.feature_scores), self.X_class.shape[1])

    def test_fit_transform_regression(self):
        """Тест обучения и преобразования для задачи регрессии."""
        # Тест для метода k_best_regression
        selector = FeatureSelector(method='k_best_regression', n_features=5)
        X_transformed = selector.fit_transform(self.X_reg, self.y_reg)
        
        # Проверка размерности преобразованных данных
        self.assertEqual(X_transformed.shape[0], self.X_reg.shape[0])
        self.assertEqual(X_transformed.shape[1], 5)
        
        # Проверка, что выбраны признаки
        self.assertEqual(len(selector.selected_features), 5)
        
        # Проверка, что оценки важности признаков рассчитаны
        self.assertIsNotNone(selector.feature_scores)
        self.assertEqual(len(selector.feature_scores), self.X_reg.shape[1])

    def test_percentile_methods(self):
        """Тест методов на основе процентилей."""
        # Тест для метода percentile_anova
        selector = FeatureSelector(method='percentile_anova', percentile=25)
        X_transformed = selector.fit_transform(self.X_class, self.y_class)
        
        # Проверка, что выбрано примерно 25% признаков
        expected_features = int(self.X_class.shape[1] * 0.25)
        self.assertEqual(X_transformed.shape[1], expected_features)

    def test_variance_threshold(self):
        """Тест метода удаления признаков с низкой дисперсией."""
        # Создание данных с признаками с низкой дисперсией
        X = self.X_class.copy()
        X['constant'] = 1.0  # Константный признак
        X['low_var'] = np.random.normal(0, 0.001, size=X.shape[0])  # Признак с низкой дисперсией
        
        # Тест для метода variance_threshold
        selector = FeatureSelector(method='variance_threshold', threshold=0.01)
        X_transformed = selector.fit_transform(X, self.y_class)
        
        # Проверка, что константный признак и признак с низкой дисперсией удалены
        self.assertNotIn('constant', selector.selected_features)
        self.assertNotIn('low_var', selector.selected_features)

    def test_model_based_methods(self):
        """Тест методов на основе моделей."""
        # Тест для метода random_forest
        selector = FeatureSelector(method='random_forest', threshold=0.05)
        X_transformed = selector.fit_transform(self.X_class, self.y_class)
        
        # Проверка, что выбраны признаки
        self.assertGreater(len(selector.selected_features), 0)
        
        # Тест для метода lasso
        selector = FeatureSelector(method='lasso', threshold=0.01)
        X_transformed = selector.fit_transform(self.X_reg, self.y_reg)
        
        # Проверка, что выбраны признаки
        self.assertGreater(len(selector.selected_features), 0)

    def test_recursive_methods(self):
        """Тест методов рекурсивного исключения признаков."""
        # Тест для метода rfe
        selector = FeatureSelector(method='rfe', n_features=5)
        X_transformed = selector.fit_transform(self.X_class, self.y_class)
        
        # Проверка размерности преобразованных данных
        self.assertEqual(X_transformed.shape[1], 5)
        
        # Проверка, что выбраны признаки
        self.assertEqual(len(selector.selected_features), 5)

    def test_get_feature_ranking(self):
        """Тест получения ранжированного списка признаков."""
        # Обучение селектора
        selector = FeatureSelector(method='k_best_anova', n_features=5)
        selector.fit(self.X_class, self.y_class)
        
        # Получение ранжированного списка признаков
        ranking = selector.get_feature_ranking()
        
        # Проверка формата ранжированного списка
        self.assertEqual(len(ranking), self.X_class.shape[1])
        self.assertIsInstance(ranking[0], tuple)
        self.assertEqual(len(ranking[0]), 2)
        
        # Проверка, что список отсортирован по убыванию оценок важности
        for i in range(len(ranking) - 1):
            self.assertGreaterEqual(ranking[i][1], ranking[i + 1][1])

    def test_plot_feature_importance(self):
        """Тест визуализации важности признаков."""
        # Обучение селектора
        selector = FeatureSelector(method='k_best_anova', n_features=5)
        selector.fit(self.X_class, self.y_class)
        
        # Создание временного файла для сохранения графика
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            # Визуализация важности признаков с сохранением графика
            selector.plot_feature_importance(top_n=5, save_path=tmp.name)
            
            # Проверка, что файл создан и не пустой
            self.assertTrue(os.path.exists(tmp.name))
            self.assertGreater(os.path.getsize(tmp.name), 0)

    def test_save_load(self):
        """Тест сохранения и загрузки селектора признаков."""
        # Обучение селектора
        selector = FeatureSelector(method='k_best_anova', n_features=5)
        selector.fit(self.X_class, self.y_class)
        
        # Создание временного файла для сохранения селектора
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            try:
                # Сохранение селектора
                path = selector.save(tmp.name)
                
                # Проверка, что файл создан и не пустой
                self.assertTrue(os.path.exists(path))
                self.assertGreater(os.path.getsize(path), 0)
                
                # Загрузка селектора
                loaded_selector = FeatureSelector().load(path)
                
                # Проверка, что загруженный селектор имеет те же параметры
                self.assertEqual(loaded_selector.method, selector.method)
                self.assertEqual(loaded_selector.n_features, selector.n_features)
                self.assertEqual(loaded_selector.selected_features, selector.selected_features)
                
                # Проверка, что загруженный селектор может преобразовывать данные
                X_transformed = loaded_selector.transform(self.X_class)
                self.assertEqual(X_transformed.shape[1], 5)
            finally:
                # Удаление временного файла
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def test_get_support_mask(self):
        """Тест получения маски выбранных признаков."""
        # Обучение селектора
        selector = FeatureSelector(method='k_best_anova', n_features=5)
        selector.fit(self.X_class, self.y_class)
        
        # Получение маски выбранных признаков
        mask = selector.get_support_mask()
        
        # Проверка формата маски
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.shape, (self.X_class.shape[1],))
        self.assertEqual(mask.dtype, bool)
        
        # Проверка, что количество True в маске соответствует количеству выбранных признаков
        self.assertEqual(np.sum(mask), 5)

    def test_get_set_params(self):
        """Тест получения и установки параметров селектора."""
        # Создание селектора
        selector = FeatureSelector(method='k_best_anova', n_features=5)
        
        # Получение параметров
        params = selector.get_params()
        
        # Проверка параметров
        self.assertEqual(params['method'], 'k_best_anova')
        self.assertEqual(params['n_features'], 5)
        
        # Установка новых параметров
        selector.set_params(method='lasso', threshold=0.05)
        
        # Проверка, что параметры изменились
        self.assertEqual(selector.method, 'lasso')
        self.assertEqual(selector.threshold, 0.05)
        
        # Проверка, что селектор сброшен
        self.assertIsNone(selector.selector)
        
        # Проверка, что установка неизвестного параметра вызывает ошибку
        with self.assertRaises(ValueError):
            selector.set_params(unknown_param=42)

    def test_str_repr(self):
        """Тест строкового представления селектора."""
        # Создание селектора
        selector = FeatureSelector(method='k_best_anova', n_features=5)
        
        # Проверка строкового представления
        str_repr = str(selector)
        self.assertIn('FeatureSelector', str_repr)
        self.assertIn("method='k_best_anova'", str_repr)
        self.assertIn("n_features=5", str_repr)
        
        # Проверка представления для отладки
        repr_str = repr(selector)
        self.assertEqual(str_repr, repr_str)


if __name__ == '__main__':
    unittest.main() 