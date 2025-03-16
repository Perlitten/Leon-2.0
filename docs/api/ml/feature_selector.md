# Модуль выбора признаков (FeatureSelector)

Модуль `FeatureSelector` предоставляет инструменты для отбора наиболее информативных признаков в моделях машинного обучения. Это позволяет улучшить качество моделей, уменьшить переобучение и ускорить процесс обучения.

## Класс FeatureSelector

```python
from ml.models import FeatureSelector
```

Класс `FeatureSelector` предоставляет различные методы отбора признаков:
- Фильтрация на основе статистических тестов (ANOVA, корреляция и т.д.)
- Встроенные методы (на основе моделей, например, важность признаков в деревьях)
- Рекурсивное исключение признаков
- Удаление признаков с низкой дисперсией

### Инициализация

```python
selector = FeatureSelector(
    method='k_best_anova',  # Метод отбора признаков
    n_features=10,          # Количество признаков для отбора
    percentile=20,          # Процент признаков для отбора
    threshold=0.01,         # Порог для отбора признаков
    estimator=None,         # Модель для встроенных методов отбора
    feature_names=None      # Имена признаков
)
```

### Доступные методы отбора признаков

| Метод | Описание |
|-------|----------|
| `k_best_anova` | SelectKBest с f_classif |
| `k_best_regression` | SelectKBest с f_regression |
| `k_best_mutual_info_classif` | SelectKBest с mutual_info_classif |
| `k_best_mutual_info_regression` | SelectKBest с mutual_info_regression |
| `percentile_anova` | SelectPercentile с f_classif |
| `percentile_regression` | SelectPercentile с f_regression |
| `percentile_mutual_info_classif` | SelectPercentile с mutual_info_classif |
| `percentile_mutual_info_regression` | SelectPercentile с mutual_info_regression |
| `variance_threshold` | VarianceThreshold |
| `random_forest` | SelectFromModel с RandomForest |
| `lasso` | SelectFromModel с Lasso |
| `logistic_regression` | SelectFromModel с LogisticRegression |
| `rfe` | RFE (Recursive Feature Elimination) |
| `rfecv` | RFECV (Recursive Feature Elimination with Cross-Validation) |

### Основные методы

#### fit

```python
selector.fit(X, y)
```

Обучает селектор признаков на данных.

**Параметры:**
- `X` (np.ndarray или pd.DataFrame): Матрица признаков
- `y` (np.ndarray или pd.Series): Целевая переменная

**Возвращает:**
- `self`: Обученный селектор признаков

#### transform

```python
X_transformed = selector.transform(X)
```

Преобразует данные, оставляя только выбранные признаки.

**Параметры:**
- `X` (np.ndarray или pd.DataFrame): Матрица признаков

**Возвращает:**
- Преобразованные данные (np.ndarray или pd.DataFrame)

#### fit_transform

```python
X_transformed = selector.fit_transform(X, y)
```

Обучает селектор и преобразует данные.

**Параметры:**
- `X` (np.ndarray или pd.DataFrame): Матрица признаков
- `y` (np.ndarray или pd.Series): Целевая переменная

**Возвращает:**
- Преобразованные данные (np.ndarray или pd.DataFrame)

#### get_feature_ranking

```python
ranking = selector.get_feature_ranking()
```

Возвращает ранжированный список признаков по их важности.

**Возвращает:**
- Список кортежей (имя признака, оценка важности)

#### plot_feature_importance

```python
selector.plot_feature_importance(
    top_n=20,                # Количество наиболее важных признаков для отображения
    figsize=(12, 8),         # Размер графика
    save_path=None           # Путь для сохранения графика
)
```

Визуализирует важность признаков.

#### save и load

```python
# Сохранение селектора
path = selector.save('path/to/save/selector.pkl')

# Загрузка селектора
selector = FeatureSelector().load('path/to/save/selector.pkl')
```

Сохраняет и загружает селектор признаков.

#### get_support_mask

```python
mask = selector.get_support_mask()
```

Возвращает булеву маску выбранных признаков.

#### get_params и set_params

```python
# Получение параметров
params = selector.get_params()

# Установка параметров
selector.set_params(method='lasso', threshold=0.05)
```

Получает и устанавливает параметры селектора.

## Примеры использования

### Пример 1: Отбор признаков для задачи классификации

```python
import pandas as pd
import numpy as np
from ml.models import FeatureSelector
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Загрузка данных
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание и обучение селектора признаков
selector = FeatureSelector(method='k_best_anova', n_features=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Визуализация важности признаков
selector.plot_feature_importance(top_n=10)

# Обучение модели на выбранных признаках
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_selected, y_train)

# Оценка качества модели
y_pred = model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели с выбранными признаками: {accuracy:.4f}")
```

### Пример 2: Отбор признаков для задачи регрессии

```python
import pandas as pd
import numpy as np
from ml.models import FeatureSelector
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Загрузка данных
data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание и обучение селектора признаков
selector = FeatureSelector(method='random_forest', threshold=0.05)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Получение ранжированного списка признаков
feature_ranking = selector.get_feature_ranking()
print("Ранжированный список признаков:")
for feature, score in feature_ranking:
    print(f"{feature}: {score:.4f}")

# Обучение модели на выбранных признаках
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train_selected, y_train)

# Оценка качества модели
y_pred = model.predict(X_test_selected)
mse = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка модели с выбранными признаками: {mse:.4f}")
```

### Пример 3: Сравнение различных методов отбора признаков

```python
import pandas as pd
import numpy as np
from ml.models import FeatureSelector
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

# Загрузка данных
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Список методов для сравнения
methods = ['k_best_anova', 'percentile_mutual_info_classif', 'random_forest', 'rfe']
results = {}

# Сравнение методов
for method in methods:
    # Создание и обучение селектора признаков
    selector = FeatureSelector(method=method, n_features=5)
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    # Оценка качества с помощью кросс-валидации
    model = SVC(kernel='linear', C=1.0, random_state=42)
    scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='accuracy')
    
    # Сохранение результатов
    results[method] = {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std(),
        'selected_features': selector.selected_features
    }

# Вывод результатов
for method, result in results.items():
    print(f"Метод: {method}")
    print(f"Средняя точность: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
    print(f"Выбранные признаки: {result['selected_features']}")
    print()
```

## Интеграция с другими модулями

`FeatureSelector` может быть интегрирован с другими модулями системы Leon для улучшения качества моделей машинного обучения:

```python
from ml.models import FeatureSelector, ClassificationModel
from ml.training import ModelTrainer
import pandas as pd

# Загрузка данных
data = pd.read_csv('historical_data.csv')
X = data.drop('target', axis=1)
y = data['target']

# Создание селектора признаков
selector = FeatureSelector(method='random_forest', threshold=0.05)

# Предобработка данных
X_selected = selector.fit_transform(X, y)

# Создание и обучение модели
model = ClassificationModel(name='price_direction_predictor', model_type='random_forest')
trainer = ModelTrainer(model=model)
trainer.train(X_selected, y)

# Сохранение модели и селектора
model.save('models/price_direction_predictor.pkl')
selector.save('models/feature_selector.pkl')
``` 