# Модуль управления рисками

## Обзор

Модуль `trading.risk` предоставляет классы и функции для управления рисками при торговле. Он включает в себя инструменты для расчета размера позиции, установки стоп-лоссов и тейк-профитов, контроля максимальных убытков и управления риском на портфель.

## Классы

### RiskManager

```python
class RiskManager:
    def __init__(self, config: Dict[str, Any]):
        ...
```

Класс для управления рисками при торговле.

#### Параметры конструктора

- `config` (Dict[str, Any]): Конфигурация менеджера рисков, которая может включать:
  - `max_open_positions` (int): Максимальное количество открытых позиций (по умолчанию 5)
  - `max_daily_loss` (float): Максимальный дневной убыток в процентах от баланса (по умолчанию 5.0)
  - `max_position_size` (float): Максимальный размер позиции в процентах от баланса (по умолчанию 10.0)
  - `default_stop_loss` (float): Стоп-лосс по умолчанию в процентах от цены входа (по умолчанию 2.0)
  - `default_take_profit` (float): Тейк-профит по умолчанию в процентах от цены входа (по умолчанию 3.0)
  - `position_sizer` (Dict[str, Any]): Конфигурация калькулятора размера позиции

#### Методы

##### calculate_position_size

```python
def calculate_position_size(self, balance: float, entry_price: float, stop_loss: float,
                           method: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> float:
    ...
```

Расчет размера позиции с учетом ограничений.

**Параметры:**
- `balance` (float): Доступный баланс
- `entry_price` (float): Цена входа
- `stop_loss` (float): Цена стоп-лосса
- `method` (Optional[str], optional): Метод расчета
- `params` (Optional[Dict[str, Any]], optional): Параметры метода

**Возвращает:**
- `float`: Размер позиции

##### calculate_stop_loss

```python
def calculate_stop_loss(self, entry_price: float, direction: str) -> float:
    ...
```

Расчет уровня стоп-лосса.

**Параметры:**
- `entry_price` (float): Цена входа
- `direction` (str): Направление сделки ("BUY" или "SELL")

**Возвращает:**
- `float`: Уровень стоп-лосса

##### calculate_take_profit

```python
def calculate_take_profit(self, entry_price: float, direction: str) -> float:
    ...
```

Расчет уровня тейк-профита.

**Параметры:**
- `entry_price` (float): Цена входа
- `direction` (str): Направление сделки ("BUY" или "SELL")

**Возвращает:**
- `float`: Уровень тейк-профита

##### can_open_position

```python
def can_open_position(self, balance: float) -> Tuple[bool, str]:
    ...
```

Проверка возможности открытия новой позиции.

**Параметры:**
- `balance` (float): Текущий баланс

**Возвращает:**
- `Tuple[bool, str]`: Кортеж (можно ли открыть позицию, причина)

##### add_position

```python
def add_position(self, position: Dict[str, Any]) -> None:
    ...
```

Добавление новой позиции.

**Параметры:**
- `position` (Dict[str, Any]): Информация о позиции

##### close_position

```python
def close_position(self, position: Dict[str, Any], exit_price: float, pnl: float) -> None:
    ...
```

Закрытие позиции.

**Параметры:**
- `position` (Dict[str, Any]): Информация о позиции
- `exit_price` (float): Цена выхода
- `pnl` (float): Прибыль/убыток

##### get_position

```python
def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
    ...
```

Получение информации о позиции по ID.

**Параметры:**
- `position_id` (str): ID позиции

**Возвращает:**
- `Optional[Dict[str, Any]]`: Информация о позиции или None, если позиция не найдена

##### get_open_positions

```python
def get_open_positions(self) -> List[Dict[str, Any]]:
    ...
```

Получение списка открытых позиций.

**Возвращает:**
- `List[Dict[str, Any]]`: Список открытых позиций

##### get_closed_positions

```python
def get_closed_positions(self, limit: int = 10) -> List[Dict[str, Any]]:
    ...
```

Получение списка закрытых позиций.

**Параметры:**
- `limit` (int, optional): Максимальное количество позиций. По умолчанию 10.

**Возвращает:**
- `List[Dict[str, Any]]`: Список закрытых позиций

##### get_statistics

```python
def get_statistics(self) -> Dict[str, Any]:
    ...
```

Получение статистики торговли.

**Возвращает:**
- `Dict[str, Any]`: Статистика торговли, включая:
  - `daily_pnl`: Дневная прибыль/убыток
  - `total_pnl`: Общая прибыль/убыток
  - `win_count`: Количество выигрышных сделок
  - `loss_count`: Количество проигрышных сделок
  - `total_trades`: Общее количество сделок
  - `win_rate`: Процент выигрышных сделок
  - `open_positions_count`: Количество открытых позиций
  - `closed_positions_count`: Количество закрытых позиций

##### reset_daily_statistics

```python
def reset_daily_statistics(self) -> None:
    ...
```

Сброс дневной статистики.

##### update_config

```python
def update_config(self, config: Dict[str, Any]) -> None:
    ...
```

Обновление конфигурации менеджера рисков.

**Параметры:**
- `config` (Dict[str, Any]): Новая конфигурация

### PositionSizer

```python
class PositionSizer:
    def __init__(self, default_method: str = "fixed_risk", default_params: Dict[str, Any] = None):
        ...
```

Класс для расчета размера позиции на основе различных методов управления рисками.

#### Параметры конструктора

- `default_method` (str, optional): Метод расчета по умолчанию. По умолчанию "fixed_risk".
  - Поддерживаемые методы: "fixed_risk", "fixed_percent", "fixed_size", "kelly"
- `default_params` (Dict[str, Any], optional): Параметры метода по умолчанию

#### Методы

##### calculate

```python
def calculate(self, balance: float, entry_price: float, stop_loss: float, 
             method: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> float:
    ...
```

Расчет размера позиции.

**Параметры:**
- `balance` (float): Доступный баланс
- `entry_price` (float): Цена входа
- `stop_loss` (float): Цена стоп-лосса
- `method` (Optional[str], optional): Метод расчета
- `params` (Optional[Dict[str, Any]], optional): Параметры метода

**Возвращает:**
- `float`: Размер позиции

## Методы расчета размера позиции

### Фиксированный риск на сделку (fixed_risk)

Расчет размера позиции на основе фиксированного процента риска от баланса.

**Параметры:**
- `risk_per_trade` (float): Процент риска на сделку (по умолчанию 2.0)

**Формула:**
```
risk_amount = balance * (risk_per_trade / 100)
position_size = risk_amount / (entry_price - stop_loss)
```

### Фиксированный процент от баланса (fixed_percent)

Расчет размера позиции на основе фиксированного процента от баланса.

**Параметры:**
- `percent` (float): Процент от баланса (по умолчанию 5.0)

**Формула:**
```
position_value = balance * (percent / 100)
position_size = position_value / entry_price
```

### Фиксированный размер позиции (fixed_size)

Возвращает фиксированный размер позиции.

**Параметры:**
- `position_size` (float): Размер позиции (по умолчанию 0.001)

### Метод Келли (kelly)

Расчет размера позиции на основе критерия Келли.

**Параметры:**
- `win_rate` (float): Вероятность выигрыша (по умолчанию 0.5)
- `win_loss_ratio` (float): Соотношение выигрыша к проигрышу (по умолчанию 1.5)

**Формула:**
```
kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Ограничение до 25%
position_value = balance * kelly_fraction
position_size = position_value / entry_price
```

## Пример использования

```python
import asyncio
from trading.risk.manager import RiskManager
from trading.risk.position_sizer import PositionSizer

async def main():
    # Создание менеджера рисков
    risk_config = {
        "max_open_positions": 5,
        "max_daily_loss": 5.0,
        "max_position_size": 10.0,
        "default_stop_loss": 2.0,
        "default_take_profit": 3.0,
        "position_sizer": {
            "method": "fixed_risk",
            "params": {
                "risk_per_trade": 2.0
            }
        }
    }
    
    risk_manager = RiskManager(config=risk_config)
    
    # Расчет размера позиции
    balance = 1000.0  # USDT
    entry_price = 50000.0  # BTC/USDT
    stop_loss = 49000.0  # BTC/USDT
    
    position_size = risk_manager.calculate_position_size(
        balance=balance,
        entry_price=entry_price,
        stop_loss=stop_loss
    )
    
    print(f"Размер позиции: {position_size} BTC")
    
    # Проверка возможности открытия позиции
    can_open, reason = risk_manager.can_open_position(balance)
    print(f"Можно открыть позицию: {can_open}, причина: {reason}")
    
    # Добавление позиции
    if can_open:
        position = {
            "id": "pos-1",
            "symbol": "BTCUSDT",
            "direction": "BUY",
            "entry_price": entry_price,
            "size": position_size,
            "stop_loss": stop_loss,
            "take_profit": risk_manager.calculate_take_profit(entry_price, "BUY"),
            "status": "open",
            "timestamp": 1625097600000
        }
        
        risk_manager.add_position(position)
        
        # Закрытие позиции
        exit_price = 51000.0
        pnl = (exit_price - entry_price) * position_size
        
        risk_manager.close_position(position, exit_price, pnl)
        
        # Получение статистики
        stats = risk_manager.get_statistics()
        print(f"Статистика: {stats}")

if __name__ == "__main__":
    asyncio.run(main()) 