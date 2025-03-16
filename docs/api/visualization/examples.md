# Примеры использования модуля визуализации

В этом документе представлены примеры использования модуля визуализации для различных сценариев.

## Базовый пример консольной визуализации

```python
from visualization.base import ConsoleVisualizer

# Создание простого консольного визуализатора
visualizer = ConsoleVisualizer(name="simple_console", config={
    "refresh_rate": 1.0,
    "width": 100,
    "height": 30,
    "clear_screen": True
})

# Запуск визуализатора
visualizer.start()

# Обновление данных
visualizer.update({
    "title": "Простая консольная визуализация",
    "content": "Это пример простой консольной визуализации",
    "data": [1, 2, 3, 4, 5]
})

# Остановка визуализатора
visualizer.stop()
```

## Торговая панель с реальными данными

```python
import time
from visualization.trading_dashboard import TradingDashboard
from core.constants import TRADING_MODES

# Создание торговой панели
dashboard = TradingDashboard(config={
    "symbol": "BTCUSDT",
    "mode": TRADING_MODES["DRY"],
    "initial_balance": 1000.0,
    "current_balance": 1050.0,
    "refresh_rate": 1.0,
    "trading_params": {
        "leverage": 20,
        "risk_per_trade": 1.0,
        "stop_loss": 0.5,
        "take_profit": 1.0
    }
})

# Запуск панели
dashboard.start()

# Обновление данных
dashboard.update({
    "current_balance": 1100.0,
    "total_trades": 5,
    "winning_trades": 3,
    "losing_trades": 2,
    "positions": [
        {
            "id": 1,
            "symbol": "BTCUSDT",
            "type": "LONG",
            "size": 0.01,
            "entry_price": 50000.0,
            "current_price": 51000.0,
            "pnl": 10.0,
            "pnl_percent": 2.0
        }
    ],
    "signals": [
        {"indicator": "RSI", "value": "44.03", "signal": "NEUTRAL"},
        {"indicator": "MACD", "value": "-168.6", "signal": "SELL"},
        {"indicator": "Bollinger", "value": "87488.5", "signal": "BUY"},
        {"indicator": "MA Cross", "value": "2432.1", "signal": "NEUTRAL"}
    ],
    "recommendation": "Рекомендуется открыть LONG позицию по BTCUSDT",
    "recommendation_color": "green"
})

# Пауза для демонстрации
time.sleep(5)

# Остановка панели
dashboard.stop()
```

## Интеграция с оркестратором

```python
import time
from core.orchestrator import LeonOrchestrator
from visualization.trading_dashboard import TradingDashboard

# Создание экземпляра оркестратора
orchestrator = LeonOrchestrator()

# Создание торговой панели
dashboard = TradingDashboard()

# Регистрация обработчиков событий
orchestrator._event_bus.subscribe("PRICE_UPDATED", lambda data: dashboard.update({"current_price": data["price"]}))
orchestrator._event_bus.subscribe("BALANCE_UPDATED", lambda data: dashboard.update({"current_balance": data["balance"]}))
orchestrator._event_bus.subscribe("POSITION_OPENED", lambda data: dashboard.update({"positions": orchestrator.get_positions()}))
orchestrator._event_bus.subscribe("POSITION_CLOSED", lambda data: dashboard.update({"positions": orchestrator.get_positions()}))
orchestrator._event_bus.subscribe("PREDICTION_RECEIVED", lambda data: dashboard.update({"signals": data["signals"]}))

# Запуск оркестратора
orchestrator.start()

# Запуск торговой панели
dashboard.start()

# Основной цикл
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Остановка при нажатии Ctrl+C
    dashboard.stop()
    orchestrator.stop()
```

## Симуляция торговли с обновлением данных

```python
import logging
import time
import random
from visualization.trading_dashboard import TradingDashboard
from core.constants import TRADING_MODES

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Создание торговой панели
dashboard = TradingDashboard(config={
    "symbol": "BTCUSDT",
    "mode": TRADING_MODES["DRY"],
    "initial_balance": 1000.0,
    "current_balance": 1000.0,
    "refresh_rate": 1.0
})

# Запуск панели
dashboard.start()

# Симуляция обновления данных
try:
    initial_balance = 1000.0
    current_balance = initial_balance
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    positions = []
    
    while True:
        # Симуляция изменения цены
        btc_price = 50000.0 + random.uniform(-1000, 1000)
        
        # Симуляция открытия/закрытия позиций
        if random.random() < 0.1 and len(positions) < 3:  # 10% шанс открыть новую позицию
            position_type = random.choice(["LONG", "SHORT"])
            position_size = round(random.uniform(0.001, 0.01), 3)
            entry_price = btc_price
            
            positions.append({
                "id": len(positions) + 1,
                "symbol": "BTCUSDT",
                "type": position_type,
                "size": position_size,
                "entry_price": entry_price,
                "current_price": btc_price,
                "pnl": 0.0,
                "pnl_percent": 0.0
            })
        
        # Обновление существующих позиций
        for position in positions:
            position["current_price"] = btc_price
            
            # Расчет PnL
            if position["type"] == "LONG":
                pnl = (position["current_price"] - position["entry_price"]) * position["size"]
                pnl_percent = (position["current_price"] / position["entry_price"] - 1) * 100
            else:  # SHORT
                pnl = (position["entry_price"] - position["current_price"]) * position["size"]
                pnl_percent = (position["entry_price"] / position["current_price"] - 1) * 100
            
            position["pnl"] = round(pnl, 2)
            position["pnl_percent"] = round(pnl_percent, 2)
        
        # Симуляция закрытия позиций
        positions_to_remove = []
        for i, position in enumerate(positions):
            if random.random() < 0.05:  # 5% шанс закрыть позицию
                total_trades += 1
                if position["pnl"] > 0:
                    winning_trades += 1
                    current_balance += position["pnl"]
                else:
                    losing_trades += 1
                    current_balance += position["pnl"]
                positions_to_remove.append(i)
        
        # Удаление закрытых позиций
        for i in sorted(positions_to_remove, reverse=True):
            positions.pop(i)
        
        # Генерация сигналов
        signals = [
            {"indicator": "RSI", "value": f"{random.uniform(30, 70):.2f}", "signal": random.choice(["BUY", "SELL", "NEUTRAL"])},
            {"indicator": "MACD", "value": f"{random.uniform(-200, 200):.2f}", "signal": random.choice(["BUY", "SELL", "NEUTRAL"])},
            {"indicator": "Bollinger", "value": f"{random.uniform(50000, 100000):.2f}", "signal": random.choice(["BUY", "SELL", "NEUTRAL"])},
            {"indicator": "MA Cross", "value": f"{random.uniform(2000, 3000):.2f}", "signal": random.choice(["BUY", "SELL", "NEUTRAL"])}
        ]
        
        # Генерация рекомендации
        buy_signals = sum(1 for s in signals if s["signal"] == "BUY")
        sell_signals = sum(1 for s in signals if s["signal"] == "SELL")
        
        if buy_signals > sell_signals:
            recommendation = "Рекомендуется открыть LONG позицию по BTCUSDT"
            recommendation_color = "green"
        elif sell_signals > buy_signals:
            recommendation = "Рекомендуется открыть SHORT позицию по BTCUSDT"
            recommendation_color = "red"
        else:
            recommendation = "Рекомендуется воздержаться от открытия позиций"
            recommendation_color = "yellow"
        
        # Обновление данных на панели
        dashboard.update({
            "current_balance": round(current_balance, 2),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "positions": positions.copy(),
            "signals": signals,
            "recommendation": recommendation,
            "recommendation_color": recommendation_color
        })
        
        # Пауза перед следующим обновлением
        time.sleep(1)
except KeyboardInterrupt:
    # Остановка панели при нажатии Ctrl+C
    dashboard.stop()
```

## Создание собственного визуализатора

```python
from typing import Dict, Any, Optional
from visualization.base import BaseVisualizer
import logging

class CustomVisualizer(BaseVisualizer):
    """
    Пример создания собственного визуализатора.
    """
    
    def __init__(self, name: str = "custom_visualizer", config: Optional[Dict[str, Any]] = None):
        """
        Инициализация визуализатора.
        
        Args:
            name: Имя визуализатора
            config: Конфигурация визуализатора
        """
        super().__init__(name, config or {})
        self.logger = logging.getLogger(f"visualization.{name}")
        self.data = {}
        self.is_running = False
        
        # Настройка параметров визуализатора
        self.output_format = self.config.get("output_format", "text")
        self.output_file = self.config.get("output_file", None)
        
        self.logger.info(f"Визуализатор {name} инициализирован с конфигурацией: {config}")
    
    def start(self) -> bool:
        """
        Запуск визуализатора.
        
        Returns:
            bool: True, если визуализатор успешно запущен
        """
        self.logger.info(f"Запуск визуализатора {self.name}")
        self.is_running = True
        
        if self.output_file:
            with open(self.output_file, 'w') as f:
                f.write(f"# Визуализатор {self.name}\n\n")
        
        return True
    
    def stop(self) -> bool:
        """
        Остановка визуализатора.
        
        Returns:
            bool: True, если визуализатор успешно остановлен
        """
        self.logger.info(f"Остановка визуализатора {self.name}")
        self.is_running = False
        return True
    
    def update(self, data: Dict[str, Any]) -> bool:
        """
        Обновление данных визуализатора.
        
        Args:
            data: Новые данные для визуализации
            
        Returns:
            bool: True, если данные успешно обновлены
        """
        self.logger.debug(f"Обновление данных визуализатора {self.name}: {data}")
        self.data.update(data)
        
        # Автоматическая отрисовка при обновлении данных
        if self.is_running:
            self.render()
        
        return True
    
    def render(self) -> str:
        """
        Отрисовка визуализации.
        
        Returns:
            str: Отрисованная визуализация
        """
        self.logger.debug(f"Отрисовка визуализатора {self.name}")
        
        # Формирование вывода в зависимости от формата
        if self.output_format == "text":
            output = self._render_text()
        elif self.output_format == "json":
            output = self._render_json()
        elif self.output_format == "html":
            output = self._render_html()
        else:
            output = str(self.data)
        
        # Вывод в файл, если указан
        if self.output_file:
            with open(self.output_file, 'a') as f:
                f.write(f"\n{output}\n")
        
        # Вывод в консоль
        print(output)
        
        return output
    
    def _render_text(self) -> str:
        """
        Отрисовка в текстовом формате.
        
        Returns:
            str: Отрисованная визуализация в текстовом формате
        """
        lines = [f"=== {self.name} ==="]
        
        for key, value in self.data.items():
            lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
    def _render_json(self) -> str:
        """
        Отрисовка в формате JSON.
        
        Returns:
            str: Отрисованная визуализация в формате JSON
        """
        import json
        return json.dumps(self.data, indent=2)
    
    def _render_html(self) -> str:
        """
        Отрисовка в формате HTML.
        
        Returns:
            str: Отрисованная визуализация в формате HTML
        """
        html = f"<div class='visualizer' id='{self.name}'>\n"
        html += f"  <h2>{self.name}</h2>\n"
        html += "  <table>\n"
        
        for key, value in self.data.items():
            html += f"    <tr><td>{key}</td><td>{value}</td></tr>\n"
        
        html += "  </table>\n"
        html += "</div>"
        
        return html

# Пример использования
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Создание визуализатора с выводом в текстовом формате
    text_visualizer = CustomVisualizer(name="text_visualizer", config={
        "output_format": "text"
    })
    
    # Создание визуализатора с выводом в формате JSON
    json_visualizer = CustomVisualizer(name="json_visualizer", config={
        "output_format": "json",
        "output_file": "visualization.json"
    })
    
    # Запуск визуализаторов
    text_visualizer.start()
    json_visualizer.start()
    
    # Обновление данных
    data = {
        "symbol": "BTCUSDT",
        "price": 50000.0,
        "volume": 1000000.0,
        "change_24h": 2.5
    }
    
    text_visualizer.update(data)
    json_visualizer.update(data)
    
    # Остановка визуализаторов
    text_visualizer.stop()
    json_visualizer.stop()
``` 