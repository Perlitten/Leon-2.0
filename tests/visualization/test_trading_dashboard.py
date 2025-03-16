"""
Тесты для модуля trading_dashboard.
"""

import unittest
from unittest.mock import MagicMock, patch
import time
from datetime import datetime

from visualization.trading_dashboard import TradingDashboard
from core.constants import TRADING_MODES


class TestTradingDashboard(unittest.TestCase):
    """Тесты для класса TradingDashboard."""
    
    def setUp(self):
        """Настройка тестов."""
        # Патчинг логгера
        self.logger_patch = patch('logging.getLogger')
        self.mock_logger = self.logger_patch.start()
        
        # Патчинг Live
        self.live_patch = patch('visualization.trading_dashboard.Live')
        self.mock_live = self.live_patch.start()
        
        # Создание экземпляра торговой панели
        self.dashboard = TradingDashboard(config={
            "symbol": "BTCUSDT",
            "mode": TRADING_MODES["DRY"],
            "initial_balance": 1000.0,
            "current_balance": 1000.0,
            "refresh_rate": 0.1
        })
    
    def tearDown(self):
        """Очистка после тестов."""
        self.logger_patch.stop()
        self.live_patch.stop()
    
    def test_init(self):
        """Тест инициализации торговой панели."""
        # Проверка инициализации данных
        self.assertEqual(self.dashboard.data["symbol"], "BTCUSDT")
        self.assertEqual(self.dashboard.data["mode"], TRADING_MODES["DRY"])
        self.assertEqual(self.dashboard.data["initial_balance"], 1000.0)
        self.assertEqual(self.dashboard.data["current_balance"], 1000.0)
        self.assertEqual(self.dashboard.data["profit"], 0.0)
        self.assertEqual(self.dashboard.data["profit_percent"], 0.0)
        self.assertEqual(self.dashboard.data["total_trades"], 0)
        self.assertEqual(self.dashboard.data["winning_trades"], 0)
        self.assertEqual(self.dashboard.data["losing_trades"], 0)
        self.assertEqual(self.dashboard.data["positions"], [])
        self.assertEqual(self.dashboard.data["signals"], [])
        
        # Проверка инициализации макета
        self.assertIsNotNone(self.dashboard.layout)
        self.assertIn("header", self.dashboard.layout)
        self.assertIn("body", self.dashboard.layout)
        self.assertIn("footer", self.dashboard.layout)
        self.assertIn("left", self.dashboard.layout["body"])
        self.assertIn("right", self.dashboard.layout["body"])
        self.assertIn("stats", self.dashboard.layout["left"])
        self.assertIn("positions", self.dashboard.layout["left"])
        self.assertIn("params", self.dashboard.layout["right"])
        self.assertIn("signals", self.dashboard.layout["right"])
    
    def test_start_stop(self):
        """Тест запуска и остановки торговой панели."""
        # Запуск панели
        result = self.dashboard.start()
        
        # Проверка результата запуска
        self.assertTrue(result)
        self.assertTrue(self.dashboard.is_running)
        
        # Проверка вызова Live.start()
        self.mock_live.return_value.start.assert_called_once()
        
        # Остановка панели
        result = self.dashboard.stop()
        
        # Проверка результата остановки
        self.assertTrue(result)
        self.assertFalse(self.dashboard.is_running)
        
        # Проверка вызова Live.stop()
        self.mock_live.return_value.stop.assert_called_once()
    
    def test_update(self):
        """Тест обновления данных торговой панели."""
        # Обновление данных
        new_data = {
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
                {"indicator": "RSI", "value": "44.03", "signal": "NEUTRAL"}
            ],
            "recommendation": "Рекомендуется открыть LONG позицию",
            "recommendation_color": "green"
        }
        
        result = self.dashboard.update(new_data)
        
        # Проверка результата обновления
        self.assertTrue(result)
        
        # Проверка обновленных данных
        self.assertEqual(self.dashboard.data["current_balance"], 1100.0)
        self.assertEqual(self.dashboard.data["total_trades"], 5)
        self.assertEqual(self.dashboard.data["winning_trades"], 3)
        self.assertEqual(self.dashboard.data["losing_trades"], 2)
        self.assertEqual(len(self.dashboard.data["positions"]), 1)
        self.assertEqual(len(self.dashboard.data["signals"]), 1)
        self.assertEqual(self.dashboard.data["recommendation"], "Рекомендуется открыть LONG позицию")
        self.assertEqual(self.dashboard.data["recommendation_color"], "green")
        
        # Проверка вычисленных данных
        self.assertEqual(self.dashboard.data["profit"], 100.0)
        self.assertEqual(self.dashboard.data["profit_percent"], 10.0)
    
    def test_update_with_live(self):
        """Тест обновления данных с активным Live."""
        # Запуск панели
        self.dashboard.start()
        
        # Обновление данных
        new_data = {"current_balance": 1100.0}
        result = self.dashboard.update(new_data)
        
        # Проверка результата обновления
        self.assertTrue(result)
        
        # Проверка вызова Live.update()
        self.mock_live.return_value.update.assert_called_once()
    
    def test_generate_layout(self):
        """Тест генерации макета панели."""
        # Генерация макета
        layout = self.dashboard.generate_layout()
        
        # Проверка результата генерации
        self.assertIsNotNone(layout)
        self.assertEqual(layout, self.dashboard.layout)
    
    def test_render(self):
        """Тест отрисовки панели."""
        # Отрисовка панели
        result = self.dashboard.render()
        
        # Проверка результата отрисовки
        self.assertIsNotNone(result)
        self.assertEqual(result, self.dashboard.layout)
    
    def test_update_with_exception(self):
        """Тест обработки исключений при обновлении данных."""
        # Патчинг метода generate_layout, чтобы он вызывал исключение
        with patch.object(self.dashboard, 'generate_layout', side_effect=Exception("Test exception")):
            # Запуск панели
            self.dashboard.start()
            
            # Обновление данных
            result = self.dashboard.update({"current_balance": 1100.0})
            
            # Проверка результата обновления
            self.assertFalse(result)
            
            # Проверка вызова логгера
            self.mock_logger.return_value.error.assert_called_once()


class TestTradingDashboardIntegration(unittest.TestCase):
    """Интеграционные тесты для класса TradingDashboard."""
    
    @patch('visualization.trading_dashboard.Live')
    def test_full_workflow(self, mock_live):
        """Тест полного рабочего процесса торговой панели."""
        # Создание экземпляра торговой панели
        dashboard = TradingDashboard(config={
            "symbol": "BTCUSDT",
            "mode": TRADING_MODES["DRY"],
            "initial_balance": 1000.0,
            "current_balance": 1000.0,
            "refresh_rate": 0.1
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
                {"indicator": "RSI", "value": "44.03", "signal": "NEUTRAL"}
            ],
            "recommendation": "Рекомендуется открыть LONG позицию",
            "recommendation_color": "green"
        })
        
        # Проверка обновленных данных
        self.assertEqual(dashboard.data["current_balance"], 1100.0)
        self.assertEqual(dashboard.data["profit"], 100.0)
        self.assertEqual(dashboard.data["profit_percent"], 10.0)
        
        # Остановка панели
        dashboard.stop()
        
        # Проверка состояния панели
        self.assertFalse(dashboard.is_running)


if __name__ == '__main__':
    unittest.main() 