"""
Пакет управления рисками для Leon Trading Bot.

Предоставляет классы и функции для управления рисками при торговле.
"""

from trading.risk.manager import RiskManager
from trading.risk.position_sizer import PositionSizer

# Функция для создания контроллера рисков
def create_risk_controller(max_daily_loss: float = 5.0, max_daily_trades: int = 10):
    """
    Создает и возвращает контроллер рисков с указанными параметрами.
    
    Args:
        max_daily_loss: Максимальный дневной убыток в процентах
        max_daily_trades: Максимальное количество сделок в день
        
    Returns:
        RiskManager: Экземпляр менеджера рисков
    """
    config = {
        "max_daily_loss": max_daily_loss,
        "max_daily_trades": max_daily_trades,
        "max_open_positions": 5,
        "default_stop_loss": 2.0,
        "default_take_profit": 3.0,
        "position_sizer": {
            "method": "fixed_risk",
            "params": {
                "risk_percent": 1.0
            }
        }
    }
    
    return RiskManager(config)

__all__ = ['RiskManager', 'PositionSizer', 'create_risk_controller'] 