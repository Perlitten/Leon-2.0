"""
Конфигурация для pytest.
"""

import os
import sys
import pytest

# Добавляем корневую директорию проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def temp_dir(tmpdir):
    """
    Фикстура для создания временной директории.
    
    Args:
        tmpdir: Встроенная фикстура pytest для создания временной директории
        
    Returns:
        py.path.local: Путь к временной директории
    """
    return tmpdir

@pytest.fixture
def set_dry_mode():
    """
    Фикстура для установки и сброса режима dry_mode.
    
    Yields:
        function: Функция для установки режима dry_mode
    """
    from core.localization import set_dry_mode
    
    # Сохраняем текущее состояние
    from core.localization import _dry_mode
    original_dry_mode = _dry_mode
    
    # Возвращаем функцию для установки режима
    yield set_dry_mode
    
    # Восстанавливаем исходное состояние
    set_dry_mode(original_dry_mode) 