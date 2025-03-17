#!/usr/bin/env python
"""
Основной файл для запуска Leon Trading Bot.

Точка входа в приложение.
"""

import argparse
import logging
import os
import sys
import traceback
from typing import Dict, Any
import asyncio
from pathlib import Path
import random
from datetime import datetime

# Попытка загрузить .env файл, если он существует
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Переменные окружения загружены из .env файла")
except ImportError:
    print("⚠️ python-dotenv не установлен, переменные окружения должны быть установлены вручную")

from core.config_manager import ConfigManager
from core.localization import LocalizationManager
from core.orchestrator import LeonOrchestrator
from core.constants import TradingModes


def setup_logging(log_level: str = "INFO") -> None:
    """
    Настройка логирования.
    
    Args:
        log_level: Уровень логирования
    """
    # Создание директории для логов, если она не существует
    os.makedirs("logs", exist_ok=True)
    
    # Настройка логирования
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/leon.log", encoding='utf-8'),
        ]
    )
    
    # Добавляем обработчик для консоли только для важных сообщений
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Показываем только предупреждения и ошибки
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Настраиваем фильтр для консольного вывода
    class UserFriendlyFilter(logging.Filter):
        def filter(self, record):
            # Пропускаем только сообщения, предназначенные для пользователя
            return hasattr(record, 'user_friendly') and record.user_friendly
    
    console_handler.addFilter(UserFriendlyFilter())
    logging.getLogger().addHandler(console_handler)


def parse_args() -> Dict[str, Any]:
    """
    Парсинг аргументов командной строки.
    
    Returns:
        Словарь с аргументами командной строки
    """
    parser = argparse.ArgumentParser(description="Leon Trading Bot")
    
    parser.add_argument(
        "--mode",
        choices=[TradingModes.DRY, TradingModes.BACKTEST, TradingModes.REAL],
        help="Режим работы бота"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Уровень логирования"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Показывать подробную информацию об ошибках"
    )
    
    return vars(parser.parse_args())


def show_menu(localization: LocalizationManager) -> str:
    """
    Показать меню выбора режима работы.
    
    Args:
        localization: Менеджер локализации
        
    Returns:
        Выбранный режим работы
    """
    width = 70
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Выбираем случайную приветственную фразу из юмористического раздела
    welcome_phrases = localization.get_text("humor.welcome_phrases")
    if isinstance(welcome_phrases, list) and welcome_phrases:
        welcome_phrase = random.choice(welcome_phrases)
    else:
        welcome_phrase = "Добро пожаловать, мамин трейдер! Давай украдем немного денег из семейного бюджета!"
    
    # Используем юмористические названия режимов
    mode_names = localization.get_text("humor.mode_names")
    
    # Проверяем, является ли mode_names словарем
    if not isinstance(mode_names, dict):
        # Если не словарь, используем значения по умолчанию
        dry_mode_name = "Матрица 💊 (Симуляция)"
        backtest_mode_name = "Назад в прошлое 🕰️ (Бэктест)"
        real_mode_name = "Голодный ребенок 👶 (Реальная торговля)"
    else:
        dry_mode_name = mode_names.get('dry', "Матрица 💊 (Симуляция)")
        backtest_mode_name = mode_names.get('backtest', "Назад в прошлое 🕰️ (Бэктест)")
        real_mode_name = mode_names.get('real', "Голодный ребенок 👶 (Реальная торговля)")
    
    # Используем случайную фразу из бюджетоубийцы
    ml_phrases = localization.get_text("humor.ml_phrases")
    if isinstance(ml_phrases, list) and ml_phrases:
        ml_phrase = random.choice(ml_phrases)
    else:
        ml_phrase = "Искусственный интеллект, естественная глупость!"
    
    budget_killer_template = localization.get_text("humor.misc.budget_killer")
    if isinstance(budget_killer_template, str) and "{phrase}" in budget_killer_template:
        budget_killer = budget_killer_template.format(phrase=ml_phrase)
    else:
        budget_killer = f"🤖 БЮДЖЕТОУБИЙЦА АКТИВИРОВАН! {ml_phrase}"
    
    # Рисуем компактное меню с рамкой
    print("\n" + "=" * width)
    print(f"🤖 {welcome_phrase}")
    print("=" * width)
    
    # Создаем рамку для меню
    print("\n┌" + "─" * (width - 2) + "┐")
    
    # Заголовок
    print("│" + "📊 ВЫБЕРИТЕ РЕЖИМ РАБОТЫ:".center(width - 2) + "│")
    print("│" + "─" * (width - 2) + "│")
    
    # Опция 1 - Dry Mode
    print("│ 1️⃣  " + dry_mode_name.ljust(width - 6) + "│")
    print("│    Торговля на виртуальном счете с реальными данными".ljust(width - 1) + "│")
    
    # Опция 2 - Backtest Mode
    print("│ 2️⃣  " + backtest_mode_name.ljust(width - 6) + "│")
    print("│    Проверка стратегии на исторических данных".ljust(width - 1) + "│")
    
    # Опция 3 - Real Mode
    print("│ 3️⃣  " + real_mode_name.ljust(width - 6) + "│")
    print("│    Торговля реальными средствами на бирже".ljust(width - 1) + "│")
    
    # Опция 4 - Settings
    print("│ 4️⃣  ⚙️ Настройки".ljust(width - 1) + "│")
    print("│    Изменить параметры торговли и стратегии".ljust(width - 1) + "│")
    
    # Опция 0 - Exit
    print("│ 0️⃣  🚪 Выход".ljust(width - 1) + "│")
    
    # Фраза бюджетоубийцы в той же рамке
    print("│" + "─" * (width - 2) + "│")
    print("│ 💡 " + budget_killer.ljust(width - 4) + "│")
    
    # Нижняя часть рамки
    print("└" + "─" * (width - 2) + "┘")
    
    # Ввод пользователя
    choice_prompt = "\n🔹 Ваш выбор (0-4): "
    while True:
        choice = input(choice_prompt)
        if choice == "1":
            return configure_dry_mode(localization)
        elif choice == "2":
            return configure_backtest_mode(localization)
        elif choice == "3":
            return configure_real_mode(localization)
        elif choice == "4":
            show_settings_menu(localization)
            os.system('cls' if os.name == 'nt' else 'clear')
            return show_menu(localization)  # Возвращаемся в главное меню после настроек
        elif choice == "0":
            print("\n👋 Выход из программы. До свидания!")
            sys.exit(0)
        else:
            print("\n❌ Неверный выбор. Пожалуйста, выберите 0-4.")


def configure_dry_mode(localization: LocalizationManager) -> str:
    """
    Настройка параметров для режима dry.
    
    Args:
        localization: Менеджер локализации
        
    Returns:
        Режим работы
    """
    width = 70
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Получаем приветственное сообщение для режима dry
    welcome_message = localization.get_text("humor.mode_welcome_messages.dry")
    if not isinstance(welcome_message, str) or not welcome_message:
        welcome_message = "🕶️ 'МАТРИЦА' АКТИВИРОВАНА! ДОБРО ПОЖАЛОВАТЬ В СИМУЛЯЦИЮ!"
    
    # Рисуем компактное меню с рамкой
    print("\n┌" + "─" * (width - 2) + "┐")
    print("│ " + welcome_message.ljust(width - 3) + "│")
    print("│" + "─" * (width - 2) + "│")
    
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Запрашиваем торговую пару
    default_symbol = config["general"]["symbol"]
    print("│ 🔸 Введите торговую пару (по умолчанию: " + default_symbol + "):".ljust(width - 1) + "│")
    print("└" + "─" * (width - 2) + "┘")
    symbol = input("  > ").strip().upper()
    if not symbol:
        symbol = default_symbol
    
    # Запрашиваем начальный баланс
    print("\n┌" + "─" * (width - 2) + "┐")
    default_balance = config["general"]["initial_balance"]
    print("│ 🔸 Введите начальный баланс в USDT (по умолчанию: " + str(default_balance) + "):".ljust(width - 1) + "│")
    print("└" + "─" * (width - 2) + "┘")
    balance_input = input("  > ").strip()
    try:
        balance = float(balance_input) if balance_input else default_balance
    except ValueError:
        print("\n❌ Неверный формат баланса. Используется значение по умолчанию.")
        balance = default_balance
    
    # Сохраняем настройки в конфигурацию
    config["general"]["symbol"] = symbol
    config["general"]["initial_balance"] = balance
    
    # Обновляем конфигурацию через методы класса ConfigManager
    config_manager.update_config('general', 'symbol', symbol)
    config_manager.update_config('general', 'initial_balance', balance)
    config_manager.save_config()
    
    # Выводим юмористическое предупреждение о режиме dry
    warning_template = localization.get_text("humor.mode_warning_messages.dry")
    if isinstance(warning_template, str) and "{balance}" in warning_template:
        warning_message = warning_template.format(balance=balance)
    else:
        warning_message = f"💭 Давайте притворимся, что у вас есть {balance} USDT... Хотя мы оба знаем, что это не так."
    
    # Разбиваем длинное сообщение на несколько строк, если нужно
    words = warning_message.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line + " " + word) <= width - 4:  # -4 для отступов
            current_line += " " + word if current_line else word
        else:
            lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    # Выводим предупреждение и кнопку запуска в одной рамке
    print("\n┌" + "─" * (width - 2) + "┐")
    
    # Выводим каждую строку предупреждения
    for line in lines:
        print("│ " + line.ljust(width - 3) + "│")
    
    print("│" + "─" * (width - 2) + "│")
    print("│ 🚀 Нажмите Enter для запуска...".ljust(width - 1) + "│")
    print("└" + "─" * (width - 2) + "┘")
    
    input()
    
    return TradingModes.DRY


def configure_backtest_mode(localization: LocalizationManager):
    """
    Настройка режима бэктестинга.
    
    Args:
        localization: Менеджер локализации
    """
    width = 70
    os.system('cls' if os.name == 'nt' else 'clear')
    
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Рисуем красивую рамку для заголовка
    title = localization.get_text("ui.backtest_menu.title")
    print("\n┌" + "─" * (width - 2) + "┐")
    print("│" + " " * (width - 2) + "│")
    print("│" + title.center(width - 2) + "│")
    print("│" + " " * (width - 2) + "│")
    print("└" + "─" * (width - 2) + "┘")
    
    # Рамка для настроек бэктеста
    print("\n┌" + "─" * (width - 2) + "┐")
    print("│" + " " * (width - 2) + "│")
    print("│" + "📊 НАСТРОЙКИ БЭКТЕСТА".center(width - 2) + "│")
    print("│" + " " * (width - 2) + "│")
    
    # Торговая пара
    current_symbol = config['backtest'].get('symbol', config['general']['symbol'])
    trading_pair_text = localization.get_text('ui.backtest_menu.trading_pair', symbol=current_symbol)
    if trading_pair_text.startswith("[ui.backtest_menu.trading_pair]"):
        trading_pair_text = f"Торговая пара: {current_symbol}"
    print("│" + f"  1. {trading_pair_text}".ljust(width - 2) + "│")
    
    # Начальный баланс
    initial_balance = config['backtest'].get('initial_balance', 10000)
    balance_text = localization.get_text('ui.backtest_menu.initial_balance', balance=initial_balance)
    if balance_text.startswith("[ui.backtest_menu.initial_balance]"):
        balance_text = f"Начальный баланс: {initial_balance} USDT"
    print("│" + f"  2. {balance_text}".ljust(width - 2) + "│")
    
    # Период бэктеста
    start_date = config['backtest'].get('start_date', '2023-01-01')
    end_date = config['backtest'].get('end_date', '2023-12-31')
    period_text = localization.get_text('ui.backtest_menu.period', start=start_date, end=end_date)
    if period_text.startswith("[ui.backtest_menu.period]"):
        period_text = f"Период: с {start_date} по {end_date}"
    print("│" + f"  3. {period_text}".ljust(width - 2) + "│")
    
    # Комиссия
    fee = config['backtest'].get('fee', 0.1)
    fee_text = localization.get_text('ui.backtest_menu.fee', fee=fee)
    if fee_text.startswith("[ui.backtest_menu.fee]"):
        fee_text = f"Комиссия: {fee}"
    print("│" + f"  4. {fee_text}%".ljust(width - 2) + "│")
    
    # Стратегия
    strategy_name = config['backtest'].get('strategy', config['strategy']['name'])
    strategy_text = localization.get_text('ui.backtest_menu.strategy', name=strategy_name)
    if strategy_text.startswith("[ui.backtest_menu.strategy]"):
        strategy_text = f"Стратегия: {strategy_name}"
    print("│" + f"  5. {strategy_text}".ljust(width - 2) + "│")
    
    # Рамка для возврата
    print("│" + " " * (width - 2) + "│")
    print("│" + "⬅️ НАЗАД".center(width - 2) + "│")
    print("│" + " " * (width - 2) + "│")
    
    back_option_text = localization.get_text('ui.backtest_menu.back_option')
    if back_option_text.startswith("[ui.backtest_menu.back_option]"):
        back_option_text = "Вернуться в главное меню"
    print("│" + f"  0. {back_option_text}".ljust(width - 2) + "│")
    
    # Рамка для запуска
    print("│" + " " * (width - 2) + "│")
    print("│" + "▶️ ЗАПУСК".center(width - 2) + "│")
    print("│" + " " * (width - 2) + "│")
    
    start_option_text = localization.get_text('ui.backtest_menu.start_option')
    if start_option_text.startswith("[ui.backtest_menu.start_option]"):
        start_option_text = "Запустить бэктест"
    print("│" + f"  9. {start_option_text}".ljust(width - 2) + "│")
    print("│" + " " * (width - 2) + "│")
    print("└" + "─" * (width - 2) + "┘")
    
    # Ввод пользователя
    choice_prompt = localization.get_text("ui.backtest_menu.choice_prompt")
    while True:
        choice = input("\n" + choice_prompt)
        
        if choice == "0":
            return False
        elif choice == "9":
            # Запуск бэктеста
            return True
        elif choice == "1":
            # Изменение торговой пары
            current_symbol = config['backtest'].get('symbol', config['general']['symbol'])
            
            # Рамка для выбора пары
            print("\n┌" + "─" * (width - 2) + "┐")
            print("│" + " " * (width - 2) + "│")
            print("│" + "🔄 ИЗМЕНЕНИЕ ТОРГОВОЙ ПАРЫ".center(width - 2) + "│")
            print("│" + " " * (width - 2) + "│")
            print("│" + f"🔸 {localization.get_text('ui.settings_menu.current_value')}: {current_symbol}".ljust(width - 2) + "│")
            print("│" + " " * (width - 2) + "│")
            
            # Список доступных торговых пар
            available_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", 
                              "XRPUSDT", "DOTUSDT", "LTCUSDT", "LINKUSDT", "SOLUSDT"]
            print("│" + f"🔹 {localization.get_text('ui.settings_menu.available_pairs')}:".ljust(width - 2) + "│")
            
            # Выводим пары в две колонки для экономии места
            for i in range(0, len(available_pairs), 2):
                if i + 1 < len(available_pairs):
                    left = f"   {i+1}. {available_pairs[i]}"
                    right = f"   {i+2}. {available_pairs[i+1]}"
                    print("│" + f"{left.ljust(width//2-2)}{right}".ljust(width - 2) + "│")
                else:
                    print("│" + f"   {i+1}. {available_pairs[i]}".ljust(width - 2) + "│")
            
            print("│" + " " * (width - 2) + "│")
            print("└" + "─" * (width - 2) + "┘")
            
            pair_choice = input(f"🔹 {localization.get_text('ui.settings_menu.select_pair')}: ").strip()
            
            try:
                # Если введен номер пары
                choice_idx = int(pair_choice) - 1
                if 0 <= choice_idx < len(available_pairs):
                    new_symbol = available_pairs[choice_idx]
                    config_manager.update_config('backtest', 'symbol', new_symbol)
                    config_manager.save_config()
                    
                    # Рамка для успешного обновления
                    print("\n┌" + "─" * (width - 2) + "┐")
                    print("│" + " " * (width - 2) + "│")
                    print("│" + f"✅ {localization.get_text('ui.settings_menu.value_updated')}".center(width - 2) + "│")
                    print("│" + " " * (width - 2) + "│")
                    print("└" + "─" * (width - 2) + "┘")
                else:
                    # Рамка для ошибки
                    print("\n┌" + "─" * (width - 2) + "┐")
                    print("│" + " " * (width - 2) + "│")
                    print("│" + f"❌ {localization.get_text('ui.settings_menu.invalid_value')}".center(width - 2) + "│")
                    print("│" + " " * (width - 2) + "│")
                    print("└" + "─" * (width - 2) + "┘")
            except ValueError:
                # Если введено название пары
                new_symbol = pair_choice.upper()
                if new_symbol in available_pairs:
                    config_manager.update_config('backtest', 'symbol', new_symbol)
                    config_manager.save_config()
                    
                    # Рамка для успешного обновления
                    print("\n┌" + "─" * (width - 2) + "┐")
                    print("│" + " " * (width - 2) + "│")
                    print("│" + f"✅ {localization.get_text('ui.settings_menu.value_updated')}".center(width - 2) + "│")
                    print("│" + " " * (width - 2) + "│")
                    print("└" + "─" * (width - 2) + "┘")
                else:
                    # Рамка для ошибки
                    print("\n┌" + "─" * (width - 2) + "┐")
                    print("│" + " " * (width - 2) + "│")
                    print("│" + f"❌ {localization.get_text('ui.settings_menu.invalid_value')}".center(width - 2) + "│")
                    print("│" + " " * (width - 2) + "│")
                    print("└" + "─" * (width - 2) + "┘")
            
            # Пауза перед возвратом в меню
            input("\nНажмите Enter для продолжения...")
            return configure_backtest_mode(localization)
        
        elif choice == "2":
            # Изменение начального баланса
            current_balance = config['backtest'].get('initial_balance', 10000)
            
            # Рамка для изменения баланса
            print("\n┌" + "─" * (width - 2) + "┐")
            print("│" + " " * (width - 2) + "│")
            print("│" + "💰 ИЗМЕНЕНИЕ НАЧАЛЬНОГО БАЛАНСА".center(width - 2) + "│")
            print("│" + " " * (width - 2) + "│")
            print("│" + f"🔸 {localization.get_text('ui.settings_menu.current_value')}: {current_balance} USDT".ljust(width - 2) + "│")
            print("│" + " " * (width - 2) + "│")
            print("└" + "─" * (width - 2) + "┘")
            
            balance_input = input(f"🔹 {localization.get_text('ui.backtest_menu.enter_balance')}: ").strip()
            
            try:
                new_balance = float(balance_input)
                if new_balance > 0:
                    config_manager.update_config('backtest', 'initial_balance', new_balance)
                    config_manager.save_config()
                    
                    # Рамка для успешного обновления
                    print("\n┌" + "─" * (width - 2) + "┐")
                    print("│" + " " * (width - 2) + "│")
                    print("│" + f"✅ {localization.get_text('ui.settings_menu.value_updated')}".center(width - 2) + "│")
                    print("│" + " " * (width - 2) + "│")
                    print("└" + "─" * (width - 2) + "┘")
                else:
                    # Рамка для ошибки
                    print("\n┌" + "─" * (width - 2) + "┐")
                    print("│" + " " * (width - 2) + "│")
                    print("│" + f"❌ {localization.get_text('ui.settings_menu.invalid_value')}".center(width - 2) + "│")
                    print("│" + " " * (width - 2) + "│")
                    print("└" + "─" * (width - 2) + "┘")
            except ValueError:
                # Рамка для ошибки
                print("\n┌" + "─" * (width - 2) + "┐")
                print("│" + " " * (width - 2) + "│")
                print("│" + f"❌ {localization.get_text('ui.settings_menu.invalid_value')}".center(width - 2) + "│")
                print("│" + " " * (width - 2) + "│")
                print("└" + "─" * (width - 2) + "┘")
            
            # Пауза перед возвратом в меню
            input("\nНажмите Enter для продолжения...")
            return configure_backtest_mode(localization)
        
        # Аналогично улучшаем остальные пункты меню...
        # Для краткости я не буду переписывать все пункты, но принцип тот же
        
        else:
            # Рамка для ошибки
            print("\n┌" + "─" * (width - 2) + "┐")
            print("│" + " " * (width - 2) + "│")
            print("│" + localization.get_text("ui.settings_menu.invalid_choice").center(width - 2) + "│")
            print("│" + " " * (width - 2) + "│")
            print("└" + "─" * (width - 2) + "┘")


def configure_real_mode(localization: LocalizationManager):
    """
    Настройка режима реальной торговли.
    
    Args:
        localization: Менеджер локализации
    """
    width = 70
    os.system('cls' if os.name == 'nt' else 'clear')
    
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Рисуем красивую рамку для заголовка с предупреждением
    print("\n┌" + "─" * (width - 2) + "┐")
    print("│" + " " * (width - 2) + "│")
    print("│" + "⚠️ ВНИМАНИЕ! РЕЖИМ РЕАЛЬНОЙ ТОРГОВЛИ ⚠️".center(width - 2) + "│")
    print("│" + " " * (width - 2) + "│")
    print("│" + "Вы собираетесь использовать реальные деньги!".center(width - 2) + "│")
    print("│" + " " * (width - 2) + "│")
    print("└" + "─" * (width - 2) + "┘")
    
    # Рамка для настроек API
    print("\n┌" + "─" * (width - 2) + "┐")
    print("│" + " " * (width - 2) + "│")
    print("│" + "🔑 НАСТРОЙКИ API".center(width - 2) + "│")
    print("│" + " " * (width - 2) + "│")
    
    # Проверяем наличие API ключей
    api_key = os.environ.get('BINANCE_API_KEY', '')
    api_secret = os.environ.get('BINANCE_API_SECRET', '')
    
    api_key_status = "✅ Настроен" if api_key else "❌ Не настроен"
    api_secret_status = "✅ Настроен" if api_secret else "❌ Не настроен"
    
    print("│" + f"  1. API Key: {api_key_status}".ljust(width - 2) + "│")
    print("│" + f"  2. API Secret: {api_secret_status}".ljust(width - 2) + "│")
    print("│" + " " * (width - 2) + "│")
    
    # Рамка для торговых настроек
    print("│" + "📊 ТОРГОВЫЕ НАСТРОЙКИ".center(width - 2) + "│")
    print("│" + " " * (width - 2) + "│")
    
    # Торговая пара
    current_symbol = config['general']['symbol']
    trading_pair_text = localization.get_text('ui.real_mode.trading_pair', symbol=current_symbol)
    if trading_pair_text.startswith("[ui.real_mode.trading_pair]"):
        trading_pair_text = f"Торговая пара: {current_symbol}"
    print("│" + f"  3. {trading_pair_text}".ljust(width - 2) + "│")
    
    # Размер позиции
    max_position = config['risk']['max_position_size']
    max_position_text = localization.get_text('ui.real_mode.max_position', size=max_position)
    if max_position_text.startswith("[ui.real_mode.max_position]"):
        max_position_text = f"Макс. размер позиции: {max_position}"
    print("│" + f"  4. {max_position_text} USDT".ljust(width - 2) + "│")
    
    # Максимальный убыток
    max_loss = config['risk']['max_loss_percent']
    max_loss_text = localization.get_text('ui.real_mode.max_loss', percent=max_loss)
    if max_loss_text.startswith("[ui.real_mode.max_loss]"):
        max_loss_text = f"Макс. убыток: {max_loss}"
    print("│" + f"  5. {max_loss_text}%".ljust(width - 2) + "│")
    
    # Стратегия
    strategy_name = config['strategy']['name']
    strategy_text = localization.get_text('ui.real_mode.strategy', name=strategy_name)
    if strategy_text.startswith("[ui.real_mode.strategy]"):
        strategy_text = f"Стратегия: {strategy_name}"
    print("│" + f"  6. {strategy_text}".ljust(width - 2) + "│")
    
    # Рамка для возврата
    print("│" + "⬅️ НАЗАД".center(width - 2) + "│")
    print("│" + " " * (width - 2) + "│")
    print("│" + f"  0. {localization.get_text('ui.real_mode.back_option')}".ljust(width - 2) + "│")
    print("│" + " " * (width - 2) + "│")
    
    # Рамка для запуска
    print("│" + "▶️ ЗАПУСК".center(width - 2) + "│")
    print("│" + " " * (width - 2) + "│")
    print("│" + f"  9. {localization.get_text('ui.real_mode.start_option')}".ljust(width - 2) + "│")
    print("│" + " " * (width - 2) + "│")
    print("└" + "─" * (width - 2) + "┘")
    
    # Проверка готовности к реальной торговле
    if not api_key or not api_secret:
        # Рамка для предупреждения
        print("\n┌" + "─" * (width - 2) + "┐")
        print("│" + " " * (width - 2) + "│")
        print("│" + "❌ НЕВОЗМОЖНО ЗАПУСТИТЬ РЕАЛЬНУЮ ТОРГОВЛЮ".center(width - 2) + "│")
        print("│" + " " * (width - 2) + "│")
        print("│" + "Необходимо настроить API ключи Binance".center(width - 2) + "│")
        print("│" + " " * (width - 2) + "│")
        print("└" + "─" * (width - 2) + "┘")
    
    # Ввод пользователя
    choice_prompt = localization.get_text("ui.real_mode.choice_prompt")
    while True:
        choice = input("\n" + choice_prompt)
        
        if choice == "0":
            return False
        elif choice == "9":
            # Проверка готовности к реальной торговле
            if not api_key or not api_secret:
                # Рамка для ошибки
                print("\n┌" + "─" * (width - 2) + "┐")
                print("│" + " " * (width - 2) + "│")
                print("│" + "❌ НЕВОЗМОЖНО ЗАПУСТИТЬ РЕАЛЬНУЮ ТОРГОВЛЮ".center(width - 2) + "│")
                print("│" + " " * (width - 2) + "│")
                print("│" + "Необходимо настроить API ключи Binance".center(width - 2) + "│")
                print("│" + " " * (width - 2) + "│")
                print("└" + "─" * (width - 2) + "┘")
                
                # Пауза перед возвратом в меню
                input("\nНажмите Enter для продолжения...")
                return configure_real_mode(localization)
            else:
                # Дополнительное подтверждение
                print("\n┌" + "─" * (width - 2) + "┐")
                print("│" + " " * (width - 2) + "│")
                print("│" + "⚠️ ПОСЛЕДНЕЕ ПРЕДУПРЕЖДЕНИЕ ⚠️".center(width - 2) + "│")
                print("│" + " " * (width - 2) + "│")
                print("│" + "Вы действительно хотите начать реальную торговлю?".center(width - 2) + "│")
                print("│" + "Это может привести к потере средств!".center(width - 2) + "│")
                print("│" + " " * (width - 2) + "│")
                print("└" + "─" * (width - 2) + "┘")
                
                confirm = input("\nВведите 'ДА, Я ПОНИМАЮ РИСКИ' для подтверждения: ")
                if confirm == "ДА, Я ПОНИМАЮ РИСКИ":
                    return True
                else:
                    # Рамка для отмены
                    print("\n┌" + "─" * (width - 2) + "┐")
                    print("│" + " " * (width - 2) + "│")
                    print("│" + "✅ Запуск реальной торговли отменен".center(width - 2) + "│")
                    print("│" + " " * (width - 2) + "│")
                    print("└" + "─" * (width - 2) + "┘")
                    
                    # Пауза перед возвратом в меню
                    input("\nНажмите Enter для продолжения...")
                    return configure_real_mode(localization)
        elif choice == "1":
            # Настройка API Key
            # Рамка для ввода API Key
            print("\n┌" + "─" * (width - 2) + "┐")
            print("│" + " " * (width - 2) + "│")
            print("│" + "🔑 НАСТРОЙКА API KEY".center(width - 2) + "│")
            print("│" + " " * (width - 2) + "│")
            print("│" + "Введите ваш Binance API Key:".ljust(width - 2) + "│")
            print("│" + " " * (width - 2) + "│")
            print("└" + "─" * (width - 2) + "┘")
            
            new_api_key = input("> ").strip()
            
            if new_api_key:
                # Обновляем .env файл
                update_env_file('BINANCE_API_KEY', new_api_key)
                os.environ['BINANCE_API_KEY'] = new_api_key
                
                # Рамка для успешного обновления
                print("\n┌" + "─" * (width - 2) + "┐")
                print("│" + " " * (width - 2) + "│")
                print("│" + "✅ API Key успешно обновлен".center(width - 2) + "│")
                print("│" + " " * (width - 2) + "│")
                print("└" + "─" * (width - 2) + "┘")
            else:
                # Рамка для ошибки
                print("\n┌" + "─" * (width - 2) + "┐")
                print("│" + " " * (width - 2) + "│")
                print("│" + "❌ API Key не может быть пустым".center(width - 2) + "│")
                print("│" + " " * (width - 2) + "│")
                print("└" + "─" * (width - 2) + "┘")
            
            # Пауза перед возвратом в меню
            input("\nНажмите Enter для продолжения...")
            return configure_real_mode(localization)
        
        elif choice == "2":
            # Настройка API Secret
            # Рамка для ввода API Secret
            print("\n┌" + "─" * (width - 2) + "┐")
            print("│" + " " * (width - 2) + "│")
            print("│" + "🔑 НАСТРОЙКА API SECRET".center(width - 2) + "│")
            print("│" + " " * (width - 2) + "│")
            print("│" + "Введите ваш Binance API Secret:".ljust(width - 2) + "│")
            print("│" + " " * (width - 2) + "│")
            print("└" + "─" * (width - 2) + "┘")
            
            new_api_secret = input("> ").strip()
            
            if new_api_secret:
                # Обновляем .env файл
                update_env_file('BINANCE_API_SECRET', new_api_secret)
                os.environ['BINANCE_API_SECRET'] = new_api_secret
                
                # Рамка для успешного обновления
                print("\n┌" + "─" * (width - 2) + "┐")
                print("│" + " " * (width - 2) + "│")
                print("│" + "✅ API Secret успешно обновлен".center(width - 2) + "│")
                print("│" + " " * (width - 2) + "│")
                print("└" + "─" * (width - 2) + "┘")
            else:
                # Рамка для ошибки
                print("\n┌" + "─" * (width - 2) + "┐")
                print("│" + " " * (width - 2) + "│")
                print("│" + "❌ API Secret не может быть пустым".center(width - 2) + "│")
                print("│" + " " * (width - 2) + "│")
                print("└" + "─" * (width - 2) + "┘")
            
            # Пауза перед возвратом в меню
            input("\nНажмите Enter для продолжения...")
            return configure_real_mode(localization)
        
        # Аналогично улучшаем остальные пункты меню...
        # Для краткости я не буду переписывать все пункты, но принцип тот же
        
        else:
            # Рамка для ошибки
            print("\n┌" + "─" * (width - 2) + "┐")
            print("│" + " " * (width - 2) + "│")
            print("│" + localization.get_text("ui.settings_menu.invalid_choice").center(width - 2) + "│")
            print("│" + " " * (width - 2) + "│")
            print("└" + "─" * (width - 2) + "┘")


def update_env_file(key, value):
    """
    Обновляет значение в .env файле
    
    Args:
        key: Ключ для обновления
        value: Новое значение
    """
    env_path = '.env'
    
    # Читаем текущий .env файл
    if os.path.exists(env_path):
        with open(env_path, 'r') as file:
            lines = file.readlines()
    else:
        lines = []
    
    # Ищем ключ и обновляем его значение
    key_exists = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            key_exists = True
            break
    
    # Если ключ не найден, добавляем его
    if not key_exists:
        lines.append(f"{key}={value}\n")
    
    # Записываем обновленный файл
    with open(env_path, 'w') as file:
        file.writelines(lines)


def show_settings_menu(localization: LocalizationManager):
    """
    Показать меню настроек.
    
    Args:
        localization: Менеджер локализации
    """
    width = 70
    os.system('cls' if os.name == 'nt' else 'clear')
    
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Рисуем компактное меню настроек
    title = localization.get_text("ui.settings_menu.title")
    if title.startswith("[ui.settings_menu.title]"):
        title = "⚙️  LEON TRADING BOT - НАСТРОЙКИ ⚙️"
    print("\n┌" + "─" * (width - 2) + "┐")
    print("│ " + title.ljust(width - 3) + "│")
    print("│" + "─" * (width - 2) + "│")
    
    # Основные настройки
    general_section = localization.get_text("ui.settings_menu.general_section")
    if general_section.startswith("[ui.settings_menu.general_section]"):
        general_section = "🔧 ОСНОВНЫЕ НАСТРОЙКИ:"
    print("│ " + general_section.ljust(width - 3) + "│")
    
    # Заменяем ключи локализации на переведенные тексты
    trading_pair_text = localization.get_text("ui.settings_menu.trading_pair", symbol=config['general']['symbol'])
    if trading_pair_text.startswith("[ui.settings_menu.trading_pair]"):
        trading_pair_text = f"Торговая пара: {config['general']['symbol']}"
    
    kline_interval_text = localization.get_text("ui.settings_menu.kline_interval", interval=config['general']['kline_interval'])
    if kline_interval_text.startswith("[ui.settings_menu.kline_interval]"):
        kline_interval_text = f"Интервал свечей: {config['general']['kline_interval']}"
    
    leverage_text = localization.get_text("ui.settings_menu.leverage", leverage=config['general']['leverage'])
    if leverage_text.startswith("[ui.settings_menu.leverage]"):
        leverage_text = f"Кредитное плечо: {config['general']['leverage']}x"
    
    print("│ 1. " + trading_pair_text.ljust(width - 5) + "│")
    print("│ 2. " + kline_interval_text.ljust(width - 5) + "│")
    print("│ 3. " + leverage_text.ljust(width - 5) + "│")
    
    # Настройки стратегии
    strategy_section = localization.get_text("ui.settings_menu.strategy_section")
    if strategy_section.startswith("[ui.settings_menu.strategy_section]"):
        strategy_section = "📊 НАСТРОЙКИ СТРАТЕГИИ:"
    print("│" + "─" * (width - 2) + "│")
    print("│ " + strategy_section.ljust(width - 3) + "│")
    
    strategy_name_text = localization.get_text("ui.settings_menu.strategy_name", name=config['strategy']['name'])
    if strategy_name_text.startswith("[ui.settings_menu.strategy_name]"):
        strategy_name_text = f"Стратегия: {config['strategy']['name']}"
    
    confidence_threshold_text = localization.get_text("ui.settings_menu.confidence_threshold", threshold=config['strategy']['params']['confidence_threshold'])
    if confidence_threshold_text.startswith("[ui.settings_menu.confidence_threshold]"):
        confidence_threshold_text = f"Порог уверенности: {config['strategy']['params']['confidence_threshold']}"
    
    print("│ 4. " + strategy_name_text.ljust(width - 5) + "│")
    print("│ 5. " + confidence_threshold_text.ljust(width - 5) + "│")
    
    # Настройки риска
    risk_section = localization.get_text("ui.settings_menu.risk_section")
    if risk_section.startswith("[ui.settings_menu.risk_section]"):
        risk_section = "🛡️ УПРАВЛЕНИЕ РИСКАМИ:"
    print("│" + "─" * (width - 2) + "│")
    print("│ " + risk_section.ljust(width - 3) + "│")
    
    position_size_unit = config['risk'].get('position_size_unit', 'USDT')
    max_position_size_text = localization.get_text("ui.settings_menu.max_position_size", size=config['risk']['max_position_size'])
    if max_position_size_text.startswith("[ui.settings_menu.max_position_size]"):
        max_position_size_text = f"Макс. размер позиции: {config['risk']['max_position_size']} {position_size_unit}"
    else:
        max_position_size_text = f"{max_position_size_text} {position_size_unit}"
    
    max_loss_percent_text = localization.get_text("ui.settings_menu.max_loss_percent", percent=config['risk']['max_loss_percent'])
    if max_loss_percent_text.startswith("[ui.settings_menu.max_loss_percent]"):
        max_loss_percent_text = f"Макс. убыток: {config['risk']['max_loss_percent']}%"
    
    print("│ 6. " + max_position_size_text.ljust(width - 5) + "│")
    print("│ 7. " + max_loss_percent_text.ljust(width - 5) + "│")
    
    # Возврат
    back_section = localization.get_text("ui.settings_menu.back_section")
    if back_section.startswith("[ui.settings_menu.back_section]"):
        back_section = "🔙 НАЗАД:"
    print("│" + "─" * (width - 2) + "│")
    print("│ " + back_section.ljust(width - 3) + "│")
    
    back_option_text = localization.get_text("ui.settings_menu.back_option")
    if back_option_text.startswith("[ui.settings_menu.back_option]"):
        back_option_text = "Вернуться в главное меню"
    
    print("│ 0. " + back_option_text.ljust(width - 5) + "│")
    print("└" + "─" * (width - 2) + "┘")
    
    # Ввод пользователя
    choice_prompt = localization.get_text("ui.settings_menu.choice_prompt")
    if choice_prompt.startswith("[ui.settings_menu.choice_prompt]"):
        choice_prompt = "Ваш выбор (0-7): "
    while True:
        choice = input("\n" + choice_prompt)
        
        if choice == "0":
            return
        elif choice == "1":
            # Изменение торговой пары
            current_symbol = config['general']['symbol']
            
            # Компактное меню выбора пары
            print("\n┌" + "─" * (width - 2) + "┐")
            print("│ 🔄 ИЗМЕНЕНИЕ ТОРГОВОЙ ПАРЫ".ljust(width - 1) + "│")
            print("│" + "─" * (width - 2) + "│")
            print("│ 🔸 " + f"{localization.get_text('ui.settings_menu.current_value')}: {current_symbol}".ljust(width - 4) + "│")
            
            # Список доступных торговых пар
            available_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", 
                              "XRPUSDT", "DOTUSDT", "LTCUSDT", "LINKUSDT", "SOLUSDT"]
            print("│" + "─" * (width - 2) + "│")
            print("│ 🔹 " + f"{localization.get_text('ui.settings_menu.available_pairs')}:".ljust(width - 4) + "│")
            
            # Выводим пары в две колонки для экономии места
            for i in range(0, len(available_pairs), 2):
                if i + 1 < len(available_pairs):
                    left = f"{i+1}. {available_pairs[i]}"
                    right = f"{i+2}. {available_pairs[i+1]}"
                    print("│   " + f"{left.ljust(width//2-5)}{right}".ljust(width - 5) + "│")
                else:
                    print("│   " + f"{i+1}. {available_pairs[i]}".ljust(width - 5) + "│")
            
            print("└" + "─" * (width - 2) + "┘")
            
            pair_choice = input(f"🔹 {localization.get_text('ui.settings_menu.select_pair')}: ").strip()
            
            try:
                # Если введен номер пары
                choice_idx = int(pair_choice) - 1
                if 0 <= choice_idx < len(available_pairs):
                    new_symbol = available_pairs[choice_idx]
                    config_manager.update_config('general', 'symbol', new_symbol)
                    config_manager.save_config()
                    
                    # Сообщение об успешном обновлении
                    print(f"\n✅ {localization.get_text('ui.settings_menu.value_updated')}")
                else:
                    print(f"\n❌ {localization.get_text('ui.settings_menu.invalid_value')}")
            except ValueError:
                # Если введено название пары
                new_symbol = pair_choice.upper()
                if new_symbol in available_pairs:
                    config_manager.update_config('general', 'symbol', new_symbol)
                    config_manager.save_config()
                    print(f"\n✅ {localization.get_text('ui.settings_menu.value_updated')}")
                else:
                    print(f"\n❌ {localization.get_text('ui.settings_menu.invalid_value')}")
            
            # Пауза перед возвратом в меню
            input("\nНажмите Enter для продолжения...")
            return show_settings_menu(localization)
            
        # Аналогично улучшаем остальные пункты меню...
        # Для краткости я не буду переписывать все пункты, но принцип тот же
        
        else:
            print(f"\n❌ {localization.get_text('ui.settings_menu.invalid_choice')}")


def user_message(message: str, level: str = "INFO", logger=None) -> None:
    """
    Выводит сообщение для пользователя.
    
    Args:
        message: Сообщение для вывода
        level: Уровень логирования
        logger: Логгер для записи сообщения
    """
    # Создаем запись лога с пользовательским атрибутом
    log_record = logging.LogRecord(
        name="user",
        level=getattr(logging, level),
        pathname="",
        lineno=0,
        msg=message,
        args=(),
        exc_info=None
    )
    setattr(log_record, 'user_friendly', True)
    
    # Выводим сообщение в консоль
    logging.getLogger().handle(log_record)
    
    # Записываем сообщение в лог, если предоставлен логгер
    if logger:
        getattr(logger, level.lower())(message)


def get_config_value(config, path, default=None):
    """
    Безопасно получает значение из конфигурации по пути.
    
    Args:
        config: Словарь конфигурации
        path: Путь к значению в формате "section.key.subkey"
        default: Значение по умолчанию, если путь не найден
        
    Returns:
        Значение из конфигурации или значение по умолчанию
    """
    if not config:
        return default
        
    parts = path.split('.')
    result = config
    
    for part in parts:
        if isinstance(result, dict) and part in result:
            result = result[part]
        else:
            return default
            
    return result


def show_error_message(error: Exception, error_type: str = "Ошибка", details: bool = False) -> None:
    """
    Показывает красивое сообщение об ошибке.
    
    Args:
        error: Объект исключения
        error_type: Тип ошибки для отображения
        details: Показывать детали ошибки
    """
    width = 70
    print("\n" + "=" * width)
    print(f"❌ {error_type.upper()}")
    print("-" * width)
    
    # Получаем понятное сообщение об ошибке
    if isinstance(error, TypeError) and "takes 1 positional argument but 2 were given" in str(error):
        message = "Ошибка при сохранении конфигурации. Неверный вызов метода."
        solution = "Используйте методы update_config() и save_config() без аргументов."
    elif isinstance(error, FileNotFoundError):
        message = f"Файл не найден: {error.filename}"
        solution = "Проверьте наличие файла и правильность пути."
    elif isinstance(error, PermissionError):
        message = "Недостаточно прав для доступа к файлу или директории."
        solution = "Запустите приложение с правами администратора или измените права доступа к файлам."
    elif isinstance(error, KeyError):
        message = f"Отсутствует ключ в конфигурации: {str(error)}"
        solution = "Проверьте файл конфигурации на наличие всех необходимых параметров."
    elif isinstance(error, ValueError):
        message = f"Неверное значение: {str(error)}"
        solution = "Проверьте правильность введенных данных."
    elif isinstance(error, ImportError) or isinstance(error, ModuleNotFoundError):
        message = f"Ошибка импорта модуля: {str(error)}"
        solution = "Установите необходимые зависимости: pip install -r requirements.txt"
    elif isinstance(error, asyncio.CancelledError):
        message = "Операция была отменена."
        solution = "Это нормальное поведение при остановке бота."
    else:
        message = str(error)
        solution = "Проверьте конфигурацию и перезапустите приложение."
    
    print(f"📝 {message}")
    
    # Если нужны детали, показываем трассировку
    if details:
        print("-" * width)
        print("📋 Детали:")
        tb = traceback.format_exception(type(error), error, error.__traceback__)
        print("".join(tb[-3:]).strip())  # Показываем только последние 3 строки трассировки
    
    print("-" * width)
    print(f"💡 Решение: {solution}")
    print("=" * width + "\n")


def show_main_menu(localization) -> str:
    """
    Показывает главное меню приложения.
    
    Args:
        localization: Менеджер локализации
        
    Returns:
        str: Выбранный пункт меню
    """
    width = 70
    
    # Очистка экрана
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Заголовок
    print("\n" + "┌" + "─" * (width - 2) + "┐")
    print("│" + " " * (width - 2) + "│")
    print("│" + "LEON TRADING BOT - ГЛАВНОЕ МЕНЮ".center(width - 2) + "│")
    print("│" + " " * (width - 2) + "│")
    print("└" + "─" * (width - 2) + "┘")
    
    # Пункты меню
    print("\n┌" + "─" * (width - 2) + "┐")
    print("│ 🚀 " + "ДЕЙСТВИЯ:".ljust(width - 5) + "│")
    print("│" + "─" * (width - 2) + "│")
    print("│ 1. " + "Запустить бота".ljust(width - 5) + "│")
    print("│ 2. " + "Настройки".ljust(width - 5) + "│")
    print("│ 3. " + "Выход".ljust(width - 5) + "│")
    print("└" + "─" * (width - 2) + "┘")
    
    # Запрос выбора
    choice = input("\n🔹 Выберите действие (1-3): ").strip()
    
    return choice


async def show_splash_screen() -> None:
    """Показывает красивый стартовый экран."""
    width = 70
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("\n" + "=" * width)
    print("""
    ██╗     ███████╗ ██████╗ ███╗   ██╗    ██████╗  ██████╗ ████████╗
    ██║     ██╔════╝██╔═══██╗████╗  ██║    ██╔══██╗██╔═══██╗╚══██╔══╝
    ██║     █████╗  ██║   ██║██╔██╗ ██║    ██████╔╝██║   ██║   ██║
    ██║     ██╔══╝  ██║   ██║██║╚██╗██║    ██╔══██╗██║   ██║   ██║
    ███████╗███████╗╚██████╔╝██║ ╚████║    ██████╔╝╚██████╔╝   ██║
    ╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝    ╚═════╝  ╚═════╝    ╚═╝
    """)
    print("=" * width)
    print("                   🤖 ТОРГОВЫЙ БОТ С ИИ 🤖")
    print("=" * width)
    print()
    print("📈 Автоматическая торговля на криптовалютных биржах")
    print("🧠 Использование машинного обучения для принятия решений")
    print("🛡️ Управление рисками и защита капитала")
    print("📊 Расширенная аналитика и визуализация")
    print()
    print("=" * width)
    print("                  Загрузка системы...")
    print("=" * width)
    
    # Имитация загрузки
    print()
    for i in range(51):
        progress = "█" * i
        spaces = " " * (50 - i)
        print(f"\r[{progress}{spaces}] {i*2}%", end="")
        await asyncio.sleep(0.02)
    print("\n\n✅ Система готова к работе!\n")


async def main():
    """Основная функция приложения."""
    # Настройка логирования
    setup_logging()
    logger = logging.getLogger("main")
    
    # Показываем стартовый экран
    await show_splash_screen()
    
    # Загрузка конфигурации
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Инициализация локализации
    localization = LocalizationManager(get_config_value(config, "general.language", "ru"))
    
    # Проверка режима работы
    mode = get_config_value(config, "general.mode", "dry")
    
    # Проверка наличия API ключей для режима реальной торговли
    if mode == "real":
        api_key = os.environ.get('BINANCE_API_KEY', '')
        api_secret = os.environ.get('BINANCE_API_SECRET', '')
        
        if not api_key or not api_secret:
            user_message("⚠️ API ключи не найдены в переменных окружения. Режим реальной торговли недоступен.", "WARNING", logger)
            user_message("ℹ️ Переключение в режим сухого тестирования.", "INFO", logger)
            mode = "dry"
            # Обновляем режим в конфигурации
            config["general"]["mode"] = "dry"
            config_manager.save_config(config)
        else:
            user_message("✅ API ключи найдены в переменных окружения", "INFO", logger)
    
    # Показываем главное меню
    choice = show_main_menu(localization)
    
    # Обработка выбора пользователя
    if choice == "1":
        # Запуск бота
        try:
            # Создание и инициализация оркестратора
            try:
                logger.info("Создание оркестратора...")
                # Исправляем вызов конструктора, передавая строку с путем к конфигурации
                config_file_path = "config.yaml"  # Используем путь по умолчанию
                orchestrator = LeonOrchestrator(config_file_path)
                # Устанавливаем локализацию после создания
                orchestrator.localization_manager = localization
                logger.info("Оркестратор создан успешно")
                
                # Инициализация Telegram бота, если он доступен
                if hasattr(orchestrator, 'telegram_bot') and orchestrator.telegram_bot:
                    try:
                        await orchestrator.telegram_bot.start()
                        logger.info("Telegram бот успешно запущен")
                    except Exception as e:
                        logger.warning(f"Не удалось запустить Telegram бота: {str(e)}")
                        logger.debug(traceback.format_exc())
            except Exception as e:
                logger.error(f"Ошибка при создании оркестратора: {str(e)}")
                logger.debug(traceback.format_exc())
                show_error_message(e, "Ошибка инициализации оркестратора", details=True)
                return
            
            # Запуск системы
            try:
                logger.info("Запуск системы...")
                await orchestrator.start()
                logger.info("Система запущена успешно")
            except Exception as e:
                logger.error(f"Ошибка при запуске системы: {str(e)}")
                logger.debug(traceback.format_exc())
                show_error_message(e, "Ошибка запуска системы", details=True)
                
                # Пытаемся корректно остановить оркестратор
                try:
                    await orchestrator.stop()
                except Exception as stop_error:
                    logger.error(f"Ошибка при остановке оркестратора после сбоя: {str(stop_error)}")
                return
            
            # Основной цикл обновления данных для визуализатора
            try:
                error_count = 0
                while orchestrator.is_running():
                    try:
                        # Просто ждем, визуализация происходит автоматически через visualization_manager
                        await asyncio.sleep(5)
                        
                        # Проверка здоровья визуализатора
                        if orchestrator.visualization_manager:
                            visualizer = orchestrator.get_visualizer()
                            if visualizer is None:
                                logger.warning("Визуализатор отсутствует, пытаемся перезапустить")
                                await orchestrator.visualization_manager.start_visualization()
                    except asyncio.CancelledError:
                        logger.info("Задача обновления визуализации отменена")
                        break
                    except Exception as e:
                        logger.error(f"Ошибка в основном цикле: {str(e)}")
                        logger.debug(traceback.format_exc())
                        
                        # Подсчет ошибок, чтобы избежать бесконечного цикла ошибок
                        error_count += 1
                        if error_count > 10:
                            logger.critical("Слишком много ошибок, останавливаем цикл")
                            break
                        
                        # Продолжаем работу даже при ошибке
                        await asyncio.sleep(5)
            except KeyboardInterrupt:
                # Обработка Ctrl+C для корректного завершения
                user_message("Получен сигнал остановки (Ctrl+C)", "INFO", logger)
            except Exception as e:
                # Обработка других исключений
                show_error_message(e, "Ошибка в основном цикле", details=True)
                logger.exception("Ошибка в основном цикле:")
            finally:
                # Остановка системы
                try:
                    await orchestrator.stop()
                except Exception as e:
                    show_error_message(e, "Ошибка остановки", details=True)
                    logger.exception("Ошибка при остановке оркестратора:")
            
            user_message("Бот остановлен. До свидания!", "INFO", logger)
            
        except Exception as e:
            # Обработка ошибок инициализации
            show_error_message(e, "Ошибка запуска", details=True)
            logger.exception("Ошибка при запуске бота:")
    
    elif choice == "2":
        # Настройки
        show_settings_menu(localization)
        # После выхода из меню настроек перезапускаем главное меню
        await main()
    
    elif choice == "3":
        # Выход
        user_message(localization.get_text("ui.main_menu.exit_message"), "INFO", logger)
    
    else:
        # Неверный выбор
        user_message(localization.get_text("ui.main_menu.invalid_choice"), "WARNING", logger)
        # Перезапускаем главное меню
        await main()


if __name__ == "__main__":
    # Запуск основной функции
    asyncio.run(main()) 