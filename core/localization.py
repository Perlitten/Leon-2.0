"""
Модуль локализации для управления текстами и сообщениями в системе Leon Trading Bot.

Этот модуль предоставляет функциональность для загрузки, хранения и получения 
локализованных текстов и сообщений из YAML-файлов. Он поддерживает несколько 
языков и позволяет легко переключаться между ними.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from core.exceptions import ConfigLoadError

logger = logging.getLogger(__name__)

class LocalizationManager:
    """
    Класс для управления локализованными текстами и сообщениями.
    
    Этот класс загружает тексты из YAML-файлов и предоставляет методы для 
    получения локализованных сообщений по ключам.
    """
    
    def __init__(self, locales_dir: str = "locales", default_language: str = "ru", dry_mode: bool = False):
        """
        Инициализирует менеджер локализации.
        
        Args:
            locales_dir (str): Директория с файлами локализации
            default_language (str): Язык по умолчанию
            dry_mode (bool): Режим без сохранения изменений на диск (для тестирования)
        """
        self.locales_dir = locales_dir
        self.default_language = default_language
        self.current_language = default_language
        self.texts: Dict[str, Dict[str, Any]] = {}
        self.dry_mode = dry_mode
        
        # Создаем директорию для локализаций, если она не существует и не в dry_mode
        if not dry_mode:
            os.makedirs(locales_dir, exist_ok=True)
        
        # Загружаем доступные языки
        self.available_languages = self._get_available_languages()
        
        # Загружаем тексты для языка по умолчанию
        self.load_language(default_language)
    
    def _get_available_languages(self) -> List[str]:
        """
        Получает список доступных языков из директории локализаций.
        
        Returns:
            List[str]: Список доступных языков
        """
        languages = []
        
        if self.dry_mode:
            # В режиме dry_mode возвращаем только язык по умолчанию
            return [self.default_language]
        
        try:
            for file in os.listdir(self.locales_dir):
                if file.endswith('.yaml') or file.endswith('.yml'):
                    lang = file.split('.')[0]
                    languages.append(lang)
        except FileNotFoundError:
            logger.warning(f"Директория локализаций {self.locales_dir} не найдена")
        
        return languages
    
    def load_language(self, language: str) -> bool:
        """
        Загружает тексты для указанного языка.
        
        Args:
            language (str): Код языка для загрузки
            
        Returns:
            bool: True, если загрузка успешна, иначе False
            
        Raises:
            ConfigLoadError: Если не удалось загрузить файл локализации
        """
        if self.dry_mode:
            # В режиме dry_mode создаем пустой словарь для языка
            if language not in self.texts:
                self.texts[language] = {}
            return True
        
        file_path = os.path.join(self.locales_dir, f"{language}.yaml")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.texts[language] = yaml.safe_load(file)
            
            logger.info(f"Загружен язык: {language}")
            return True
        except FileNotFoundError:
            logger.error(f"Файл локализации не найден: {file_path}")
            if language != self.default_language:
                logger.info(f"Используется язык по умолчанию: {self.default_language}")
                return False
            else:
                # Создаем базовый файл локализации для языка по умолчанию
                self._create_default_locale_file(file_path)
                return self.load_language(language)
        except yaml.YAMLError as e:
            logger.error(f"Ошибка при разборе файла локализации {file_path}: {e}")
            raise ConfigLoadError(f"Ошибка при разборе файла локализации", file_path=file_path)
    
    def _create_default_locale_file(self, file_path: str) -> None:
        """
        Создает файл локализации по умолчанию, если он не существует.
        
        Args:
            file_path (str): Путь к файлу локализации
        """
        default_texts = {
            "common": {
                "yes": "Да",
                "no": "Нет",
                "ok": "OK",
                "cancel": "Отмена",
                "error": "Ошибка",
                "warning": "Предупреждение",
                "info": "Информация",
                "success": "Успешно",
                "loading": "Загрузка...",
                "processing": "Обработка...",
                "please_wait": "Пожалуйста, подождите..."
            },
            "errors": {
                "general": "Произошла ошибка: {message}",
                "api_error": "Ошибка API: {message}",
                "connection_error": "Ошибка соединения: {message}",
                "authentication_error": "Ошибка аутентификации: {message}",
                "data_error": "Ошибка данных: {message}",
                "config_error": "Ошибка конфигурации: {message}"
            },
            "trading": {
                "order_created": "Создан ордер: {order_id}",
                "order_cancelled": "Ордер отменен: {order_id}",
                "order_filled": "Ордер исполнен: {order_id}",
                "position_opened": "Открыта позиция: {symbol} {side} {quantity}",
                "position_closed": "Закрыта позиция: {symbol} {side} {quantity} (P&L: {pnl})",
                "insufficient_funds": "Недостаточно средств: требуется {required}, доступно {available}"
            },
            "notifications": {
                "telegram": {
                    "welcome": "Добро пожаловать в Leon Trading Bot!",
                    "help": "Доступные команды:\n/status - Статус бота\n/balance - Баланс аккаунта\n/positions - Открытые позиции\n/orders - Активные ордера\n/start - Запустить торговлю\n/stop - Остановить торговлю\n/help - Показать эту справку",
                    "status": "Статус бота: {status}\nРежим: {mode}\nАктивные пары: {symbols}",
                    "balance": "Баланс аккаунта:\n{balances}",
                    "positions": "Открытые позиции:\n{positions}",
                    "orders": "Активные ордера:\n{orders}",
                    "bot_started": "Бот запущен в режиме {mode}",
                    "bot_stopped": "Бот остановлен",
                    "error_message": "Ошибка: {message}"
                }
            },
            "ui": {
                "dashboard": {
                    "title": "Leon Trading Bot - Панель управления",
                    "status": "Статус",
                    "balance": "Баланс",
                    "positions": "Позиции",
                    "orders": "Ордера",
                    "history": "История",
                    "settings": "Настройки"
                },
                "settings": {
                    "title": "Настройки",
                    "general": "Общие",
                    "trading": "Торговля",
                    "notifications": "Уведомления",
                    "api": "API",
                    "save": "Сохранить",
                    "reset": "Сбросить"
                }
            }
        }
        
        if self.dry_mode:
            # В режиме dry_mode не создаем файл, но сохраняем тексты в памяти
            self.texts[self.default_language] = default_texts
            logger.info(f"Режим dry_mode: тексты по умолчанию загружены в память")
            return
        
        try:
            # Убедимся, что директория существует
            # Используем self.locales_dir вместо os.path.dirname(file_path)
            os.makedirs(self.locales_dir, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                yaml.dump(default_texts, file, allow_unicode=True, sort_keys=False)
            
            logger.info(f"Создан файл локализации по умолчанию: {file_path}")
        except Exception as e:
            logger.error(f"Не удалось создать файл локализации по умолчанию: {e}")
    
    def set_language(self, language: str) -> bool:
        """
        Устанавливает текущий язык.
        
        Args:
            language (str): Код языка для установки
            
        Returns:
            bool: True, если язык успешно установлен, иначе False
        """
        if language not in self.texts:
            if not self.load_language(language):
                return False
        
        self.current_language = language
        logger.info(f"Установлен язык: {language}")
        return True
    
    def get_text(self, key: str, default: str = None, **kwargs) -> str:
        """
        Получает локализованный текст по ключу.
        
        Args:
            key (str): Ключ текста в формате "section.subsection.key"
            default (str, optional): Текст по умолчанию, если ключ не найден
            **kwargs: Параметры для форматирования текста
            
        Returns:
            str: Локализованный текст
        """
        # Получаем текст для текущего языка
        text = self._get_text_by_key(self.current_language, key)
        
        # Если текст не найден, пробуем получить его для языка по умолчанию
        if text is None and self.current_language != self.default_language:
            text = self._get_text_by_key(self.default_language, key)
        
        # Если текст все еще не найден, используем значение по умолчанию
        if text is None:
            if default is not None:
                return default
            return f"[{key}]"  # Возвращаем ключ в квадратных скобках
        
        # Форматируем текст, если переданы параметры
        if kwargs and isinstance(text, str):
            try:
                return text.format(**kwargs)
            except KeyError as e:
                logger.warning(f"Отсутствует ключ {e} для форматирования текста '{key}'")
                return text
        
        return text
    
    def _get_text_by_key(self, language: str, key: str) -> Optional[str]:
        """
        Получает текст по ключу для указанного языка.
        
        Args:
            language (str): Код языка
            key (str): Ключ для поиска текста (формат: "section.subsection.key")
            
        Returns:
            Optional[str]: Найденный текст или None, если текст не найден
        """
        if language not in self.texts:
            return None
        
        # Разбиваем ключ на части
        parts = key.split('.')
        
        # Ищем текст в словаре
        current = self.texts[language]
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        # Проверяем, что найденное значение является строкой
        if isinstance(current, str):
            return current
        
        return None
    
    def save_texts(self, language: str) -> bool:
        """
        Сохраняет тексты для указанного языка в файл.
        
        Args:
            language (str): Код языка для сохранения
            
        Returns:
            bool: True, если сохранение успешно, иначе False
        """
        if language not in self.texts:
            logger.error(f"Язык {language} не загружен")
            return False
        
        if self.dry_mode:
            logger.info(f"Режим dry_mode: тексты для языка {language} не сохраняются на диск")
            return True
        
        file_path = os.path.join(self.locales_dir, f"{language}.yaml")
        
        try:
            # Убедимся, что директория существует
            os.makedirs(self.locales_dir, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.texts[language], file, allow_unicode=True, sort_keys=False)
            
            logger.info(f"Сохранен язык: {language}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении файла локализации {file_path}: {e}")
            return False
    
    def add_text(self, language: str, key: str, text: str) -> bool:
        """
        Добавляет или обновляет текст для указанного языка.
        
        Args:
            language (str): Код языка
            key (str): Ключ для текста (формат: "section.subsection.key")
            text (str): Текст для добавления
            
        Returns:
            bool: True, если текст успешно добавлен, иначе False
        """
        if language not in self.texts:
            if not self.load_language(language):
                # Если язык не загружен, создаем новый словарь
                self.texts[language] = {}
        
        # Разбиваем ключ на части
        parts = key.split('.')
        
        # Добавляем текст в словарь
        current = self.texts[language]
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # Если текущая часть не является словарем, заменяем ее словарем
                current[part] = {}
            
            current = current[part]
        
        # Устанавливаем значение для последней части ключа
        current[parts[-1]] = text
        
        logger.info(f"Добавлен текст для языка {language}: {key}")
        return True
    
    def get_all_keys(self, prefix: str = "") -> List[str]:
        """
        Получает список всех доступных ключей.
        
        Args:
            prefix (str): Префикс для фильтрации ключей
            
        Returns:
            List[str]: Список ключей
        """
        keys = []
        
        # Используем язык по умолчанию для получения ключей
        if self.default_language in self.texts:
            self._collect_keys(self.texts[self.default_language], keys, prefix)
        
        return keys
    
    def _collect_keys(self, data: Dict[str, Any], keys: List[str], prefix: str = "", current_path: str = "") -> None:
        """
        Рекурсивно собирает ключи из словаря.
        
        Args:
            data (Dict[str, Any]): Словарь для сбора ключей
            keys (List[str]): Список для сохранения ключей
            prefix (str): Префикс для фильтрации ключей
            current_path (str): Текущий путь в иерархии ключей
        """
        for key, value in data.items():
            path = f"{current_path}.{key}" if current_path else key
            
            if isinstance(value, dict):
                self._collect_keys(value, keys, prefix, path)
            elif isinstance(value, str):
                if not prefix or path.startswith(prefix):
                    keys.append(path)
    
    def update_text(self, key: str, value: str, language: str = None) -> bool:
        """
        Обновляет локализованный текст по ключу.
        
        Args:
            key (str): Ключ текста в формате "section.subsection.key"
            value (str): Новое значение текста
            language (str, optional): Язык для обновления (по умолчанию текущий)
            
        Returns:
            bool: True, если обновление успешно, иначе False
        """
        if language is None:
            language = self.current_language
        
        if language not in self.texts:
            if not self.load_language(language):
                return False
        
        # Разбиваем ключ на части
        parts = key.split('.')
        
        # Получаем ссылку на словарь текстов для указанного языка
        current_dict = self.texts[language]
        
        # Проходим по всем частям ключа, кроме последней
        for i, part in enumerate(parts[:-1]):
            # Если текущая часть не существует в словаре, создаем новый словарь
            if part not in current_dict:
                current_dict[part] = {}
            
            # Если значение не является словарем, заменяем его на словарь
            if not isinstance(current_dict[part], dict):
                current_dict[part] = {}
            
            # Переходим к следующему уровню
            current_dict = current_dict[part]
        
        # Устанавливаем значение для последней части ключа
        current_dict[parts[-1]] = value
        
        # Сохраняем изменения в файл, если не в режиме dry_mode
        if not self.dry_mode:
            self._save_language(language)
        
        return True
    
    def _save_language(self, language: str) -> bool:
        """
        Сохраняет тексты для указанного языка в файл.
        
        Args:
            language (str): Код языка для сохранения
            
        Returns:
            bool: True, если сохранение успешно, иначе False
        """
        if self.dry_mode:
            return True
        
        file_path = os.path.join(self.locales_dir, f"{language}.yaml")
        
        try:
            # Убедимся, что директория существует
            os.makedirs(self.locales_dir, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.texts[language], file, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"Сохранен язык: {language}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении языка {language}: {e}")
            return False


# Создаем глобальный экземпляр менеджера локализации
_localization_manager = None
_dry_mode = False

def set_dry_mode(dry_mode: bool) -> None:
    """
    Устанавливает режим dry_mode для модуля локализации.
    
    Args:
        dry_mode (bool): True для включения режима dry_mode, False для отключения
    """
    global _dry_mode, _localization_manager
    _dry_mode = dry_mode
    
    # Сбрасываем экземпляр менеджера локализации, чтобы он был пересоздан с новым значением dry_mode
    _localization_manager = None
    
    logger.info(f"Режим dry_mode {'включен' if dry_mode else 'отключен'}")

def get_localization_manager() -> LocalizationManager:
    """
    Получает глобальный экземпляр менеджера локализации.
    
    Returns:
        LocalizationManager: Экземпляр менеджера локализации
    """
    global _localization_manager, _dry_mode
    if _localization_manager is None:
        _localization_manager = LocalizationManager(dry_mode=_dry_mode)
    
    return _localization_manager

def get_text(key: str, default: Optional[str] = None, **kwargs) -> str:
    """
    Получает локализованный текст по ключу.
    
    Args:
        key (str): Ключ для поиска текста (формат: "section.subsection.key")
        default (Optional[str]): Текст по умолчанию, если ключ не найден
        **kwargs: Параметры для форматирования текста
        
    Returns:
        str: Локализованный текст
    """
    return get_localization_manager().get_text(key, default, **kwargs)

def set_language(language: str) -> bool:
    """
    Устанавливает текущий язык.
    
    Args:
        language (str): Код языка для установки
        
    Returns:
        bool: True, если язык успешно установлен, иначе False
    """
    return get_localization_manager().set_language(language) 