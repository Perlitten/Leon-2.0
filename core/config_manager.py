import os

import yaml

import logging

from typing import Dict, Any, Optional

from dotenv import load_dotenv



class ConfigManager:

    """

    Класс для управления конфигурацией системы.

    Загружает настройки из config.yaml и переменных окружения.

    """

    

    def __init__(self, config_path: str = "config.yaml"):

        """

        Инициализация менеджера конфигурации.

        

        Args:

            config_path: Путь к файлу конфигурации

        """

        self.logger = logging.getLogger(__name__)

        self.config_path = config_path

        self.config = {}

        

        # Загрузка переменных окружения

        load_dotenv()

        

        # Загрузка конфигурации

        self.load_config()

        

        # Используем debug уровень для технических сообщений, которые не будут показаны пользователю

        self.logger.debug("Менеджер конфигурации инициализирован")

    

    def load_config(self) -> None:

        """Загрузка конфигурации из файла."""

        try:

            with open(self.config_path, 'r', encoding='utf-8') as file:

                self.config = yaml.safe_load(file)

            self.logger.debug(f"Конфигурация загружена из {self.config_path}")

            

            # Переопределение значений из переменных окружения

            self._override_from_env()

        except Exception as e:

            self.logger.error(f"Ошибка при загрузке конфигурации: {e}")

            raise

    

    def _override_from_env(self) -> None:

        """Переопределение значений конфигурации из переменных окружения."""

        # Binance API ключи

        if os.getenv("BINANCE_API_KEY"):

            self.config["binance"]["api_key"] = os.getenv("BINANCE_API_KEY")

        

        if os.getenv("BINANCE_API_SECRET"):

            self.config["binance"]["api_secret"] = os.getenv("BINANCE_API_SECRET")

        

        # Telegram настройки

        if os.getenv("TELEGRAM_BOT_TOKEN"):

            self.config["telegram"]["bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN")

        

        if os.getenv("TELEGRAM_CHAT_ID"):

            self.config["telegram"]["chat_id"] = os.getenv("TELEGRAM_CHAT_ID")

        

        # Торговая пара

        if os.getenv("TRADING_SYMBOL"):

            self.config["general"]["symbol"] = os.getenv("TRADING_SYMBOL")

        

        # Интервал свечей

        if os.getenv("KLINE_INTERVAL"):

            self.config["general"]["kline_interval"] = os.getenv("KLINE_INTERVAL")

    

    def get_config(self) -> Dict[str, Any]:

        """

        Получение всей конфигурации.

        

        Returns:

            Dict[str, Any]: Словарь с конфигурацией

        """

        return self.config

    

    def get_section(self, section: str) -> Optional[Dict[str, Any]]:

        """

        Получение секции конфигурации.

        

        Args:

            section: Название секции

            

        Returns:

            Optional[Dict[str, Any]]: Словарь с настройками секции или None

        """

        return self.config.get(section)

    

    def get_value(self, section: str, key: str, default: Any = None) -> Any:

        """

        Получение значения из конфигурации.

        

        Args:

            section: Название секции

            key: Ключ настройки

            default: Значение по умолчанию

            

        Returns:

            Any: Значение настройки или default

        """

        section_data = self.get_section(section)

        if section_data:

            return section_data.get(key, default)

        return default

    

    def update_config(self, section: str, key: str, value: Any) -> None:

        """

        Обновление значения в конфигурации.

        

        Args:

            section: Название секции

            key: Ключ настройки

            value: Новое значение

        """

        if section not in self.config:

            self.config[section] = {}

        

        self.config[section][key] = value

        self.logger.debug(f"Обновлена настройка {section}.{key} = {value}")

    

    def save_config(self) -> None:

        """Сохранение конфигурации в файл."""

        try:

            with open(self.config_path, 'w', encoding='utf-8') as file:

                yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)

            self.logger.debug(f"Конфигурация сохранена в {self.config_path}")

        except Exception as e:

            self.logger.error(f"Ошибка при сохранении конфигурации: {e}")

            raise

