import os
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv

class ConfigManager:
    """
    Класс для управления конфигурацией системы.
    Загружает настройки из файла config.yaml и переменных окружения.
    """
    
    def __init__(self, config_path="config.yaml", env_path=".env"):
        """
        Инициализация менеджера конфигурации.
        
        Args:
            config_path: Путь к файлу конфигурации
            env_path: Путь к файлу с переменными окружения
        """
        self.logger = logging.getLogger("ConfigManager")
        self.config_path = config_path
        self.env_path = env_path
        self.config = {}
        
        # Загрузка переменных окружения
        self._load_env_vars()
        
        # Загрузка конфигурации
        self._load_config()
        
    def _load_env_vars(self):
        """Загрузка переменных окружения из .env файла"""
        env_path = Path(self.env_path)
        if env_path.exists():
            self.logger.info(f"Загрузка переменных окружения из {self.env_path}")
            load_dotenv(dotenv_path=self.env_path)
        else:
            self.logger.warning(f"Файл {self.env_path} не найден. Используются только системные переменные окружения.")
    
    def _load_config(self):
        """Загрузка конфигурации из YAML файла с подстановкой переменных окружения"""
        try:
            with open(self.config_path, 'r') as file:
                # Загрузка YAML с подстановкой переменных окружения
                config_str = file.read()
                
                # Замена переменных окружения в строке конфигурации
                for key, value in os.environ.items():
                    placeholder = "${" + key + "}"
                    config_str = config_str.replace(placeholder, value)
                
                # Парсинг YAML после замены переменных
                self.config = yaml.safe_load(config_str)
                
            self.logger.info(f"Конфигурация успешно загружена из {self.config_path}")
            self._validate_config()
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке конфигурации: {e}")
            raise
    
    def _validate_config(self):
        """Валидация загруженной конфигурации"""
        # Проверка обязательных секций
        required_sections = ["general", "risk", "strategy", "binance", "telegram"]
        for section in required_sections:
            if section not in self.config:
                self.logger.error(f"Отсутствует обязательная секция '{section}' в конфигурации")
                raise ValueError(f"Отсутствует обязательная секция '{section}' в конфигурации")
        
        # Проверка режима работы
        valid_modes = ["dry", "real", "backtest"]
        mode = self.config["general"]["mode"]
        if mode not in valid_modes:
            self.logger.error(f"Некорректный режим работы: {mode}. Допустимые значения: {valid_modes}")
            raise ValueError(f"Некорректный режим работы: {mode}. Допустимые значения: {valid_modes}")
        
        # Проверка API ключей для реального режима
        if mode == "real":
            if not self.config["binance"]["api_key"] or not self.config["binance"]["api_secret"]:
                self.logger.error("Для режима реальной торговли необходимо указать API ключи Binance")
                raise ValueError("Для режима реальной торговли необходимо указать API ключи Binance")
        
        self.logger.info("Валидация конфигурации успешно завершена")
    
    def get_config(self, section=None, key=None):
        """
        Получение значения из конфигурации.
        
        Args:
            section: Секция конфигурации
            key: Ключ в секции
            
        Returns:
            Значение из конфигурации или вся конфигурация, если section и key не указаны
        """
        if section is None:
            return self.config
        
        if section not in self.config:
            self.logger.error(f"Секция '{section}' не найдена в конфигурации")
            return None
        
        if key is None:
            return self.config[section]
        
        if key not in self.config[section]:
            self.logger.error(f"Ключ '{key}' не найден в секции '{section}'")
            return None
        
        return self.config[section][key]
    
    def update_config(self, section, key, value):
        """
        Обновление значения в конфигурации.
        
        Args:
            section: Секция конфигурации
            key: Ключ в секции
            value: Новое значение
        """
        if section not in self.config:
            self.logger.error(f"Секция '{section}' не найдена в конфигурации")
            return False
        
        self.config[section][key] = value
        self.logger.info(f"Обновлено значение '{key}' в секции '{section}'")
        return True
    
    def save_config(self):
        """Сохранение текущей конфигурации в файл"""
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            self.logger.info(f"Конфигурация сохранена в {self.config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении конфигурации: {e}")
            return False


# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Создание экземпляра ConfigManager
    config_manager = ConfigManager()
    
    # Получение всей конфигурации
    config = config_manager.get_config()
    print("Полная конфигурация:", config)
    
    # Получение значения из конфигурации
    mode = config_manager.get_config("general", "mode")
    print(f"Режим работы: {mode}")
    
    # Получение API ключей
    api_key = config_manager.get_config("binance", "api_key")
    print(f"API ключ Binance: {api_key[:5]}..." if api_key else "API ключ не найден") 