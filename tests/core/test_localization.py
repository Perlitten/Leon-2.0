"""
Тесты для модуля локализации.
"""

import os
import pytest
from unittest.mock import patch, mock_open, MagicMock
import yaml

from core.localization import (
    LocalizationManager, 
    get_localization_manager, 
    get_text, 
    set_language,
    set_dry_mode
)
from core.exceptions import ConfigLoadError


class TestLocalizationManager:
    """Тесты для класса LocalizationManager."""

    def test_init(self):
        """Тест инициализации менеджера локализации."""
        with patch('os.makedirs') as mock_makedirs:
            manager = LocalizationManager(locales_dir="test_locales", default_language="en")
            
            assert manager.locales_dir == "test_locales"
            assert manager.default_language == "en"
            assert manager.current_language == "en"
            assert isinstance(manager.texts, dict)
            mock_makedirs.assert_called_once_with("test_locales", exist_ok=True)
    
    def test_init_dry_mode(self):
        """Тест инициализации менеджера локализации в режиме dry_mode."""
        with patch('os.makedirs') as mock_makedirs:
            manager = LocalizationManager(locales_dir="test_locales", default_language="en", dry_mode=True)
            
            assert manager.locales_dir == "test_locales"
            assert manager.default_language == "en"
            assert manager.current_language == "en"
            assert isinstance(manager.texts, dict)
            assert manager.dry_mode is True
            mock_makedirs.assert_not_called()
    
    def test_get_available_languages(self):
        """Тест получения списка доступных языков."""
        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = ["en.yaml", "ru.yml", "fr.txt", "de.yaml"]
            
            manager = LocalizationManager(locales_dir="test_locales")
            languages = manager._get_available_languages()
            
            assert set(languages) == {"en", "ru", "de"}
            mock_listdir.assert_called_once_with("test_locales")
    
    def test_get_available_languages_dry_mode(self):
        """Тест получения списка доступных языков в режиме dry_mode."""
        with patch('os.listdir') as mock_listdir:
            manager = LocalizationManager(locales_dir="test_locales", default_language="en", dry_mode=True)
            languages = manager._get_available_languages()
            
            assert languages == ["en"]
            mock_listdir.assert_not_called()
    
    def test_load_language(self):
        """Тест загрузки языка."""
        mock_yaml_data = {
            "common": {
                "yes": "Yes",
                "no": "No"
            }
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('yaml.safe_load') as mock_yaml_load:
                mock_yaml_load.return_value = mock_yaml_data
                
                manager = LocalizationManager(locales_dir="test_locales")
                result = manager.load_language("en")
                
                assert result is True
                assert manager.texts["en"] == mock_yaml_data
                mock_file.assert_called_once_with(os.path.join("test_locales", "en.yaml"), 'r', encoding='utf-8')
                mock_yaml_load.assert_called_once()
    
    def test_load_language_dry_mode(self):
        """Тест загрузки языка в режиме dry_mode."""
        with patch('builtins.open') as mock_file:
            with patch('yaml.safe_load') as mock_yaml_load:
                manager = LocalizationManager(locales_dir="test_locales", dry_mode=True)
                result = manager.load_language("en")
                
                assert result is True
                assert "en" in manager.texts
                assert manager.texts["en"] == {}
                mock_file.assert_not_called()
                mock_yaml_load.assert_not_called()
    
    def test_load_language_file_not_found(self):
        """Тест загрузки языка при отсутствии файла."""
        with patch('builtins.open') as mock_file:
            mock_file.side_effect = FileNotFoundError
            
            with patch.object(LocalizationManager, '_create_default_locale_file') as mock_create:
                manager = LocalizationManager(locales_dir="test_locales")
                manager.default_language = "test_lang"  # Устанавливаем другой язык по умолчанию
                
                result = manager.load_language("en")
                
                assert result is False
                mock_create.assert_not_called()
    
    def test_load_language_default_file_not_found(self):
        """Тест загрузки языка по умолчанию при отсутствии файла."""
        with patch('builtins.open') as mock_file:
            mock_file.side_effect = FileNotFoundError
            
            with patch.object(LocalizationManager, '_create_default_locale_file') as mock_create:
                with patch.object(LocalizationManager, 'load_language', wraps=LocalizationManager.load_language) as mock_load:
                    manager = LocalizationManager(locales_dir="test_locales", default_language="en")
                    
                    # Сбрасываем счетчик вызовов, так как конструктор уже вызвал load_language
                    mock_load.reset_mock()
                    
                    # Первый вызов вызовет FileNotFoundError и создаст файл по умолчанию
                    # Второй вызов должен загрузить созданный файл
                    mock_file.side_effect = [FileNotFoundError, None]
                    
                    result = manager.load_language("en")
                    
                    mock_create.assert_called_once()
                    assert mock_load.call_count == 2
    
    def test_load_language_yaml_error(self):
        """Тест загрузки языка при ошибке парсинга YAML."""
        with patch('builtins.open', mock_open()):
            with patch('yaml.safe_load') as mock_yaml_load:
                mock_yaml_load.side_effect = yaml.YAMLError
                
                manager = LocalizationManager(locales_dir="test_locales")
                
                with pytest.raises(ConfigLoadError):
                    manager.load_language("en")
    
    def test_create_default_locale_file(self):
        """Тест создания файла локализации по умолчанию."""
        with patch('os.makedirs') as mock_makedirs:
            with patch('builtins.open', mock_open()) as mock_file:
                with patch('yaml.dump') as mock_yaml_dump:
                    manager = LocalizationManager(locales_dir="test_locales")
                    file_path = os.path.join("test_locales", "en.yaml")
                    
                    manager._create_default_locale_file(file_path)
                    
                    mock_makedirs.assert_called_once_with(os.path.dirname(file_path), exist_ok=True)
                    mock_file.assert_called_once_with(file_path, 'w', encoding='utf-8')
                    mock_yaml_dump.assert_called_once()
    
    def test_create_default_locale_file_dry_mode(self):
        """Тест создания файла локализации по умолчанию в режиме dry_mode."""
        with patch('os.makedirs') as mock_makedirs:
            with patch('builtins.open') as mock_file:
                with patch('yaml.dump') as mock_yaml_dump:
                    manager = LocalizationManager(locales_dir="test_locales", dry_mode=True)
                    file_path = os.path.join("test_locales", "en.yaml")
                    
                    manager._create_default_locale_file(file_path)
                    
                    assert manager.default_language in manager.texts
                    assert isinstance(manager.texts[manager.default_language], dict)
                    mock_makedirs.assert_not_called()
                    mock_file.assert_not_called()
                    mock_yaml_dump.assert_not_called()
    
    def test_set_language(self):
        """Тест установки языка."""
        manager = LocalizationManager()
        
        # Добавляем тестовые данные
        manager.texts["en"] = {"test": "value"}
        
        result = manager.set_language("en")
        
        assert result is True
        assert manager.current_language == "en"
    
    def test_set_language_not_loaded(self):
        """Тест установки языка, который еще не загружен."""
        with patch.object(LocalizationManager, 'load_language') as mock_load:
            mock_load.return_value = True
            
            manager = LocalizationManager()
            result = manager.set_language("fr")
            
            assert result is True
            assert manager.current_language == "fr"
            mock_load.assert_called_once_with("fr")
    
    def test_set_language_load_failed(self):
        """Тест установки языка при неудачной загрузке."""
        with patch.object(LocalizationManager, 'load_language') as mock_load:
            mock_load.return_value = False
            
            manager = LocalizationManager()
            result = manager.set_language("fr")
            
            assert result is False
            assert manager.current_language != "fr"
            mock_load.assert_called_once_with("fr")
    
    def test_get_text(self):
        """Тест получения текста."""
        manager = LocalizationManager()
        
        # Добавляем тестовые данные
        manager.texts["ru"] = {
            "section": {
                "key": "Значение"
            }
        }
        manager.current_language = "ru"
        
        text = manager.get_text("section.key")
        
        assert text == "Значение"
    
    def test_get_text_with_params(self):
        """Тест получения текста с параметрами."""
        manager = LocalizationManager()
        
        # Добавляем тестовые данные
        manager.texts["ru"] = {
            "section": {
                "key": "Значение {param}"
            }
        }
        manager.current_language = "ru"
        
        text = manager.get_text("section.key", param="тест")
        
        assert text == "Значение тест"
    
    def test_get_text_not_found_default_language(self):
        """Тест получения текста, который отсутствует в текущем языке, но есть в языке по умолчанию."""
        manager = LocalizationManager(default_language="en")
        
        # Добавляем тестовые данные
        manager.texts["en"] = {
            "section": {
                "key": "Value"
            }
        }
        manager.texts["ru"] = {}
        manager.current_language = "ru"
        
        text = manager.get_text("section.key")
        
        assert text == "Value"
    
    def test_get_text_not_found(self):
        """Тест получения текста, который отсутствует."""
        manager = LocalizationManager()
        
        # Добавляем тестовые данные
        manager.texts["ru"] = {}
        manager.current_language = "ru"
        
        text = manager.get_text("section.key")
        
        assert text == "section.key"
    
    def test_get_text_not_found_with_default(self):
        """Тест получения текста, который отсутствует, с указанием значения по умолчанию."""
        manager = LocalizationManager()
        
        # Добавляем тестовые данные
        manager.texts["ru"] = {}
        manager.current_language = "ru"
        
        text = manager.get_text("section.key", default="Значение по умолчанию")
        
        assert text == "Значение по умолчанию"
    
    def test_get_text_format_error(self):
        """Тест получения текста с ошибкой форматирования."""
        manager = LocalizationManager()
        
        # Добавляем тестовые данные
        manager.texts["ru"] = {
            "section": {
                "key": "Значение {param}"
            }
        }
        manager.current_language = "ru"
        
        # Вызываем без необходимого параметра
        text = manager.get_text("section.key")
        
        # Должен вернуть исходный текст без форматирования
        assert text == "Значение {param}"
    
    def test_get_text_by_key(self):
        """Тест получения текста по ключу."""
        manager = LocalizationManager()
        
        # Добавляем тестовые данные
        manager.texts["ru"] = {
            "section": {
                "subsection": {
                    "key": "Значение"
                }
            }
        }
        
        text = manager._get_text_by_key("ru", "section.subsection.key")
        
        assert text == "Значение"
    
    def test_get_text_by_key_not_found(self):
        """Тест получения текста по ключу, который отсутствует."""
        manager = LocalizationManager()
        
        # Добавляем тестовые данные
        manager.texts["ru"] = {
            "section": {}
        }
        
        text = manager._get_text_by_key("ru", "section.subsection.key")
        
        assert text is None
    
    def test_get_text_by_key_language_not_loaded(self):
        """Тест получения текста по ключу для языка, который не загружен."""
        manager = LocalizationManager()
        
        text = manager._get_text_by_key("fr", "section.key")
        
        assert text is None
    
    def test_get_text_by_key_not_string(self):
        """Тест получения текста по ключу, значение которого не является строкой."""
        manager = LocalizationManager()
        
        # Добавляем тестовые данные
        manager.texts["ru"] = {
            "section": {
                "key": {
                    "subkey": "Значение"
                }
            }
        }
        
        text = manager._get_text_by_key("ru", "section.key")
        
        assert text is None
    
    def test_save_texts(self):
        """Тест сохранения текстов."""
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('yaml.dump') as mock_yaml_dump:
                manager = LocalizationManager(locales_dir="test_locales")
                
                # Добавляем тестовые данные
                manager.texts["ru"] = {
                    "section": {
                        "key": "Значение"
                    }
                }
                
                result = manager.save_texts("ru")
                
                assert result is True
                mock_file.assert_called_once_with(os.path.join("test_locales", "ru.yaml"), 'w', encoding='utf-8')
                mock_yaml_dump.assert_called_once()
    
    def test_save_texts_dry_mode(self):
        """Тест сохранения текстов в режиме dry_mode."""
        with patch('builtins.open') as mock_file:
            with patch('yaml.dump') as mock_yaml_dump:
                manager = LocalizationManager(locales_dir="test_locales", dry_mode=True)
                
                # Добавляем тестовые данные
                manager.texts["ru"] = {
                    "section": {
                        "key": "Значение"
                    }
                }
                
                result = manager.save_texts("ru")
                
                assert result is True
                mock_file.assert_not_called()
                mock_yaml_dump.assert_not_called()
    
    def test_save_texts_language_not_loaded(self):
        """Тест сохранения текстов для языка, который не загружен."""
        manager = LocalizationManager()
        
        result = manager.save_texts("fr")
        
        assert result is False
    
    def test_save_texts_error(self):
        """Тест сохранения текстов при возникновении ошибки."""
        with patch('builtins.open') as mock_file:
            mock_file.side_effect = Exception("Test error")
            
            manager = LocalizationManager()
            
            # Добавляем тестовые данные
            manager.texts["ru"] = {
                "section": {
                    "key": "Значение"
                }
            }
            
            result = manager.save_texts("ru")
            
            assert result is False
    
    def test_add_text(self):
        """Тест добавления текста."""
        manager = LocalizationManager()
        
        # Добавляем тестовые данные
        manager.texts["ru"] = {
            "section": {}
        }
        
        result = manager.add_text("ru", "section.key", "Значение")
        
        assert result is True
        assert manager.texts["ru"]["section"]["key"] == "Значение"
    
    def test_add_text_nested(self):
        """Тест добавления текста с вложенными секциями."""
        manager = LocalizationManager()
        
        # Добавляем тестовые данные
        manager.texts["ru"] = {}
        
        result = manager.add_text("ru", "section.subsection.key", "Значение")
        
        assert result is True
        assert manager.texts["ru"]["section"]["subsection"]["key"] == "Значение"
    
    def test_add_text_language_not_loaded(self):
        """Тест добавления текста для языка, который не загружен."""
        with patch.object(LocalizationManager, 'load_language') as mock_load:
            mock_load.return_value = False
            
            manager = LocalizationManager()
            
            result = manager.add_text("fr", "section.key", "Value")
            
            assert result is True
            assert "fr" in manager.texts
            assert manager.texts["fr"]["section"]["key"] == "Value"
            mock_load.assert_called_once_with("fr")
    
    def test_add_text_replace_non_dict(self):
        """Тест добавления текста с заменой не-словаря."""
        manager = LocalizationManager()
        
        # Добавляем тестовые данные
        manager.texts["ru"] = {
            "section": "Не словарь"
        }
        
        result = manager.add_text("ru", "section.key", "Значение")
        
        assert result is True
        assert isinstance(manager.texts["ru"]["section"], dict)
        assert manager.texts["ru"]["section"]["key"] == "Значение"
    
    def test_get_all_keys(self):
        """Тест получения всех ключей."""
        manager = LocalizationManager()
        
        # Добавляем тестовые данные
        manager.texts["ru"] = {
            "section1": {
                "key1": "Значение 1",
                "key2": "Значение 2"
            },
            "section2": {
                "subsection": {
                    "key3": "Значение 3"
                }
            }
        }
        
        keys = manager.get_all_keys()
        
        assert set(keys) == {"section1.key1", "section1.key2", "section2.subsection.key3"}
    
    def test_get_all_keys_with_prefix(self):
        """Тест получения всех ключей с префиксом."""
        manager = LocalizationManager()
        
        # Добавляем тестовые данные
        manager.texts["ru"] = {
            "section1": {
                "key1": "Значение 1",
                "key2": "Значение 2"
            },
            "section2": {
                "subsection": {
                    "key3": "Значение 3"
                }
            }
        }
        
        keys = manager.get_all_keys(prefix="section1")
        
        assert set(keys) == {"section1.key1", "section1.key2"}
    
    def test_get_all_keys_language_not_loaded(self):
        """Тест получения всех ключей, когда язык по умолчанию не загружен."""
        manager = LocalizationManager()
        
        # Очищаем словарь текстов
        manager.texts = {}
        
        keys = manager.get_all_keys()
        
        assert keys == []
    
    def test_collect_keys(self):
        """Тест сбора ключей."""
        manager = LocalizationManager()
        
        data = {
            "key1": "Значение 1",
            "section": {
                "key2": "Значение 2",
                "subsection": {
                    "key3": "Значение 3"
                }
            },
            "not_string": {
                "not_string2": {}
            }
        }
        
        keys = []
        manager._collect_keys(data, keys)
        
        assert set(keys) == {"key1", "section.key2", "section.subsection.key3"}
    
    def test_collect_keys_with_prefix(self):
        """Тест сбора ключей с префиксом."""
        manager = LocalizationManager()
        
        data = {
            "key1": "Значение 1",
            "section": {
                "key2": "Значение 2",
                "subsection": {
                    "key3": "Значение 3"
                }
            }
        }
        
        keys = []
        manager._collect_keys(data, keys, prefix="section")
        
        assert set(keys) == {"section.key2", "section.subsection.key3"}
    
    def test_collect_keys_with_current_path(self):
        """Тест сбора ключей с текущим путем."""
        manager = LocalizationManager()
        
        data = {
            "key1": "Значение 1",
            "key2": "Значение 2"
        }
        
        keys = []
        manager._collect_keys(data, keys, current_path="prefix")
        
        assert set(keys) == {"prefix.key1", "prefix.key2"}


class TestGlobalFunctions:
    """Тесты для глобальных функций модуля локализации."""
    
    def test_set_dry_mode(self):
        """Тест установки режима dry_mode."""
        with patch('core.localization._localization_manager', None):
            set_dry_mode(True)
            
            from core.localization import _dry_mode
            assert _dry_mode is True
    
    def test_get_localization_manager(self):
        """Тест получения глобального экземпляра менеджера локализации."""
        with patch('core.localization._localization_manager', None):
            with patch('core.localization.LocalizationManager') as mock_manager_class:
                mock_manager = MagicMock()
                mock_manager_class.return_value = mock_manager
                
                manager = get_localization_manager()
                
                assert manager is mock_manager
                mock_manager_class.assert_called_once_with(dry_mode=False)
    
    def test_get_localization_manager_dry_mode(self):
        """Тест получения глобального экземпляра менеджера локализации в режиме dry_mode."""
        with patch('core.localization._localization_manager', None):
            with patch('core.localization.LocalizationManager') as mock_manager_class:
                with patch('core.localization._dry_mode', True):
                    mock_manager = MagicMock()
                    mock_manager_class.return_value = mock_manager
                    
                    manager = get_localization_manager()
                    
                    assert manager is mock_manager
                    mock_manager_class.assert_called_once_with(dry_mode=True)
    
    def test_get_localization_manager_existing(self):
        """Тест получения существующего глобального экземпляра менеджера локализации."""
        mock_manager = MagicMock()
        
        with patch('core.localization._localization_manager', mock_manager):
            with patch('core.localization.LocalizationManager') as mock_manager_class:
                manager = get_localization_manager()
                
                assert manager is mock_manager
                mock_manager_class.assert_not_called()
    
    def test_get_text(self):
        """Тест функции get_text."""
        mock_manager = MagicMock()
        mock_manager.get_text.return_value = "Тестовый текст"
        
        with patch('core.localization.get_localization_manager') as mock_get_manager:
            mock_get_manager.return_value = mock_manager
            
            text = get_text("test.key", default="Default", param="value")
            
            assert text == "Тестовый текст"
            mock_get_manager.assert_called_once()
            mock_manager.get_text.assert_called_once_with("test.key", "Default", param="value")
    
    def test_set_language(self):
        """Тест функции set_language."""
        mock_manager = MagicMock()
        mock_manager.set_language.return_value = True
        
        with patch('core.localization.get_localization_manager') as mock_get_manager:
            mock_get_manager.return_value = mock_manager
            
            result = set_language("en")
            
            assert result is True
            mock_get_manager.assert_called_once()
            mock_manager.set_language.assert_called_once_with("en") 