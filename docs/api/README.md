# API Документация

В этом разделе содержится документация по API различных компонентов системы Leon Trading Bot.

## Модули

- **Core** - Документация по основным модулям системы
  - [Модуль отказоустойчивости](core/resilience.md) - API модуля отказоустойчивости
  - [Модуль обработки ошибок](core/exceptions.md) - API модуля обработки ошибок
  - [Модуль конфигурации](core/config.md) - API модуля конфигурации
  - [Модуль оркестрации](core/orchestrator.md) - API оркестратора и управления состоянием
  - [Модуль интеграции с ML](core/ml_integration.md) - API интеграции с ML-моделями

- **Exchange** - Документация по модулям взаимодействия с биржами
  - [Binance API](exchange/binance.md) - API для работы с Binance

- **Data** - Документация по модулям работы с данными
  - [Хранилище данных](data/storage.md) - API хранилища данных

- **Trading** - Документация по модулям торговли
  - [Стратегии](trading/strategies.md) - API торговых стратегий
  - [Ордера](trading/orders.md) - API работы с ордерами

- **ML** - Документация по модулям машинного обучения
  - [Модели](ml/models.md) - API моделей машинного обучения
  - [Предсказания](ml/predictions.md) - API предсказаний

- **Notification** - Документация по модулям уведомлений
  - [Telegram API](notification/telegram.md) - API для работы с Telegram ботом

## Общие интерфейсы

- [События](common/events.md) - Система событий для коммуникации между компонентами
- [Команды](common/commands.md) - Система команд для управления ботом

## Использование API

Все API модули предоставляют документированные интерфейсы для взаимодействия с соответствующими компонентами системы. Каждый модуль содержит описание доступных методов, их параметров, возвращаемых значений и возможных исключений.

## Версионирование

API следует семантическому версионированию (SemVer). Изменения, нарушающие обратную совместимость, будут отражены в увеличении мажорной версии.

## Формат документации

Документация API следует следующему формату:

1. **Обзор** - Краткое описание модуля и его назначения
2. **Классы и функции** - Описание публичных классов и функций модуля
3. **Примеры использования** - Примеры кода, демонстрирующие использование API
4. **Исключения** - Список исключений, которые могут быть вызваны при использовании API
5. **Зависимости** - Список зависимостей модуля от других модулей системы 