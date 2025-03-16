"""
Модуль для работы с рыночными данными в реальном времени.

Предоставляет функциональность для получения и обработки
рыночных данных в реальном времени.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable, Set
from datetime import datetime, timedelta
import json
import pandas as pd

from data.storage import DataStorage


class MarketDataManager:
    """
    Менеджер рыночных данных в реальном времени.
    
    Отвечает за получение, обработку и распространение рыночных данных
    в реальном времени.
    """
    
    def __init__(self, storage: Optional[DataStorage] = None):
        """
        Инициализация менеджера рыночных данных.
        
        Args:
            storage: Хранилище данных
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.storage = storage or DataStorage()
        
        # Словарь для хранения последних данных по символам
        self.latest_data: Dict[str, Dict[str, Any]] = {}
        
        # Словарь для хранения подписок на обновления
        self.subscriptions: Dict[str, Set[Callable]] = {}
        
        # Флаг работы менеджера
        self.is_running = False
        
        # Задача для обновления данных
        self.update_task = None
        
        self.logger.info("Инициализирован менеджер рыночных данных")
    
    async def start(self) -> bool:
        """
        Запуск менеджера рыночных данных.
        
        Returns:
            True, если менеджер успешно запущен, иначе False
        """
        if self.is_running:
            self.logger.warning("Менеджер рыночных данных уже запущен")
            return False
        
        self.is_running = True
        self.logger.info("Менеджер рыночных данных запущен")
        return True
    
    async def stop(self) -> bool:
        """
        Остановка менеджера рыночных данных.
        
        Returns:
            True, если менеджер успешно остановлен, иначе False
        """
        if not self.is_running:
            self.logger.warning("Менеджер рыночных данных не запущен")
            return False
        
        self.is_running = False
        
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
            self.update_task = None
        
        self.logger.info("Менеджер рыночных данных остановлен")
        return True
    
    def subscribe(self, symbol: str, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Подписка на обновления данных по символу.
        
        Args:
            symbol: Символ торговой пары
            callback: Функция обратного вызова
            
        Returns:
            True, если подписка успешно оформлена, иначе False
        """
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = set()
        
        self.subscriptions[symbol].add(callback)
        self.logger.debug(f"Оформлена подписка на {symbol}")
        return True
    
    def unsubscribe(self, symbol: str, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Отписка от обновлений данных по символу.
        
        Args:
            symbol: Символ торговой пары
            callback: Функция обратного вызова
            
        Returns:
            True, если отписка успешно оформлена, иначе False
        """
        if symbol not in self.subscriptions:
            self.logger.warning(f"Нет подписок на {symbol}")
            return False
        
        if callback not in self.subscriptions[symbol]:
            self.logger.warning(f"Подписка не найдена для {symbol}")
            return False
        
        self.subscriptions[symbol].remove(callback)
        
        if not self.subscriptions[symbol]:
            del self.subscriptions[symbol]
        
        self.logger.debug(f"Отменена подписка на {symbol}")
        return True
    
    def get_latest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Получение последних данных по символу.
        
        Args:
            symbol: Символ торговой пары
            
        Returns:
            Словарь с последними данными или None, если данные не найдены
        """
        return self.latest_data.get(symbol)
    
    async def update_data(self, symbol: str, data: Dict[str, Any]) -> bool:
        """
        Обновление данных по символу.
        
        Args:
            symbol: Символ торговой пары
            data: Новые данные
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        # Обновление данных
        self.latest_data[symbol] = data
        
        # Оповещение подписчиков
        if symbol in self.subscriptions:
            for callback in self.subscriptions[symbol]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Ошибка при вызове callback для {symbol}: {e}")
        
        self.logger.debug(f"Обновлены данные для {symbol}")
        return True
    
    async def start_websocket(self, client, symbols: List[str], interval: str = "1m") -> bool:
        """
        Запуск WebSocket для получения данных в реальном времени.
        
        Args:
            client: Клиент биржи
            symbols: Список символов торговых пар
            interval: Интервал свечей
            
        Returns:
            True, если WebSocket успешно запущен, иначе False
        """
        if not self.is_running:
            self.logger.warning("Менеджер рыночных данных не запущен")
            return False
        
        # Создание задачи для обновления данных
        self.update_task = asyncio.create_task(
            self._websocket_task(client, symbols, interval)
        )
        
        self.logger.info(f"Запущен WebSocket для {symbols}")
        return True
    
    async def _websocket_task(self, client, symbols: List[str], interval: str) -> None:
        """
        Задача для работы с WebSocket.
        
        Args:
            client: Клиент биржи
            symbols: Список символов торговых пар
            interval: Интервал свечей
        """
        try:
            # Подписка на WebSocket
            await client.start_kline_socket(symbols, interval, self._process_kline_message)
            
            # Ожидание остановки менеджера
            while self.is_running:
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Ошибка в WebSocket: {e}")
        finally:
            # Отписка от WebSocket
            await client.stop_kline_socket()
    
    async def _process_kline_message(self, message: Dict[str, Any]) -> None:
        """
        Обработка сообщения от WebSocket.
        
        Args:
            message: Сообщение от WebSocket
        """
        try:
            # Извлечение данных из сообщения
            symbol = message.get("s", "").upper()
            kline = message.get("k", {})
            
            if not symbol or not kline:
                self.logger.warning(f"Некорректное сообщение: {message}")
                return
            
            # Формирование данных
            data = {
                "symbol": symbol,
                "interval": kline.get("i", ""),
                "open_time": kline.get("t", 0),
                "open": float(kline.get("o", 0)),
                "high": float(kline.get("h", 0)),
                "low": float(kline.get("l", 0)),
                "close": float(kline.get("c", 0)),
                "volume": float(kline.get("v", 0)),
                "close_time": kline.get("T", 0),
                "is_closed": kline.get("x", False),
                "timestamp": datetime.now().timestamp() * 1000
            }
            
            # Обновление данных
            await self.update_data(symbol, data)
            
            # Сохранение закрытых свечей
            if data["is_closed"]:
                await self._save_closed_kline(symbol, data)
                
        except Exception as e:
            self.logger.error(f"Ошибка при обработке сообщения: {e}")
    
    async def _save_closed_kline(self, symbol: str, data: Dict[str, Any]) -> None:
        """
        Сохранение закрытой свечи.
        
        Args:
            symbol: Символ торговой пары
            data: Данные свечи
        """
        try:
            # Формирование данных для сохранения
            kline = [
                data["open_time"],
                data["open"],
                data["high"],
                data["low"],
                data["close"],
                data["volume"],
                data["close_time"],
                0,  # quote_asset_volume
                0,  # number_of_trades
                0,  # taker_buy_base_asset_volume
                0,  # taker_buy_quote_asset_volume
                0   # ignore
            ]
            
            # Сохранение в хранилище
            await self.storage.save_kline(symbol, data["interval"], kline)
            
            self.logger.debug(f"Сохранена закрытая свеча для {symbol}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении закрытой свечи: {e}")
    
    async def get_ticker(self, client, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Получение текущего тикера.
        
        Args:
            client: Клиент биржи
            symbol: Символ торговой пары
            
        Returns:
            Словарь с данными тикера или None, если данные не найдены
        """
        try:
            # Получение тикера
            ticker = await client.get_ticker(symbol=symbol)
            
            if not ticker:
                self.logger.warning(f"Не удалось получить тикер для {symbol}")
                return None
            
            # Формирование данных
            data = {
                "symbol": symbol,
                "price": float(ticker.get("lastPrice", 0)),
                "price_change": float(ticker.get("priceChange", 0)),
                "price_change_percent": float(ticker.get("priceChangePercent", 0)),
                "volume": float(ticker.get("volume", 0)),
                "high": float(ticker.get("highPrice", 0)),
                "low": float(ticker.get("lowPrice", 0)),
                "timestamp": datetime.now().timestamp() * 1000
            }
            
            # Обновление данных
            await self.update_data(symbol, data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении тикера: {e}")
            return None
    
    async def get_order_book(self, client, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """
        Получение книги ордеров.
        
        Args:
            client: Клиент биржи
            symbol: Символ торговой пары
            limit: Количество ордеров
            
        Returns:
            Словарь с данными книги ордеров или None, если данные не найдены
        """
        try:
            # Получение книги ордеров
            order_book = await client.get_order_book(symbol=symbol, limit=limit)
            
            if not order_book:
                self.logger.warning(f"Не удалось получить книгу ордеров для {symbol}")
                return None
            
            # Формирование данных
            data = {
                "symbol": symbol,
                "bids": order_book.get("bids", []),
                "asks": order_book.get("asks", []),
                "timestamp": datetime.now().timestamp() * 1000
            }
            
            # Обновление данных
            key = f"{symbol}_order_book"
            await self.update_data(key, data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении книги ордеров: {e}")
            return None
    
    async def analyze_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Анализ настроения рынка.
        
        Args:
            symbol: Символ торговой пары
            
        Returns:
            Словарь с результатами анализа
        """
        # Получение последних данных
        data = self.get_latest_data(symbol)
        if not data:
            self.logger.warning(f"Нет данных для анализа настроения рынка {symbol}")
            return {"symbol": symbol, "sentiment": "unknown", "confidence": 0.0}
        
        # Получение данных книги ордеров
        order_book_key = f"{symbol}_order_book"
        order_book = self.get_latest_data(order_book_key)
        
        # Анализ настроения рынка
        sentiment = "neutral"
        confidence = 0.5
        
        # Анализ на основе изменения цены
        if "price_change_percent" in data:
            price_change = data["price_change_percent"]
            if price_change > 2.0:
                sentiment = "bullish"
                confidence = min(0.5 + price_change / 10.0, 0.9)
            elif price_change < -2.0:
                sentiment = "bearish"
                confidence = min(0.5 + abs(price_change) / 10.0, 0.9)
        
        # Анализ на основе книги ордеров
        if order_book:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            
            if bids and asks:
                # Расчет объема ордеров на покупку и продажу
                bid_volume = sum(float(bid[1]) for bid in bids)
                ask_volume = sum(float(ask[1]) for ask in asks)
                
                # Соотношение объемов
                volume_ratio = bid_volume / ask_volume if ask_volume > 0 else 1.0
                
                # Корректировка настроения на основе соотношения объемов
                if volume_ratio > 1.5:
                    if sentiment == "bullish":
                        confidence = min(confidence + 0.1, 0.95)
                    else:
                        sentiment = "bullish"
                        confidence = max(confidence, 0.6)
                elif volume_ratio < 0.67:
                    if sentiment == "bearish":
                        confidence = min(confidence + 0.1, 0.95)
                    else:
                        sentiment = "bearish"
                        confidence = max(confidence, 0.6)
        
        return {
            "symbol": symbol,
            "sentiment": sentiment,
            "confidence": confidence,
            "timestamp": datetime.now().timestamp() * 1000
        }
    
    async def detect_market_anomalies(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Обнаружение аномалий на рынке.
        
        Args:
            symbol: Символ торговой пары
            
        Returns:
            Список обнаруженных аномалий
        """
        # Получение последних данных
        data = self.get_latest_data(symbol)
        if not data:
            self.logger.warning(f"Нет данных для обнаружения аномалий {symbol}")
            return []
        
        anomalies = []
        
        # Проверка на резкое изменение цены
        if "price_change_percent" in data:
            price_change = data["price_change_percent"]
            if abs(price_change) > 5.0:
                anomalies.append({
                    "type": "price_change",
                    "description": f"Резкое изменение цены: {price_change:.2f}%",
                    "severity": "high" if abs(price_change) > 10.0 else "medium",
                    "timestamp": datetime.now().timestamp() * 1000
                })
        
        # Проверка на необычно высокий объем
        if "volume" in data and "symbol" in data:
            # Получение среднего объема за последние 24 часа
            try:
                avg_volume = await self.storage.get_average_volume(data["symbol"], "1h", 24)
                if avg_volume > 0 and data["volume"] > avg_volume * 3:
                    anomalies.append({
                        "type": "high_volume",
                        "description": f"Необычно высокий объем: {data['volume']:.2f} (в {data['volume']/avg_volume:.2f} раз выше среднего)",
                        "severity": "medium",
                        "timestamp": datetime.now().timestamp() * 1000
                    })
            except Exception as e:
                self.logger.error(f"Ошибка при получении среднего объема: {e}")
        
        return anomalies
    
    async def calculate_market_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Расчет рыночных метрик.
        
        Args:
            symbol: Символ торговой пары
            
        Returns:
            Словарь с рыночными метриками
        """
        # Получение последних данных
        data = self.get_latest_data(symbol)
        if not data:
            self.logger.warning(f"Нет данных для расчета рыночных метрик {symbol}")
            return {}
        
        # Получение данных книги ордеров
        order_book_key = f"{symbol}_order_book"
        order_book = self.get_latest_data(order_book_key)
        
        metrics = {
            "symbol": symbol,
            "timestamp": datetime.now().timestamp() * 1000
        }
        
        # Добавление метрик из данных тикера
        if "price" in data:
            metrics["price"] = data["price"]
        if "price_change_percent" in data:
            metrics["price_change_percent"] = data["price_change_percent"]
        if "volume" in data:
            metrics["volume"] = data["volume"]
        
        # Расчет метрик на основе книги ордеров
        if order_book:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            
            if bids and asks:
                # Расчет объема ордеров на покупку и продажу
                bid_volume = sum(float(bid[1]) for bid in bids)
                ask_volume = sum(float(ask[1]) for ask in asks)
                
                # Соотношение объемов
                volume_ratio = bid_volume / ask_volume if ask_volume > 0 else 1.0
                
                # Спред
                best_bid = float(bids[0][0]) if bids else 0
                best_ask = float(asks[0][0]) if asks else 0
                spread = (best_ask - best_bid) / best_bid * 100 if best_bid > 0 else 0
                
                metrics.update({
                    "bid_volume": bid_volume,
                    "ask_volume": ask_volume,
                    "volume_ratio": volume_ratio,
                    "spread": spread,
                    "best_bid": best_bid,
                    "best_ask": best_ask
                })
        
        return metrics
    
    async def get_historical_data(self, client, symbol: str, interval: str, 
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None,
                                 limit: int = 1000,
                                 add_indicators: bool = True) -> pd.DataFrame:
        """
        Получение исторических данных.
        
        Args:
            client: Клиент биржи
            symbol: Символ торговой пары
            interval: Интервал свечей
            start_date: Начальная дата
            end_date: Конечная дата
            limit: Максимальное количество свечей
            add_indicators: Добавить технические индикаторы
            
        Returns:
            DataFrame с историческими данными
        """
        from data.historical_data import HistoricalDataManager
        
        try:
            # Создание менеджера исторических данных
            historical_data_manager = HistoricalDataManager(self.storage, client)
            
            # Загрузка исторических данных
            df = await historical_data_manager.load_historical_data(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
                add_indicators=add_indicators,
                detect_patterns=False
            )
            
            # Ограничение количества свечей
            if limit and len(df) > limit:
                df = df.tail(limit)
            
            self.logger.info(f"Получено {len(df)} исторических свечей для {symbol} {interval}")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении исторических данных: {e}")
            return pd.DataFrame()
    
    async def combine_data(self, client, symbol: str, interval: str, 
                          days: int = 7, add_indicators: bool = True) -> pd.DataFrame:
        """
        Объединение исторических и текущих рыночных данных.
        
        Args:
            client: Клиент биржи
            symbol: Символ торговой пары
            interval: Интервал свечей
            days: Количество дней для исторических данных
            add_indicators: Добавить технические индикаторы
            
        Returns:
            DataFrame с объединенными данными
        """
        try:
            # Получение исторических данных
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            historical_df = await self.get_historical_data(
                client, symbol, interval, start_date, end_date, add_indicators=add_indicators
            )
            
            if historical_df.empty:
                self.logger.warning(f"Нет исторических данных для {symbol} {interval}")
                return pd.DataFrame()
            
            # Получение текущих данных
            current_data = self.get_latest_data(symbol)
            
            if not current_data:
                self.logger.info(f"Нет текущих данных для {symbol}, возвращаем только исторические")
                return historical_df
            
            # Создание строки с текущими данными
            current_row = pd.Series({
                'open': current_data.get('open', historical_df['close'].iloc[-1]),
                'high': current_data.get('high', historical_df['close'].iloc[-1]),
                'low': current_data.get('low', historical_df['close'].iloc[-1]),
                'close': current_data.get('close', historical_df['close'].iloc[-1]),
                'volume': current_data.get('volume', 0)
            })
            
            # Добавление текущей строки к историческим данным
            current_time = pd.to_datetime(current_data.get('timestamp', datetime.now().timestamp() * 1000), unit='ms')
            combined_df = historical_df.copy()
            
            # Проверка, не дублируется ли последняя свеча
            if combined_df.index[-1] != current_time:
                combined_df.loc[current_time] = current_row
            
            # Пересчет индикаторов, если нужно
            if add_indicators:
                combined_df = self.storage.add_technical_indicators(combined_df)
            
            self.logger.info(f"Объединены исторические и текущие данные для {symbol} {interval}")
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Ошибка при объединении данных: {e}")
            return pd.DataFrame()


async def main():
    """Пример использования менеджера рыночных данных."""
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создание хранилища данных
    storage = DataStorage()
    
    # Создание клиента Binance
    from core.config_manager import ConfigManager
    from exchange.binance.client import BinanceClient
    
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    client = BinanceClient(
        api_key=config["binance"]["api_key"],
        api_secret=config["binance"]["api_secret"],
        testnet=config["binance"]["testnet"]
    )
    
    # Инициализация клиента
    await client.initialize()
    
    try:
        # Создание менеджера рыночных данных
        market_data_manager = MarketDataManager(storage)
        
        # Запуск менеджера
        await market_data_manager.start()
        
        # Символ для отслеживания
        symbol = "BTCUSDT"
        
        # Получение тикера
        ticker = await market_data_manager.get_ticker(client, symbol)
        print(f"Тикер {symbol}:")
        print(json.dumps(ticker, indent=2, ensure_ascii=False))
        
        # Получение книги ордеров
        order_book = await market_data_manager.get_order_book(client, symbol)
        print(f"\nКнига ордеров {symbol}:")
        print(f"Лучшая цена покупки: {order_book['bids'][0][0]}")
        print(f"Лучшая цена продажи: {order_book['asks'][0][0]}")
        
        # Анализ настроения рынка
        sentiment = await market_data_manager.analyze_market_sentiment(symbol)
        print(f"\nНастроение рынка {symbol}:")
        print(json.dumps(sentiment, indent=2, ensure_ascii=False))
        
        # Обнаружение аномалий
        anomalies = await market_data_manager.detect_market_anomalies(symbol)
        if anomalies:
            print(f"\nОбнаруженные аномалии {symbol}:")
            print(json.dumps(anomalies, indent=2, ensure_ascii=False))
        else:
            print(f"\nАномалии не обнаружены для {symbol}")
        
        # Расчет рыночных метрик
        metrics = await market_data_manager.calculate_market_metrics(symbol)
        print(f"\nРыночные метрики {symbol}:")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        
        # Подписка на обновления
        def on_data_update(data):
            print(f"Получено обновление для {data['symbol']}: {data['close']}")
        
        market_data_manager.subscribe(symbol, on_data_update)
        
        # Запуск WebSocket
        await market_data_manager.start_websocket(client, [symbol])
        
        # Ожидание 30 секунд для получения обновлений
        print("\nОжидание обновлений...")
        await asyncio.sleep(30)
        
        # Остановка менеджера
        await market_data_manager.stop()
        
    finally:
        # Закрытие клиента
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
