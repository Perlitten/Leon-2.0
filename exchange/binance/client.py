"""
Клиент для работы с Binance API.

Предоставляет методы для взаимодействия с биржей Binance через REST API и WebSocket.
"""

import logging
import time
import hmac
import hashlib
import json
from urllib.parse import urlencode
import aiohttp
import asyncio
from typing import Dict, Any, Optional, List, Union

from exchange.base import ExchangeBase

class BinanceIntegration(ExchangeBase):
    """
    Клиент для работы с Binance API.
    Поддерживает как REST API, так и WebSocket соединения.
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        """
        Инициализация клиента Binance.
        
        Args:
            api_key: API ключ Binance
            api_secret: API секрет Binance
            testnet: Использовать тестовую сеть Binance (по умолчанию False)
        """
        super().__init__(api_key, api_secret, testnet)
        
        # Базовые URL для API
        if testnet:
            self.base_url = "https://testnet.binance.vision/api"
            self.wss_url = "wss://testnet.binance.vision/ws"
        else:
            self.base_url = "https://api.binance.com/api"
            self.wss_url = "wss://stream.binance.com:9443/ws"
        
        self.session = None
        self.ws_connections = {}
        self.callbacks = {}
        
    async def initialize(self):
        """Инициализация HTTP сессии"""
        self.session = aiohttp.ClientSession()
        self.logger.info("HTTP сессия инициализирована")
        
        # Проверка соединения
        try:
            await self.ping()
            self.logger.info("Соединение с Binance API установлено")
        except Exception as e:
            self.logger.error(f"Ошибка при подключении к Binance API: {e}")
            raise
    
    async def close(self):
        """Закрытие соединений"""
        if self.session:
            await self.session.close()
            self.logger.info("HTTP сессия закрыта")
        
        # Закрытие WebSocket соединений
        for symbol, ws in self.ws_connections.items():
            if not ws.closed:
                await ws.close()
                self.logger.info(f"WebSocket соединение для {symbol} закрыто")
        
        self.ws_connections = {}
    
    def _generate_signature(self, query_string: str) -> str:
        """
        Генерация подписи для запросов к Binance API.
        
        Args:
            query_string: Строка запроса
            
        Returns:
            Подпись запроса
        """
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _request(self, method: str, endpoint: str, signed: bool = False, 
                      params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Выполнение запроса к Binance API.
        
        Args:
            method: HTTP метод (GET, POST, DELETE)
            endpoint: Конечная точка API
            signed: Требуется ли подпись запроса
            params: Параметры запроса
            
        Returns:
            Ответ от API в формате JSON
        """
        if self.session is None:
            await self.initialize()
        
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            query_string = urlencode(params)
            params['signature'] = self._generate_signature(query_string)
        
        try:
            if method == "GET":
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Ошибка API: {response.status} - {error_text}")
                        raise Exception(f"Ошибка API: {response.status} - {error_text}")
                    return await response.json()
            elif method == "POST":
                async with self.session.post(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Ошибка API: {response.status} - {error_text}")
                        raise Exception(f"Ошибка API: {response.status} - {error_text}")
                    return await response.json()
            elif method == "DELETE":
                async with self.session.delete(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Ошибка API: {response.status} - {error_text}")
                        raise Exception(f"Ошибка API: {response.status} - {error_text}")
                    return await response.json()
            else:
                raise ValueError(f"Неподдерживаемый HTTP метод: {method}")
        except aiohttp.ClientError as e:
            self.logger.error(f"Ошибка HTTP запроса: {e}")
            raise
    
    # Публичные API методы
    
    async def ping(self) -> Dict[str, Any]:
        """Проверка соединения с API"""
        return await self._request("GET", "/v3/ping")
    
    async def get_server_time(self) -> Dict[str, Any]:
        """Получение времени сервера"""
        return await self._request("GET", "/v3/time")
    
    async def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Получение информации о бирже.
        
        Args:
            symbol: Торговая пара (опционально)
            
        Returns:
            Информация о бирже
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        return await self._request("GET", "/v3/exchangeInfo", params=params)
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Получение текущей цены для торговой пары.
        
        Args:
            symbol: Торговая пара
            
        Returns:
            Информация о текущей цене
        """
        params = {'symbol': symbol}
        return await self._request("GET", "/v3/ticker/price", params=params)
    
    async def get_klines(self, symbol: str, interval: str, 
                        start_time: Optional[int] = None, 
                        end_time: Optional[int] = None,
                        limit: int = 500) -> List[List[Union[int, str, float]]]:
        """
        Получение исторических свечей.
        
        Args:
            symbol: Торговая пара
            interval: Интервал свечей (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            start_time: Начальное время в миллисекундах (опционально)
            end_time: Конечное время в миллисекундах (опционально)
            limit: Максимальное количество свечей (по умолчанию 500, максимум 1000)
            
        Returns:
            Список свечей
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        return await self._request("GET", "/v3/klines", params=params)
    
    # Приватные API методы (требуют аутентификации)
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Получение информации об аккаунте.
        
        Returns:
            Информация об аккаунте
        """
        return await self._request("GET", "/v3/account", signed=True)
    
    async def place_order(self, symbol: str, side: str, order_type: str, 
                         quantity: float, price: Optional[float] = None,
                         time_in_force: str = "GTC", **kwargs) -> Dict[str, Any]:
        """
        Размещение ордера.
        
        Args:
            symbol: Торговая пара
            side: Сторона (BUY, SELL)
            order_type: Тип ордера (LIMIT, MARKET, STOP_LOSS, STOP_LOSS_LIMIT, TAKE_PROFIT, TAKE_PROFIT_LIMIT)
            quantity: Количество
            price: Цена (для LIMIT ордеров)
            time_in_force: Время действия ордера (GTC, IOC, FOK)
            **kwargs: Дополнительные параметры
            
        Returns:
            Информация о созданном ордере
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'timestamp': int(time.time() * 1000)
        }
        
        if order_type == "LIMIT":
            params['price'] = price
            params['timeInForce'] = time_in_force
        
        # Добавление дополнительных параметров
        params.update(kwargs)
        
        return await self._request("POST", "/v3/order", signed=True, params=params)
    
    async def cancel_order(self, symbol: str, order_id: Optional[int] = None, 
                          orig_client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Отмена ордера.
        
        Args:
            symbol: Торговая пара
            order_id: ID ордера (опционально)
            orig_client_order_id: ID клиентского ордера (опционально)
            
        Returns:
            Информация об отмененном ордере
        """
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = order_id
        elif orig_client_order_id:
            params['origClientOrderId'] = orig_client_order_id
        else:
            raise ValueError("Необходимо указать order_id или orig_client_order_id")
        
        return await self._request("DELETE", "/v3/order", signed=True, params=params)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Получение открытых ордеров.
        
        Args:
            symbol: Торговая пара (опционально)
            
        Returns:
            Список открытых ордеров
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return await self._request("GET", "/v3/openOrders", signed=True, params=params)
    
    # WebSocket методы
    
    async def subscribe_to_klines(self, symbol: str, interval: str, callback):
        """
        Подписка на поток свечей.
        
        Args:
            symbol: Торговая пара
            interval: Интервал свечей
            callback: Функция обратного вызова для обработки данных
        """
        stream_name = f"{symbol.lower()}@kline_{interval}"
        await self._subscribe_to_stream(stream_name, callback)
    
    async def subscribe_to_ticker(self, symbol: str, callback):
        """
        Подписка на поток тикеров.
        
        Args:
            symbol: Торговая пара
            callback: Функция обратного вызова для обработки данных
        """
        stream_name = f"{symbol.lower()}@ticker"
        await self._subscribe_to_stream(stream_name, callback)
    
    async def _subscribe_to_stream(self, stream_name: str, callback):
        """
        Подписка на WebSocket поток.
        
        Args:
            stream_name: Имя потока
            callback: Функция обратного вызова для обработки данных
        """
        if self.session is None:
            await self.initialize()
        
        url = f"{self.wss_url}/{stream_name}"
        
        try:
            ws = await self.session.ws_connect(url)
            self.ws_connections[stream_name] = ws
            self.callbacks[stream_name] = callback
            
            self.logger.info(f"Подписка на поток {stream_name} установлена")
            
            # Запуск обработчика сообщений
            asyncio.create_task(self._handle_websocket_messages(stream_name, ws))
            
        except Exception as e:
            self.logger.error(f"Ошибка при подписке на поток {stream_name}: {e}")
            raise
    
    async def _handle_websocket_messages(self, stream_name: str, ws):
        """
        Обработка сообщений WebSocket.
        
        Args:
            stream_name: Имя потока
            ws: WebSocket соединение
        """
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    callback = self.callbacks.get(stream_name)
                    if callback:
                        await callback(data)
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    self.logger.warning(f"WebSocket соединение для {stream_name} закрыто")
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f"WebSocket ошибка для {stream_name}: {msg.data}")
                    break
        except Exception as e:
            self.logger.error(f"Ошибка при обработке сообщений WebSocket для {stream_name}: {e}")
        finally:
            # Переподключение при разрыве соединения
            if not ws.closed:
                await ws.close()
            
            self.logger.info(f"Переподключение к потоку {stream_name}...")
            await asyncio.sleep(5)  # Задержка перед переподключением
            
            callback = self.callbacks.get(stream_name)
            if callback:
                await self._subscribe_to_stream(stream_name, callback)

    async def get_balance(self) -> float:
        """
        Получение текущего баланса аккаунта в USDT.
        
        Returns:
            float: Текущий баланс в USDT
        """
        try:
            # Если API ключи не предоставлены, возвращаем тестовый баланс
            if not self.api_key or not self.api_secret:
                self.logger.info("API ключи не предоставлены, возвращается тестовый баланс")
                return 1000.0
            
            # Получение информации об аккаунте
            account_info = await self.get_account_info()
            
            # Поиск баланса USDT
            usdt_balance = 0.0
            for asset in account_info.get("balances", []):
                if asset["asset"] == "USDT":
                    usdt_balance = float(asset["free"])
                    break
            
            self.logger.info(f"Получен баланс: {usdt_balance} USDT")
            return usdt_balance
        except Exception as e:
            self.logger.error(f"Ошибка при получении баланса: {e}")
            return 0.0


# Пример использования
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Загрузка переменных окружения
    load_dotenv()
    
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    async def main():
        # Создание клиента Binance
        client = BinanceIntegration(api_key, api_secret)
        
        try:
            # Инициализация клиента
            await client.initialize()
            
            # Получение информации о бирже
            exchange_info = await client.get_exchange_info("BTCUSDT")
            print("Exchange Info:", exchange_info)
            
            # Получение текущей цены
            ticker = await client.get_ticker("BTCUSDT")
            print("Current Price:", ticker)
            
            # Получение свечей
            klines = await client.get_klines("BTCUSDT", "1h", limit=10)
            print("Klines:", klines)
            
            # Получение информации об аккаунте
            account_info = await client.get_account_info()
            print("Account Info:", account_info)
            
        except Exception as e:
            logging.error(f"Ошибка: {e}")
        finally:
            # Закрытие соединений
            await client.close()
    
    # Запуск асинхронной функции
    asyncio.run(main()) 