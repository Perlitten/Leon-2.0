/**
 * Leon Trading Bot - Web Dashboard JavaScript
 */

// Глобальные переменные
let priceChart = null;
let balanceChart = null;
let updateInterval = null;
const UPDATE_INTERVAL_MS = 1000; // Интервал обновления данных в миллисекундах

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    console.log('Инициализация панели управления...');
    
    // Инициализация графиков
    initCharts();
    
    // Загрузка начальных данных
    fetchData();
    
    // Настройка интервала обновления
    updateInterval = setInterval(fetchData, UPDATE_INTERVAL_MS);
    
    console.log('Панель управления инициализирована');
});

// Инициализация графиков
function initCharts() {
    // Инициализация графика цены
    const priceChartDiv = document.getElementById('price-chart');
    if (priceChartDiv) {
        priceChart = Plotly.newPlot('price-chart', [], {
            title: 'График цены',
            xaxis: { title: 'Время' },
            yaxis: { title: 'Цена' },
            template: 'plotly_dark',
            autosize: true,
            margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 }
        });
    }
    
    // Инициализация графика баланса
    const balanceChartDiv = document.getElementById('balance-chart');
    if (balanceChartDiv) {
        balanceChart = Plotly.newPlot('balance-chart', [], {
            title: 'График баланса',
            xaxis: { title: 'Время' },
            yaxis: { title: 'Баланс' },
            template: 'plotly_dark',
            autosize: true,
            margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 }
        });
    }
}

// Загрузка данных с сервера
function fetchData() {
    // Загрузка общих данных
    fetch('/api/data')
        .then(response => response.json())
        .then(data => {
            updateSymbolAndMode(data);
            updateStats(data);
            updateTradingParams(data);
            updatePositions(data.positions);
            updateSignals(data.signals);
            updateRecommendation(data.recommendation, data.recommendation_color);
        })
        .catch(error => console.error('Ошибка при загрузке данных:', error));
    
    // Загрузка графика цены
    fetch('/api/chart/price')
        .then(response => response.json())
        .then(data => {
            if (!data.error) {
                Plotly.react('price-chart', data.data, data.layout);
            }
        })
        .catch(error => console.error('Ошибка при загрузке графика цены:', error));
    
    // Загрузка графика баланса
    fetch('/api/chart/balance')
        .then(response => response.json())
        .then(data => {
            if (!data.error) {
                Plotly.react('balance-chart', data.data, data.layout);
            }
        })
        .catch(error => console.error('Ошибка при загрузке графика баланса:', error));
}

// Обновление символа и режима
function updateSymbolAndMode(data) {
    const symbolElement = document.getElementById('symbol');
    const modeElement = document.getElementById('mode');
    
    if (symbolElement && data.symbol) {
        symbolElement.textContent = data.symbol;
    }
    
    if (modeElement && data.mode) {
        modeElement.textContent = data.mode;
        
        // Установка цвета в зависимости от режима
        if (data.mode === 'REAL') {
            modeElement.classList.add('text-danger-bright');
            modeElement.classList.remove('text-warning', 'text-info');
        } else if (data.mode === 'BACKTEST') {
            modeElement.classList.add('text-warning');
            modeElement.classList.remove('text-danger-bright', 'text-info');
        } else {
            modeElement.classList.add('text-info');
            modeElement.classList.remove('text-danger-bright', 'text-warning');
        }
    }
}

// Обновление статистики
function updateStats(data) {
    // Обновление баланса
    const initialBalanceElement = document.getElementById('initial-balance');
    const currentBalanceElement = document.getElementById('current-balance');
    const profitElement = document.getElementById('profit');
    
    if (initialBalanceElement) {
        initialBalanceElement.textContent = `$${data.initial_balance.toFixed(2)}`;
    }
    
    if (currentBalanceElement) {
        currentBalanceElement.textContent = `$${data.current_balance.toFixed(2)}`;
    }
    
    if (profitElement) {
        const profitSign = data.profit >= 0 ? '+' : '';
        profitElement.textContent = `${profitSign}$${data.profit.toFixed(2)} (${profitSign}${data.profit_percent.toFixed(2)}%)`;
        
        if (data.profit >= 0) {
            profitElement.classList.add('text-success-bright');
            profitElement.classList.remove('text-danger-bright');
        } else {
            profitElement.classList.add('text-danger-bright');
            profitElement.classList.remove('text-success-bright');
        }
    }
    
    // Обновление статистики сделок
    const totalTradesElement = document.getElementById('total-trades');
    const winningTradesElement = document.getElementById('winning-trades');
    const losingTradesElement = document.getElementById('losing-trades');
    const winRateElement = document.getElementById('win-rate');
    
    if (totalTradesElement) {
        totalTradesElement.textContent = data.total_trades;
    }
    
    if (winningTradesElement) {
        winningTradesElement.textContent = data.winning_trades;
    }
    
    if (losingTradesElement) {
        losingTradesElement.textContent = data.losing_trades;
    }
    
    if (winRateElement) {
        const winRate = data.total_trades > 0 ? (data.winning_trades / data.total_trades) * 100 : 0;
        winRateElement.textContent = `${winRate.toFixed(1)}%`;
        
        // Установка цвета в зависимости от винрейта
        if (winRate >= 60) {
            winRateElement.classList.add('text-success-bright');
            winRateElement.classList.remove('text-warning', 'text-danger-bright');
        } else if (winRate >= 40) {
            winRateElement.classList.add('text-warning');
            winRateElement.classList.remove('text-success-bright', 'text-danger-bright');
        } else {
            winRateElement.classList.add('text-danger-bright');
            winRateElement.classList.remove('text-success-bright', 'text-warning');
        }
    }
}

// Обновление параметров торговли
function updateTradingParams(data) {
    const leverageElement = document.getElementById('leverage');
    const riskPerTradeElement = document.getElementById('risk-per-trade');
    const stopLossElement = document.getElementById('stop-loss');
    const takeProfitElement = document.getElementById('take-profit');
    
    if (data.trading_params) {
        if (leverageElement) {
            leverageElement.textContent = `${data.trading_params.leverage}x`;
        }
        
        if (riskPerTradeElement) {
            riskPerTradeElement.textContent = `${data.trading_params.risk_per_trade.toFixed(1)}%`;
        }
        
        if (stopLossElement) {
            stopLossElement.textContent = `${data.trading_params.stop_loss.toFixed(1)}%`;
        }
        
        if (takeProfitElement) {
            takeProfitElement.textContent = `${data.trading_params.take_profit.toFixed(1)}%`;
        }
    }
}

// Обновление позиций
function updatePositions(positions) {
    const positionsTableBody = document.querySelector('#positions-table tbody');
    
    if (positionsTableBody) {
        // Очистка таблицы
        positionsTableBody.innerHTML = '';
        
        if (positions && positions.length > 0) {
            // Добавление позиций
            positions.forEach(position => {
                const row = document.createElement('tr');
                
                // Установка класса в зависимости от типа позиции
                if (position.type === 'LONG') {
                    row.classList.add('position-long');
                } else if (position.type === 'SHORT') {
                    row.classList.add('position-short');
                }
                
                // Форматирование PnL
                const pnlSign = position.pnl >= 0 ? '+' : '';
                const pnlClass = position.pnl >= 0 ? 'text-success-bright' : 'text-danger-bright';
                
                // Заполнение ячеек
                row.innerHTML = `
                    <td>${position.id}</td>
                    <td>${position.symbol}</td>
                    <td class="${position.type === 'LONG' ? 'text-success-bright' : 'text-danger-bright'}">${position.type}</td>
                    <td>${position.size.toFixed(4)}</td>
                    <td>$${position.entry_price.toFixed(2)}</td>
                    <td>$${position.current_price.toFixed(2)}</td>
                    <td class="${pnlClass}">${pnlSign}$${Math.abs(position.pnl).toFixed(2)} (${pnlSign}${position.pnl_percent.toFixed(2)}%)</td>
                `;
                
                positionsTableBody.appendChild(row);
            });
        } else {
            // Если нет позиций, добавляем сообщение
            const row = document.createElement('tr');
            row.innerHTML = '<td colspan="7" class="text-center">Нет активных позиций</td>';
            positionsTableBody.appendChild(row);
        }
    }
}

// Обновление сигналов
function updateSignals(signals) {
    const signalsTableBody = document.querySelector('#signals-table tbody');
    
    if (signalsTableBody) {
        // Очистка таблицы
        signalsTableBody.innerHTML = '';
        
        if (signals && signals.length > 0) {
            // Добавление сигналов
            signals.forEach(signal => {
                const row = document.createElement('tr');
                
                // Определение класса для сигнала
                let signalClass = 'signal-neutral';
                if (signal.signal === 'BUY') {
                    signalClass = 'signal-buy';
                } else if (signal.signal === 'SELL') {
                    signalClass = 'signal-sell';
                }
                
                // Заполнение ячеек
                row.innerHTML = `
                    <td>${signal.indicator}</td>
                    <td>${signal.value}</td>
                    <td class="${signalClass}">${signal.signal}</td>
                `;
                
                signalsTableBody.appendChild(row);
            });
        } else {
            // Если нет сигналов, добавляем сообщение
            const row = document.createElement('tr');
            row.innerHTML = '<td colspan="3" class="text-center">Нет сигналов</td>';
            signalsTableBody.appendChild(row);
        }
    }
}

// Обновление рекомендации
function updateRecommendation(recommendation, color) {
    const recommendationElement = document.getElementById('recommendation');
    
    if (recommendationElement) {
        recommendationElement.textContent = recommendation || 'Ожидание сигналов...';
        
        // Удаление всех классов цвета
        recommendationElement.classList.remove('bg-success-dark', 'bg-danger-dark', 'bg-warning-dark');
        
        // Установка нового класса цвета
        if (color === 'success') {
            recommendationElement.classList.add('bg-success-dark');
        } else if (color === 'danger') {
            recommendationElement.classList.add('bg-danger-dark');
        } else {
            recommendationElement.classList.add('bg-warning-dark');
        }
    }
}

// Очистка ресурсов при выгрузке страницы
window.addEventListener('beforeunload', function() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
}); 