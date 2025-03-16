module.exports = {
  // Настройки подключения к базе данных
  database: {
    host: "localhost",
    port: 3306,
    username: "username",
    password: "password",
    name: "database_name"
  },
  
  // Настройки API ключей
  apiKeys: {
    exampleService: "your_api_key_here",
    anotherService: "your_api_key_here"
  },
  
  // Настройки сервера
  server: {
    port: 3000,
    host: "localhost",
    environment: "development" // production, development, testing
  },
  
  // Настройки авторизации
  auth: {
    secretKey: "your_secret_key_here",
    tokenExpiration: "24h"
  }
};
