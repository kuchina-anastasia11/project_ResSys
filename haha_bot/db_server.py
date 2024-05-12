import sqlite3
import csv

# Функция для соединения 
def create_connection(file):
    connection = None
    try:
        connection = sqlite3.connect(file)
        return connection
    except sqlite3.Error as e:
        print(e)
    return connection

# Функция для создания таблицы
def create_table(connection, create_table_sql):
    try:
        cur = connection.cursor()
        cur.execute(create_table_sql)
    except sqlite3.Error:
        print(sqlite3.Error)

# Функция для обновления sessions(логи добавятся позже)
def add_session(connection, session_data):
    try:
        cur = connection.cursor()
        cur.execute("INSERT INTO sessions VALUES (?, ?, ?)", session_data)
        connection.commit()
    except sqlite3.Error:
        print(sqlite3.Error)

# Функция для обновления таблицы prediction 
def update_prediction_table(connection, csv_file):
    try:
        cur = connection.cursor()
        cur.execute("DELETE FROM prediction")
        with open(csv_file, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader) 
            for row in reader:
                cur.execute("INSERT INTO prediction VALUES (?, ?)", row)
        connection.commit()
    except sqlite3.Error:
        print(sqlite3.Error)

# Функция для обновления таблицы vacancies 
def update_vacancies_table(connection, csv_file):
    try:
        cur = connection.cursor()
        cur.execute("DELETE FROM vacancies")
        with open(csv_file, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader) 
            for row in reader:
                cur.execute("INSERT INTO vacancies VALUES (?, ?, ?, ?, ?)", row)
        connection.commit()
    except sqlite3.Error:
        print(sqlite3.Error)

if __name__ == '__main__':
    database = "haha_ru.db"
    connection = create_connection(database)
    if connection is not None:
        create_vacancies_table_sql = """
        CREATE TABLE IF NOT EXISTS vacancies_ru (
            vacancy_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            compensation_from REAL,
            compensation_to REAL
        );
        """

        create_prediction_table_sql = """
        CREATE TABLE IF NOT EXISTS prediction (
            user_id INTEGER PRIMARY KEY,
            vacancies TEXT
        );
        """

        create_sessions_table_sql = """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            vacancy_id TEXT,
            action_type TEXT,
            action_dt REAL
        );
        """
        create_table(connection, create_vacancies_table_sql)
        create_table(connection, create_prediction_table_sql)
        create_table(connection, create_sessions_table_sql)

        connection.close()
    else:
        print("Невозможно создать соединение с базой данных.")
