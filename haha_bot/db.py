import sqlite3
import csv
import pandas


def create_connection(db_file):
    connection = None
    try:
        connection = sqlite3.connect(db_file)
        return connection
    except sqlite3.Error as e:
        print(e)
    return connection

# Функция для создания таблицы
def create_table(connection, create_table_sql):
        cur = connection.cursor()
        cur.execute(create_table_sql)


# Функция для обновления sessions(логи добавятся позже)

def insert_session(connection, cursor, user_id: str, session_id : str, vacancy_list : str, actions: str, action_dt : str):
        cursor.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?)", (user_id, session_id, vacancy_list, actions, action_dt))
        connection.commit()

def insert_user(connection, cursor, user_id: str, experience: str, id_region: str, compensation_from : int, name : str, key_skills: str):
    cursor.execute("INSERT INTO users(user_id, experience, area_regionId, compensation_from, name, keySkills) VALUES(?, ?, ?, ?, ?, ?);", (user_id, experience, id_region, compensation_from, name, key_skills))
    connection.commit()

def get_session_count(connection, cursor, user_id: str) -> int:
    cursor.execute("SELECT user_id FROM sessions WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    if result:
        return result[0]
    return 0

def get_user_info(connection,cursor, user_id: str) -> dict:
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    user_row = cursor.fetchone()
    if user_row:
        columns = [desc[0] for desc in cursor.description]
        user_info = dict(zip(columns, user_row))
        return user_info
    else:
        return None
    
def get_last_session_info(connection, cursor,user_id: str) -> dict:
    cursor = connection.cursor()
    cursor.execute("""
        SELECT * FROM sessions
        WHERE user_id = ?
        ORDER BY action_dt DESC
        LIMIT 1
    """, (user_id,))
    session_row = cursor.fetchone()
    if session_row:
        columns = [desc[0] for desc in cursor.description]
        session_info = dict(zip(columns, session_row))
        return session_info
    else:
        return None


def extract_text_from_pdf(file_path: str) -> str:
    print('hi 24')
    text = ""
    print('25')
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    print(text)
    return text

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
    path_to_database = r"/home/jovyan/shares/SR004.nfs2/amaksimova/exp/final/haha.db"
    connection = create_connection(path_to_database)
    if connection is not None:
        create_vacancies_table_sql = """
        CREATE TABLE IF NOT EXISTS vacancies_ru (
            vacancy_id TEXT,
            name TEXT,
            company_id TEXT, 
            keySkills TEXT, 
            compensation_from TEXT,
            compensation_to TEXT,
            compensation_currencyCode TEXT,
            area_id TEXT,
            employment TEXT,
            workSchedule TEXT,
            workExperience TEXT, 
            description TEXT,
            published_at TEXT, 
            alternate_url TEXT,
            prof_codes TEXT
        );
        """

        create_sessions_table_sql = """
        CREATE TABLE IF NOT EXISTS sessions (
            user_id TEXT,
            session_id TEXT,
            vacancy_id TEXT,
            action_type TEXT,
            action_dt TEXT
        );
        """
        create_users_table_sql = """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT,
            experience TEXT,
            area_regionId TEXT,
            compensation_from REAL,
            name TEXT,
            keySkills TEXT
        );
        """

        
        create_table(connection, create_vacancies_table_sql)
        create_table(connection, create_users_table_sql)
        create_table(connection, create_sessions_table_sql)

        pandas.read_csv('../final/data/vacancy_hh_dataset_all_new.csv').to_sql("vacancies_ru", connection, if_exists='replace', index=False)
        # add mapping id_region
        pandas.read_csv('../final/data/names_to_region_id.csv').to_sql("names_to_region_id", connection, if_exists='replace', index=False)
        connection.commit()
        connection.close()
    else:
        print("Невозможно создать соединение с базой данных.")
