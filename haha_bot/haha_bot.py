import json
import nest_asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.types import InputFile
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext
from aiogram.utils import executor
from aiogram.utils.markdown import link
import os
import random
import openai
import PyPDF2
import pandas as pd
import sqlite3
import tempfile
from bs4 import BeautifulSoup
import datetime
import logging

from initial_recommendation import initial_recommendation
from predict import predictor
from db import insert_session, insert_user, get_session_count, get_last_session_info, get_user_info
from parse_resume import extract_text_from_pdf, extract_resume_info, extract_key_skills

import os
api_key = os.getenv('API_KEY')
TOKEN = os.getenv('TOKEN')
DATA_PTH =  os.getenv('DATA_PTH')


logging.basicConfig(level=logging.INFO)

nest_asyncio.apply()

kb1 = [
    [types.KeyboardButton(text="Авторизация"),
     types.KeyboardButton(text='Восстановить свой user_id'),
     types.KeyboardButton(text="Регистрация")],
]
autorization_markup = types.ReplyKeyboardMarkup(
    keyboard=kb1,
    resize_keyboard=True,
)

kb2 = [
    [types.KeyboardButton(text="Отправить резюме"),
     types.KeyboardButton(text="Ручной ввод")],
]
registration_markup = types.ReplyKeyboardMarkup(
    keyboard=kb2,
    resize_keyboard=True,
)

kb3 = [
    [types.KeyboardButton(text="Следующая"),
     types.KeyboardButton(text="Полное описание")],
]
common_markup = types.ReplyKeyboardMarkup(
    keyboard=kb3,
    resize_keyboard=True,
)

kb4 = [
    [types.KeyboardButton(text="Следующая"),
     types.KeyboardButton(text="Добавить в избранное"),
     types.KeyboardButton(text="Отклик")],
]
favourite_markup = types.ReplyKeyboardMarkup(
    keyboard=kb4,
    resize_keyboard=True,
)



bot = Bot(token=TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
connection = sqlite3.connect(DATA_PTH+'/haha.db', check_same_thread=False)
cursor = connection.cursor()
df_vacancies_clean = pd.read_sql_query("SELECT * FROM vacancies_ru", connection)
#df_vacancies= pd.read_sql_query("SELECT * FROM vacancies_ru", connection).reset_index('vacancy_id') #search by loc[]
names_to_region_id = pd.read_sql_query("SELECT * FROM names_to_region_id", connection)
first_recommendation = initial_recommendation(df_vacancies_clean)
recommendation = predictor(df_vacancies_clean)

with open(DATA_PTH+'/data/keys.txt', 'r') as file:
    key_skills_list = file.read().lower().split("\n")

def initial_recommendation_for_user(name : str, compensation_from : int, area_id : str, expireince : str, keySkills : list) -> list[str]:
    return  first_recommendation.recomend(name, compensation_from, area_id , expireince, keySkills)

def recommendation_for_user(name : str, area_id : str, expireince : str, action_list : list, vacancy_list : list, keySkills: list) -> list[str]:
    return  recommendation.recomend(name, area_id,expireince, action_list, vacancy_list, keySkills)

user_state = {}

class AutorizationStates(StatesGroup):
    waiting_for_number = State()

class RegistrationStates(StatesGroup):
    waiting_for_resume = State()
    waiting_for_region = State()
    waiting_for_job_title = State()
    waiting_for_salary = State()
    waiting_for_experience = State()
    waiting_for_key_skills = State()

@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.answer('Здравствуйте! Пожалуйста, введите свой номер или зарегистрируйтесь!', reply_markup=autorization_markup)

@dp.message_handler(lambda message: message.text == 'Авторизация')
async def handle_authorization(message: types.Message):
    await message.answer('Пожалуйста, введите свой номер:', reply_markup=autorization_markup)

@dp.message_handler(lambda message: message.text == 'Восстановить свой user_id')
async def handle_authorization(message: types.Message):
    await message.answer(f'Пожалуйста, ваш user_id:{message.from_user.id}. Теперь можете авторизоваться!', reply_markup=autorization_markup)

@dp.message_handler(lambda message: message.text.isdigit())
async def handle_number(message: types.Message):
    user_id = message.from_user.id
    number = str(message.from_user.id)
    if not message.text.isdigit():
        await message.answer('Пожалуйста, введите корректный номер, состоящий только из цифр:')
        return
    try:
        df_users = pd.read_sql_query("SELECT * FROM users", connection)
        user_row = df_users.loc[df_users['user_id'] == number]
    except: 
        print('упс, не нашли вас в базе')
    
    user_state[user_id] = {
        'number': number,
        'index': 0,
        'link' : '',
        'vacancies': [],
        'action_type' : [],
        'prev_action' : '',
        'region' : user_row['area_regionId'].values[0],
        'name' : user_row['name'].values[0],
        'salary' : int(user_row['compensation_from'].values[0]),
        'experience' : user_row['experience'].values[0],
        'keySkills' : eval(user_row['keySkills'].values[0])
    }
    await send_next_vacancy(user_id)

@dp.message_handler(lambda message: message.text == 'Регистрация')
async def handle_registration(message: types.Message):
    await message.answer('Пожалуйста, выберите способ регистрации:', reply_markup=registration_markup)

@dp.message_handler(lambda message: message.text == 'Отправить резюме')
async def handle_resume_option(message: types.Message):
    await message.answer('Пожалуйста, пришлите ваше резюме в формате PDF.')
    await RegistrationStates.waiting_for_resume.set()

@dp.message_handler(lambda message: message.text == 'Ручной ввод')
async def handle_manual_input_option(message: types.Message):
    await message.answer('Введите регион:')
    await RegistrationStates.waiting_for_region.set()

@dp.message_handler(lambda message: message.text == 'Следующая')
async def handle_next_vacancy(message: types.Message):
    user_id = message.from_user.id
    state_data = user_state[user_id]
    if state_data['prev_action'] == 'Следующая':
        state_data['vacancies'].pop(-1)
    state_data['index'] += 1
    state_data['prev_action'] = 'Следующая'
    await send_next_vacancy(user_id)

@dp.message_handler(lambda message: message.text == 'Полное описание')
async def handle_full_description(message: types.Message):
    user_id = message.from_user.id
    state_data = user_state[user_id]
    current_vacancy = state_data['vacancies'][-1]
    state_data['action_type'].append(2)
    state_data['prev_action'] = 'Полное описание'
    desc = BeautifulSoup(str(df_vacancies_clean.query('vacancy_id == @current_vacancy')['description'].values[0])).get_text()
    await message.answer(f'Описание вакансии:\n{desc}')
    await message.answer('Выберите действие:', reply_markup=favourite_markup)

@dp.message_handler(lambda message: message.text == 'Добавить в избранное')
async def handle_add_to_favorites(message: types.Message):
    user_id = message.from_user.id
    state_data = user_state[user_id]
    current_vacancy = state_data['vacancies'][-1]
    state_data['vacancies'].append(current_vacancy)
    state_data['action_type'].append(3)
    state_data['prev_action'] = 'Добавить в избранное'
    vacancy_name = str(df_vacancies_clean.query('vacancy_id == @current_vacancy')['name'].values[0])
    await message.answer(f'Вакансия {vacancy_name} добавлена в избранное.', reply_markup=favourite_markup)

@dp.message_handler(lambda message: message.text == 'Отклик')
async def handle_response(message: types.Message):
    user_id = message.from_user.id
    state_data = user_state[user_id]
    state_data['action_type'].append(1)
    state_data['prev_action'] = 'Отклик'

    await message.answer(f"[Ссылка на вакансию]({state_data['link']})", parse_mode="MarkdownV2", reply_markup=favourite_markup)

@dp.message_handler(state=RegistrationStates.waiting_for_resume, content_types=['document'])
async def handle_resume(message: types.Message, state: FSMContext):
    if message.document.mime_type == 'application/pdf':
        file_info = await bot.get_file(message.document.file_id)

        file_path = file_info.file_path
        downloaded_file = await bot.download_file(file_path)
        src = os.path.join(os.path.expanduser('~'), 'Downloads', message.document.file_name)
        with open(src, 'wb') as f:
            f.write(downloaded_file.getvalue())

        try:
            resume_text = extract_text_from_pdf(src)
            
            if resume_text:
                result = extract_resume_info(resume_text, api_key)
                number = str(message.from_user.id) #change to random
                await state.update_data(region=result.split(";")[0])
                await state.update_data(job_title=result.split(";")[1])
                await state.update_data(experience=result.split(";")[2])
                await state.update_data(key_skills=extract_key_skills(resume_text))
                await message.answer(f'Ваши данные сохранены.\nУкажите ожидаемую зарплату')
                await RegistrationStates.waiting_for_salary.set()
            else:
                await message.answer("Не удалось извлечь текст из PDF файла.")
        except Exception as e:
            await message.answer(f"Произошла ошибка при извлечении текста: {str(e)}")
        #finally:
            #os.remove(file_path)
    else:
        await message.answer('Неправильный формат файла.\nПожалуйста, проверьте, что файл имеет расширение PDF или отправьте другой')


@dp.message_handler(state=RegistrationStates.waiting_for_region)
async def handle_region(message: types.Message, state: FSMContext): 
    await state.update_data(region=message.text)
    await message.answer('Введите название последней профессии:')
    await RegistrationStates.waiting_for_job_title.set()

@dp.message_handler(state=RegistrationStates.waiting_for_job_title)
async def handle_job_title(message: types.Message, state: FSMContext):
    await state.update_data(job_title=message.text)
    await message.answer('Введите опыт работы:')
    await RegistrationStates.waiting_for_experience.set()

@dp.message_handler(state=RegistrationStates.waiting_for_experience)
async def handle_experience(message: types.Message, state: FSMContext):
    if not message.text.isdigit():
        await message.answer('Пожалуйста, введите корректный опыт работы, состоящий только из цифр:')
        return
    await state.update_data(experience=message.text)
    await message.answer('Введите укажите ожидаемую зарплату:')
    await RegistrationStates.waiting_for_salary.set()

@dp.message_handler(state=RegistrationStates.waiting_for_salary)
async def handle_salary(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    if not message.text.isdigit():
        await message.answer('Пожалуйста, введите корректное значение зарплаты, состоящее только из цифр:')
        return
    user_data.update(salary=message.text)
    user_data.update(key_skills='[]')
    user_id = message.from_user.id

    with open('user.json', 'a') as f:
        json.dump({user_id: user_data}, f)
        f.write('\n')
    number = str(message.from_user.id) #random.randint(100, 1000)
    region_of_user = str(user_data['region']).replace(' ', '')
    try:
        region = names_to_region_id.query('region_name == @region_of_user' )['area_regionId'].values[0]
    except:
        region = 113 # Россия
    exp = " "
    if int(user_data['experience']) < 1:
        exp = "noExperience"
    elif int(user_data['experience']) < 3:
        exp = "between1And3"
    elif int(user_data['experience']) < 6:
        exp = "between3And6"
    else:
        exp = 'moreThan6'
    
    insert_user(connection, cursor, str(number), str(exp), str(region), int(user_data['salary'])+0.0, str(user_data['job_title']), str(user_data['key_skills']))
    await message.answer(f'Ваши данные сохранены.\n Вот ваш айдишник:{number}. \n Вам необходимо авторизоваться: для этого еще раз отправьте "/start"!')
    await state.finish()
    

async def send_next_vacancy(user_id):
    state_data = user_state.get(user_id)
    if not state_data:
        await bot.send_message(user_id, 'Ваши данные не найдены.')
        return
    
    # from predict_function
    
    if get_session_count(connection, cursor, state_data['number']) == 0:
        try:
            vacancies_list = list(set(initial_recommendation_for_user(state_data['name'], state_data['salary'], state_data['region'] , state_data['experience'], state_data['keySkills'])))
        except Exception as e :
            print(e)
    else:
        try:
            user_info = get_user_info(connection, cursor, state_data['number'])
            last_session_info = get_last_session_info(connection, cursor, state_data['number'])
            vacancies_list = list(set(recommendation_for_user(user_info['name'], user_info['area_regionId'], user_info['experience'], eval(last_session_info['action_type']), eval(last_session_info['vacancy_id']), eval(user_info['keySkills']))))
        except Exception as e :
            print(e)

        
    index = state_data['index']

    if index >= len(vacancies_list):
        insert_session(connection, cursor, str(state_data['number']), str(int(state_data['number']) +random.randint(130, 481938402)), str(state_data['vacancies'][:(len(state_data['action_type']))]), str(state_data['action_type']), str(datetime.datetime.now()))
        await bot.send_message(user_id, 'Пока что это все!\nЕсли хотите получить новую порцию рекомендаций необходимо еще раз отправить "/start"')
        del user_state[user_id]
        return

    vacancy_number = vacancies_list[index]
    vacancy_name = str(df_vacancies_clean.query('vacancy_id == @vacancy_number')['name'].values[0])
    salary = (df_vacancies_clean.query('vacancy_id == @vacancy_number')['compensation_from'].values[0]) or '-'
    state_data['vacancies'].append(vacancy_number)
    state_data['prev_action'] = ''
    state_data['link'] = str(df_vacancies_clean.query('vacancy_id == @vacancy_number')['alternate_url'].values[0]) # link('Ссылка на вакансию', state_data['link'])
    await bot.send_message(user_id, f'Вакансия: {vacancy_name}\nЗарплата: {salary}', reply_markup=common_markup)


if __name__ == "__main__":
    dp.middleware.setup(LoggingMiddleware())
    executor.start_polling(dp, skip_updates=True)
