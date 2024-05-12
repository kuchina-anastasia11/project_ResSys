#!/usr/bin/env python
# coding: utf-8

# In[29]:


get_ipython().system('pip install aiogram')


# In[30]:


import pandas as pd
from bs4 import BeautifulSoup
import json
import gzip
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command


# In[31]:


data = []
i = 0

with gzip.open("/Users/arkuchina/Downloads/names_salary_preparing.json.gz", 'rt') as f:
  if i < 110000:
    for line in f:
        i+=1
        item = json.loads(line.strip())
        data.append(item)


# In[32]:


df = pd.DataFrame(data)


# In[33]:


df_predict = pd.read_json('/Users/arkuchina/Desktop/haha.ru/response_like_skills_preds_knn.json', orient='index')


# In[141]:





# In[111]:


kb1 = [
        [
            types.KeyboardButton(text="Авторизация"),
            types.KeyboardButton(text="Регистрация")
        ],
    ]
autorization_markup = types.ReplyKeyboardMarkup(
    keyboard=kb1,
    resize_keyboard=True,
)


kb2 = [
        [
            types.KeyboardButton(text="Отправить резюме"),
            types.KeyboardButton(text="Ручной ввод")
        ],
    ]
registration_markup = types.ReplyKeyboardMarkup(
    keyboard=kb2,
    resize_keyboard=True,
)


kb3 = [
        [
            types.KeyboardButton(text="Следующая"),
            types.KeyboardButton(text="Полное описание")
        ],
    ]
common_markup = types.ReplyKeyboardMarkup(
    keyboard=kb3,
    resize_keyboard=True,
)


kb4 = [
        [
            types.KeyboardButton(text="Следующая"),
            types.KeyboardButton(text="Добавить в избранное"),
            types.KeyboardButton(text="Отклик")
        ],
    ]
favourite_markup = types.ReplyKeyboardMarkup(
    keyboard=kb4,
    resize_keyboard=True,
)




TOKEN = '6781213285:AAH5n-vasxmUxSmdyZdCZOTaeRXOVWM3acM'
bot = Bot(token=TOKEN)
dp = Dispatcher()

user_state = {}

@dp.message(CommandStart())
async def start(message: types.Message):
    global cur_state, markup, input_ids, cnt
    
    await message.answer('Здравствуйте! Пожалуйста, введите свой номер или зарегистрируйтесь!', reply_markup=autorization_markup)

@dp.message()
async def message_reply(message: types.Message):
    global cur_state, markup, input_ids, cnt, user_state
    user_id = message.from_user.id
    if message.text == 'Авторизация':
        
        cur_state = "auto"
        await message.answer('Пожалуйста, введите свой номер:', reply_markup=autorization_markup)

       
    elif cur_state == "auto":
        number = message.text
        if number not in data.keys():
            await message.answer('Номер не найден. Попробуйте еще раз.')
            return

        user_state[user_id] = {
            'number': number,
            'index': 0,
            'favorites': []
        }
        cur_state = 'show_vacancies'
        await send_next_vacancy(user_id)
        
    elif message.text == 'Регистрация':
        pass
    
    elif message.text == "Следующая":
        state = user_state[user_id]
        state['index'] += 1
        await send_next_vacancy(user_id)
        
    elif message.text == "Полное описание":
        state = user_state[user_id]
        current_vacancy = data[state['number']][state['index']]
        desc = BeautifulSoup(str(df.query('vacancy_id == @current_vacancy')['description'].values[0])).get_text()
        await message.answer(f'Описание вакансии:\n{desc}')
        await message.answer('Выберите действие:', reply_markup=favourite_markup)

    elif message.text == "Добавить в избранное":
        state = user_state[user_id]
        current_vacancy = str(data[state['number']][state['index'] - 1])
        state['favorites'].append(current_vacancy)

        vacancy_name = str(df.query('vacancy_id == @current_vacancy')['name'].values[0])

        await message.answer(f'Вакансия {vacancy_name} добавлена в избранное.', reply_markup=favourite_markup)

    elif message.text == "Отклик":
        await message.answer('Спасибо за отклик! Надеемся, работодателя с Вами свяжется!', reply_markup=favourite_markup)
        
        
async def registration(user_id):
     await message.answer('Пожалуйста, отправьте свое резюме в формате pdf-файла или введите данные о своей карьере вручную!', reply_markup = registration_markup)
        
    
    
async def send_next_vacancy(user_id):
    state = user_state[user_id]
    vacancies_list = data[state['number']]
    index = state['index']

    if index >= len(vacancies_list):
        await bot.send_message(user_id, 'На сегодня все, покаааааааа')
        del user_state[user_id]
        return

    vacancy_number = vacancies_list[index]
    vacancy_name = str(df.query('vacancy_id == @vacancy_number')['name'].values[0])
    salary = (df.query('vacancy_id == @vacancy_number')['compensation_from'].values[0]) or '-'
    await bot.send_message(user_id, f'Вакансия: {vacancy_name}\nЗарплата: {salary}', reply_markup=common_markup)

# Запуск бота
async def global_main():
    
    await dp.start_polling(bot)


if __name__ == "__main__":
    await global_main()


# In[ ]:




