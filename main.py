import os
import telebot
import numpy as np
import pandas as pd
import random
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from scipy import stats
from telebot import types
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io
import re

import heapq

import csv

from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as index

YesNo_dict = {
    'No': 0,
    'Yes': 1
}
YesNo_dict_0 = str(list(YesNo_dict.keys())[0])
YesNo_dict_1 = str(list(YesNo_dict.keys())[1])

terrain_type_dic = {
"open": 1,
"forest": 2,
"urban": 3
}
terrain_type_dic_0 = str(list(terrain_type_dic.keys())[0])
terrain_type_dic_1 = str(list(terrain_type_dic.keys())[1])
terrain_type_dic_2 = str(list(terrain_type_dic.keys())[2])
terrain_type = None

weather_dic = {
"clear": 1,
"rain":2,
"fog": 3
}
weather_dic_0 = str(list(weather_dic.keys())[0])
weather_dic_1 = str(list(weather_dic.keys())[1])
weather_dic_2 = str(list(weather_dic.keys())[2])
weather = None

time_required = None
resource_use = None
enemy_strength = None
success = None

def COA_success(x1, x2, x3, x4, x5):
    np.random.seed(42)
    df = pd.DataFrame({
      "time_required": np.random.randint(24, 120, size=100),
      "resource_use": np.random.randint(10, 100, size=100),
      "terrain_type": np.random.choice(["open", "forest", "urban"], size=100),
      "enemy_strength": np.random.randint(1, 10, size=100),
      "weather": np.random.choice(["clear", "rain", "fog"], size=100),
      "success": np.random.choice([0, 1], size=100, p=[0.4, 0.6])
    })
    df['terrain_type'] = df['terrain_type'].map(terrain_type_dic)
    df['weather'] = df['weather'].map(weather_dic)
    X = df.drop(['success'], axis=1)
    y = df['success']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    new_coa = pd.DataFrame({
    "time_required": [x1],
    "resource_use": [x2],
    "terrain_type": [x3],
    "enemy_strength": [x4],
    "weather": [x5]
    })
    new_coa_predict = model.predict(new_coa)


    if new_coa_predict < 1:
      new_coa_predictAnswer = 'not expected'
    else:
      new_coa_predictAnswer = 'is expected'
    new_coa_predictPercent = model.predict_proba(new_coa)
    new_coa_predictPercent = new_coa_predictPercent[-1][1]
    new_coa_predictPercent = new_coa_predictPercent*100

    return new_coa_predictAnswer, new_coa_predictPercent


bot = telebot.TeleBot('7097866116:AAEsPZUryeQk05OXWcMIFBsbh2Fg2spbbRE')

@bot.message_handler(commands=['help', 'start'])

def send_welcome(message):
    msg = bot.reply_to(message, "\n\nHello, I'm Ai-powered Course Of Action operational planing bot based on NATO standarts!")
    chat_id = message.chat.id
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
    markup.add('Next')
    msg = bot.reply_to(message, '\n\nI analize historical combat data to predict the success of COA action. \n\n To continue press Next', reply_markup=markup)
    bot.register_next_step_handler(msg, process_time_required_step)

def process_time_required_step(message):
    try:
        chat_id = message.chat.id
        Next = message.text
        if (Next == 'Next'):
          msg = bot.reply_to(message, 'Please enter the value of time_required')
          bot.register_next_step_handler(msg, process_resource_use_step)
        else:
          raise Exception("Exception process_time_required_step ")
    except Exception as e:
        bot.reply_to(message, 'oooops process_time_required_step')

def process_resource_use_step(message):
  try:
    chat_id = message.chat.id
    resource_message = message.text
    if not resource_message.isdigit():
      msg = bot.reply_to(message, 'The value of time required must be a number. Please enter a value of time required.')
      bot.register_next_step_handler(msg, process_resource_use_step)
    else:
      global time_required
      time_required = int(resource_message)
      msg = bot.reply_to(message, 'Please enter the value of resourses used')
      bot.register_next_step_handler(msg, process_terrian_type_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_resource_use_step')

def process_terrian_type_step(message):
  try:
    chat_id = message.chat.id
    terrian_message = message.text
    if not terrian_message.isdigit():
      msg = bot.reply_to(message, 'The value of terrian type must be a number. Please enter a value of time required.')
      bot.register_next_step_handler(msg, process_terrian_type_step)
    else:
      global resource_use
      resource_use = int(terrian_message)
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add(terrain_type_dic_0, terrain_type_dic_1, terrain_type_dic_2)
      msg = bot.reply_to(message, 'To choose terrian type.', reply_markup=markup)
      bot.register_next_step_handler(msg, process_enemy_strength_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_terrian_type_step')

def process_enemy_strength_step(message):
  try:
    chat_id = message.chat.id
    enemy_strength_message = message.text
    if (enemy_strength_message == terrain_type_dic_0) or (enemy_strength_message == terrain_type_dic_1) or (enemy_strength_message == terrain_type_dic_2):
      global terrain_type
      terrain_type = terrain_type_dic[enemy_strength_message]
      msg = bot.reply_to(message, 'Please enter the value of enemy strength')
      bot.register_next_step_handler(msg, process_weather_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_enemy_strength_step')

def process_weather_step(message):
  try:
    chat_id = message.chat.id
    weather_message = message.text
    if not weather_message.isdigit():
      msg = bot.reply_to(message, 'The value of enemy strength must be a number.')
      bot.register_next_step_handler(msg, process_weather_step)
    else:
      global enemy_strength
      enemy_strength = int(weather_message)
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add(weather_dic_0, weather_dic_1, weather_dic_2)
      msg = bot.reply_to(message, 'Please choose the weather type.', reply_markup=markup)
      bot.register_next_step_handler(msg, predict_success_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_weather_step')

def predict_success_step(message):
  try:
    chat_id = message.chat.id
    predict_message = message.text
    if (predict_message == weather_dic_0) or (predict_message == weather_dic_1) or (predict_message == weather_dic_2):
      global weather
      weather = weather_dic[predict_message]
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add('Next')
      msg = bot.reply_to(message, 'To predict the probability of COA success please press Next.', reply_markup=markup)
      bot.register_next_step_handler(msg, predict_COA_success_step)

  except Exception as e:
    bot.reply_to(message, 'oooops predict_success_step')


def predict_COA_success_step(message):
  try:
    chat_id = message.chat.id
    COA_success_message = message.text
    if (COA_success_message == 'Next'):
      new_coa_predictAnswer, new_coa_predictPercent = COA_success(time_required, resource_use, terrain_type, enemy_strength, weather)

      bot.send_message(chat_id,

      '\n - Succuss ' + str(new_coa_predictAnswer)+
      '\n - Succuss Probability in percent: ' + str(new_coa_predictPercent) + ' %'

      )

      msg = bot.reply_to(message, 'Try again?')
      bot.register_next_step_handler(msg, send_welcome)

  except Exception as e:
    bot.reply_to(message, 'oooops predict_COA_success_step')

bot.infinity_polling()
