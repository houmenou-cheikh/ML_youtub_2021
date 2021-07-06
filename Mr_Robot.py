import os
from dotenv import load_dotenv
from discord.ext import commands
load_dotenv(dotenv_path="config")

import nltk

nltk.download('punkt')

from nltk import word_tokenize, sent_tokenize

from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

import pymongo
from pymongo import MongoClient
import pandas as pd

import discord
default_intents = discord.Intents.default()


bot = commands.Bot(command_prefix="!", intents=default_intents)


dat = {"intents": [
        {"tag": "greeting",
         "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Hey","Good day", "Whats up"],
         "responses": ["Hello!", "Good to see you again!", "Hi there, how can I help?"],
         "context_set": ""
        },

        {"tag": "goodbye",
         "patterns": ["cya", "See you later", "Goodbye", "I am Leaving", "Have a Good day","bye"],
         "responses": ["Sad to see you go..", "Talk to you later", "Goodbye!"],
         "context_set": ""
        }
         
   ]
}


with open("intents.json","w") as ff:
    json.dump(dat, ff)

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w.isalnum()]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)
  
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


#import discord

      
class MyClient(discord.Client):
    async def on_ready(self):
        print('Logged in as')
        print(self.user.name)
        print(self.user.id)
        print('------')
    @bot.command(name="del")
    async def delete(ctx, number: int):
        messages = await ctx.channel.history(limit=number + 1).flatten()
        for each_message in messages:
            await each_message.delete()
    async def on_message(self, message):
        # we do not want the bot to reply to itself
        if message.author.id == self.user.id:
            return
        elif message.content.lower().startswith("!del") :
            try:
                number = int(message.content.split()[1])
                messages = await message.channel.history(limit=number + 1).flatten()
                for each_message in messages:
                    await each_message.delete()
                await message.channel.send(message.content.lower().split()[1]+" messages viennent d'être supprimés")
            except IndexError:
                await message.channel.send("vous n'avez pas saisi le nombre de messages à supprimer\n Voici un exemple: !del 3\
		        \n ici 3 est le nombre de messages à supprimer")
            except ValueError:
                await message.channel.send("Erreur de saisie: le nombre de messages à suppimer est incorrect\n Voici un exemple: !del 3\
		        \n ici 3 est le nombre de messages à supprimer")
        else:
            inp = message.content
            result = model.predict([bag_of_words(inp, words)])[0]
            result_index = np.argmax(result)
            tag = labels[result_index]

            if result[result_index] > 0.7:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']

                bot_response = random.choice(responses)
                await message.channel.send(bot_response.format(message))
            else:
                await message.channel.send("I didn't get that. Can you explain or try again.")


bot=MyClient()
bot.run(os.getenv("TOKEN"))
