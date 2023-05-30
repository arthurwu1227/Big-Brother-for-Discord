#import needed packages
import discord
from discord import Intents
import pickle

#open our trained machine learning models
with open('NBclassifier.pickle', 'rb') as f:
    classifier = pickle.load(f)
with open('vectorizer.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

#log in to discord bot
client = discord.Client(intents = Intents.default())
@client.event
async def on_ready():
    print("We have logged in as {0.user}".format(client))

#on every message, offer these following commands
@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    #offer the .help command
    if message.content.startswith('.help'):
        await message.channel.send('Hello, I am Big Brother. I will prevent crimes before they happen. I will protect the community. You can trust big brother.')

    #use the naive bayes algorithm to predict whether someone's message was naughty or not
    vectorized_message = vectorizer.transform([message.content])
    is_naughty = classifier.predict(vectorized_message)
    if is_naughty == 1:
        await message.delete()
        await message.channel.send(f"{message.author.mention} has been caught red-handed for disobeying the mods!")

#run the client. This token is not in use.
token = 'MTExMjg4OTk5MzI3NTM4MzkxOA.GP2a6h.PultWaQCZpVfqM3dgL4pwQY6ypysvGlKVlzixo'
client.run(token)