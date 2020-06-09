"""
Simple AI chat-bot

We can give the code any article of our choice and it will try to answer our questions based on the information present
in the article. The code will parse the article and then try to give a response to our queries based on the parsed data.
The effectiveness of the chat-bot depends on the amount of information present in the article.

"""

# getting the relevant libraries and packages
import numpy as np
from newspaper import Article
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)

# getting the data or in our case the Coronavirus article. Here we can give the link to any article of our choice.
article = Article('https://www.space.com/11506-space-weather-sunspots-solar-flares-coronal-mass-ejections.html')
article.download()
article.parse()
article.nlp()
corpus = article.text

# tokenization
sentence_list = nltk.sent_tokenize(corpus) # this gives a list of sentences
print(sentence_list)

# function to respond to user greeting
def greeting(text):
    text = text.lower()

    bot_greeting = ['hello', 'namaste', 'howdy', 'hola', 'hey', 'hi', 'Merhaba']

    user_greeting = ['hey there', 'wassup', 'hello', 'namaste', 'howdy', 'hola', 'hey', 'hi', 'Merhaba']

    for word in text.split():
        if word in user_greeting:
            return random.choice(bot_greeting)

# function to sort the index
def sort_index(list_var):
    length = len(list_var)
    list_index = list(range(length))

    x = list_var
    for i in range(length):
        for j in range(length):
            if x[list_index[i]] > x[list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    return list_index


# code to create the ChatBot's response
def bot_response(user_query):
    user_query = user_query.lower()
    sentence_list.append(user_query)
    bot_response = ''
    cm = CountVectorizer().fit_transform(sentence_list)
    similarity_scores = cosine_similarity(cm[-1], cm)
    similarity_scores_list = similarity_scores.flatten()
    index = sort_index(similarity_scores_list)
    """
    index will contain the indices sorted for the highest values 
    in the similarity scores. That is finding the highest value
    and placing it at the lowest index
    """
    index = index[1:] # not including the sentence itself

    response_flag = 0
    j = 0
    for i in range(len(index)):
        if similarity_scores_list[index[i]] > 0.0:
            bot_response = bot_response + " " + sentence_list[index[i]]
            response_flag = 1
            j += 1
        if j > 2:
            break

    if response_flag == 0:
        bot_response = bot_response + " " + "Sorry, I could not find any relevant information or I do not understand"

    sentence_list.remove(user_query)

    return bot_response


print('Hi, I am AI Health Bot and I am here to answer your queries. If you want to exit then type "bye"')

while True:
    user_query = input('Please type in your query: ')
    if user_query.lower() == 'bye' or user_query == '':
        print('C u later, have a nice day!')
        break
    else:
        if greeting(user_query) != None:
            print('AI Health Bot: ' + ' ' + greeting(user_query))

        else:
            print('')
            print('AI Health Bot: ' + ' ' + bot_response((user_query)))
