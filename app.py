from flask import Flask , render_template,flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


app = Flask(__name__)


def init():
    global model,graph
    model = load_model('accenture_test_model')
    graph = tf.compat.v1.get_default_graph()

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")


@app.route('/sentiment_analysis_prediction', methods = ['POST', "GET"])
def sentiment_analysis_prediction():
    if request.method=='POST':
        text = request.form['text']
        #text = "The movie was cool "
        # The animation and the graphics were out of this world.The movie was cool. The animation and the graphics were out of this world.The movie was cool. The animation and the graphics were out of this world.The movie was cool. The animation and the graphics were out of this world.The movie was cool. The animation and the graphics were out of this world.The movie was cool. The animation and the graphics were out of this world.The movie was cool. The animation and the graphics were out of this world.The movie was cool. The animation and the graphics were out of this world. I would recommend this movie."

        # Sentiment = ''
        # max_review_length = 50000
        # word_to_id = imdb.get_word_index()
        # strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        # text = text.lower().replace("<br />", " ")
        # #print("#############################################",text)
        # text=re.sub(strip_special_chars, "", text.lower())
        # print("#############################################",text)

        # words = text.split() #split string into a list
        # x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=20000) else 0 for word in words]]

        # x_test = sequence.pad_sequences(x_test, maxlen=5000) # Should be same which you used for training data
        # print(x_test)
        # #vector = np.array([x_test.flatten()])

        # model = load_model('accenture_test_model')

        predictions = model.predict(np.array([text]))


        print(predictions[[0]])
        if predictions == 0:
            sentiment = 'Negative'

        else:
            sentiment = 'Positive'
            print('Positive')

    return render_template('index.html', text=text, sentiment=sentiment, probability=predictions)

if __name__ == "__main__":
    init()
    app.run(debug=True)
