import numpy as np
import pickle
from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok
from pyvi import ViTokenizer, ViPosTagger
from underthesea import sent_tokenize
from pyvi import ViTokenizer, ViPosTagger
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 50

app = Flask(__name__)
path_save ='pickle/'

global model
model = keras.models.load_model(path_save + "model.h5")

with open(path_save + 'word2idx.pickle', 'rb') as f:
    word2idx = pickle.load(f)

tags = ['B-LOC', 'I-MISC', 'B-PER', 'I-LOC', 'B-MISC', 'I-PER', 'I-ORG', 'B-ORG', 'O']

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dulieu")
def dulieu():
    return render_template("dulieu.html")


@app.route("/ketqua")
def phantichketqua():
    return render_template("ketqua.html")

@app.route("/phantich/", methods=['POST','GET'])
def phantich():
    paragraph = request.form['query']
    list_sents = sent_tokenize(paragraph)
    text_output = ""
    for sent in list_sents:
        example_token = ViTokenizer.tokenize(sent)
        x_example = []
        for word in example_token.split(" "):
          try:
            x_example.append(word2idx[word])
          except:
            x_example.append(word2idx["UNK"])
        x_example = pad_sequences(maxlen=max_len, sequences=[x_example], padding="post", value=word2idx["PADword"])
        output = model.predict(np.array(x_example))
        output = np.argmax(output, axis=-1)[0]
        s = ""
        for index,w in enumerate(example_token.split(" ")):
          w = w.replace("_", " ")
          if "PER" in tags[output[index]]:
            s += "<a style=\"color:red;\">" + w + "</a>" + " "
          elif "LOC" in tags[output[index]]:
            s += "<a style=\"color:green;\">" + w + "</a>" + " "
          elif "ORG" in tags[output[index]]:
            s += "<a style=\"color:yellow;\">" + w + "</a>" + " "
          elif "MISC" in tags[output[index]]:
            s += "<a style=\"color:blue;\">" + w + "</a>" + " "
          else:
            s += w + " "
        text_output += s.strip() + " "
    text_output = text_output.replace(" , ", ", ").replace(" . ", ". ").replace(" ; ", "; ").strip()
    print(text_output)
    return render_template("result.html", data = [{"label": text_output,"query":paragraph}])


if __name__=="__main__":
    run_with_ngrok(app)
    app.run()

