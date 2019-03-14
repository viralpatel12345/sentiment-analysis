from flask import Flask , render_template , request

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

app = Flask(__name__)

@app.route("/")
def hello() :
    return render_template('index.html')



@app.route("/result", methods=['POST','GET'])
def result() :
    if request.method == 'POST':
        data = request.form['value']

        def format_sentence(sent):
            return ({word: True for word in nltk.word_tokenize(sent)})

        pos = []
        with open("./pos.txt") as f:
            for i in f:
                pos.append([format_sentence(i), 'pos'])

        neg = []
        with open("./neg.txt") as f:
            for i in f:
                neg.append([format_sentence(i), 'neg'])

        # next, split labeled data into the training and test data
        training = pos[:int((.8) * len(pos))] + neg[:int((.8) * len(neg))]
        test = pos[int((.8) * len(pos)):] + neg[int((.8) * len(neg)):]

        classifier = NaiveBayesClassifier.train(training)
        # classifier.show_most_informative_features()

        ans = classifier.classify(format_sentence(data))

        return render_template("result.html",name=ans)

        #print(accuracy(classifier, test))


if __name__ == "__main__" :
    app.run()
