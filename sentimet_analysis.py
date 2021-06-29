import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from tkinter import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import random

def analyse():

    class sentiment:

        NEGATIVE='NEGATIVE'
        NEUTRAL='NEUTRAL'
        POSITIVE='POSITIVE'

    class Review:

        def __init__(self,text,score):
            self.text=text
            self.score=score
            self.sentiment=self.get_sentiment()
        
        def get_sentiment(self):
            if self.score<=2:
                return sentiment.NEGATIVE
            elif self.score==3:
                return sentiment.NEUTRAL
            else:
                return sentiment.POSITIVE
            
    class ReviewContainer:

        def __init__(self, reviews):
            self.reviews = reviews

        def get_text(self):
            return [x.text for x in self.reviews]
        
        def get_y(self):
            return [x.sentiment for x in self.reviews]

        def evenly_distribute(self):
            negative = list(filter(lambda x: x.sentiment == sentiment.NEGATIVE, self.reviews))
            positive = list(filter(lambda x: x.sentiment == sentiment.POSITIVE, self.reviews))
            neutral = list(filter(lambda x: x.sentiment == sentiment.NEUTRAL, self.reviews))
            positive_shrunk = positive[:len(negative)]
            neutral_shrunk=neutral[:len(negative)]
            #print('evenlu_distribute function')
            #print(len(positive_shrunk))
            #print(len(neutral_shrunk))
            #print(len(negative))
            self.reviews = negative + positive_shrunk + neutral_shrunk
            random.shuffle(self.reviews)
            print(self.reviews[0])
            
    file_name='/Users/mehrotra/python_program/sklearn-master/data/sentiment/Books_small_10000.json' 
    reviews=[]
    f=pd.read_json(file_name,lines=True)
    f.reviewText
    #print(f)
    for i in range(10000):
        reviews.append(Review(f.reviewText[i],f.overall[i]))

    #print(reviews[0].score)
    #print(reviews[2345].sentiment)
    #print(reviews[0].text)


    training,test=train_test_split(reviews,test_size=0.33)
    #print(len(training))
    #print(training[3057].sentiment)
    #print(training[3057].text)
    #print(training[3057].score)

    train_container=ReviewContainer(training)
    test_container=ReviewContainer(test)

    train_container.evenly_distribute()
    train_x=train_container.get_text()
    train_y=train_container.get_y()
    test_x=test_container.get_text()
    test_y=test_container.get_y()

    ##bags of words #convert all text into int

    vectorizer=TfidfVectorizer() 
    ##tfidfvectorizer gives more value to less appearing words rather than frequently appearing words such as this i etc that happens in count vectorizer
    
    train_x_vectors=vectorizer.fit_transform(train_x)
    test_x_vectors=vectorizer.transform(test_x)
    #print(train_x[0])
    #print(train_x_vectors[0])
    #print(train_x_vectors[0].toarray()) 
    #print(test_x[0]) 
    #print(test_x_vectors[0])
    #print(test_x_vectors[0].toarray()) 

    model=LogisticRegression()
    model.fit(train_x_vectors,train_y)
    print(model.predict(test_x_vectors))
    print(model.score(test_x_vectors,test_y))

    ##gui
    eg=e.get()
    eg2=list(eg.split("++++"))
    print(eg)
    #print(eg2)
    test_set_1=eg2
    new_test_1=vectorizer.transform(test_set_1)
    print(model.predict(new_test_1))
    str1=" "
    print(str1.join(model.predict(new_test_1)))
    e2.delete(0,END)
    e2.insert(0,str1.join(model.predict(new_test_1)))
    ##gui##

    print(f1_score(test_y,model.predict(test_x_vectors),average=None,labels=[sentiment.POSITIVE,sentiment.NEUTRAL,sentiment.NEGATIVE]))

    #print(train_y.count(sentiment.POSITIVE))
    #print(train_y.count(sentiment.NEGATIVE))
    #print(train_y.count(sentiment.NEUTRAL))

    #print(test_y.count(sentiment.POSITIVE))
    #print(test_y.count(sentiment.NEGATIVE))
    #print(test_y.count(sentiment.NEUTRAL))

    #test_set=['that was great ','it was so bad i cant tell','bad dont buy it','bad bad bad','it was ok']
    #new_test=vectorizer.transform(test_set)
    #print(model.predict(new_test))

    y_pred = model.predict(test_x_vectors)

    labels = [sentiment.POSITIVE, sentiment.NEUTRAL, sentiment.NEGATIVE]

    cm = confusion_matrix(test_y, y_pred, labels=labels)
    print(cm)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print(df_cm)
    sn.heatmap(df_cm, annot=True, fmt='d')
    plt.show()

    #print(cross_val_score(LogisticRegression(),train_x_vectors,train_y))
    #print(cross_val_score(DecisionTreeClassifier(),train_x_vectors,train_y))
    #print(cross_val_score(SVC(),train_x_vectors,train_y))
    #print(cross_val_score(RandomForestClassifier(),train_x_vectors,train_y))

root=Tk()
root.title('Sentiment Analysis')
large_font = ('Verdana',20)

def clear():
    e.delete(0,END)
    e2.delete(0,END)

Label(root,text='Enter string for sentiment analysis',anchor='center').grid(row=0,column=0,sticky='w',columnspan=3)
e=Entry(root,width=40,borderwidth=5,font=large_font)
e.grid(row=1,column=0,columnspan=3,padx=30,pady=10)
Label(root,text='Sentiment is').grid(row=3,column=0,sticky='w',columnspan=3)
e2=Entry(root,width=40,borderwidth=5,font=large_font)
e2.grid(row=4,column=0,columnspan=3,padx=30,pady=10)
Button(root,text='Analyse',command=analyse,padx=10,pady=10,font=("Courier", 20),anchor='center').grid(row=5,column=0,padx=10,pady=10)
Button(root,text='Clear',command=clear,padx=10,pady=10,font=("Courier", 20),anchor='center').grid(row=5,column=2,padx=10,pady=10)
root.mainloop()