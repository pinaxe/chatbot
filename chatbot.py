import sys
import numpy
import nltk
import nltk.data
from nltk.tag import StanfordNERTagger

import collections
import binaryqs
import json
from reduction import *
reduction = Reduction()
reduced_text = ""
count = 0
cnt = 0
summarization_factor = 0.5    #In progress
sent_list = []

snow = nltk.stem.SnowballStemmer("english")

st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz','stanford-ner.jar', encoding='utf-8')

sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")
 
binarywords = ["can", "could", "would", "is", "does", "has", "was", "were", "had", "have", "did", "are", "will"]
frequentwords = ["the", "a", "an", "is", "are", "were", "."]
questionwords = ["who", "what", "where", "when", "why", "how", "whose", "which", "whom"]

def processquestion(qwords):
    
    questionword = ""
    qidx = -1

    for (idx, word) in enumerate(qwords):
        if word.lower() in questionwords:
            questionword = word.lower()
            qidx = idx
            break
        elif word.lower() in binarywords:
            return ("YESNO", qwords)

    if qidx < 0:
        return ("MISC", qwords)

    if qidx > len(qwords) - 3:
        target = qwords[:qidx]
    else:
        target = qwords[qidx+1:]
    type = "MISC"

    # Question type
    if questionword in ["who", "whose", "whom"]:
        type = "PERSON"
    elif questionword == "where":
        type = "PLACE"
    elif questionword == "when":
        type = "TIME"
    elif questionword == "how":
        if target[0] in ["few", "little", "much", "many"]:
            type = "QUANTITY"
            target = target[1:]
        elif target[0] in ["young", "old", "long"]:
            type = "TIME"
            target = target[1:]

    if questionword == "which":
        target = target[1:]
    if target[0] in binarywords:
        target = target[1:]
    
    return (type, target)

articlesfile = sys.argv[1]
questionsfile = sys.argv[2]

article = open(articlesfile, 'r')

article = article.read()
article = sent_detector.tokenize(article)

questions = open(questionsfile, 'r').read()
questions = questions.splitlines()

# Loop all questions
while True:
    print("Enter Question : ")
    question = input()
    if question == "bye" or question == "Bye":
        exit()
    
    done = False

    # Tokenize question
    qwords = nltk.word_tokenize(question.replace('?', ''))
    questionPOS = nltk.pos_tag(qwords)

    # Process question
    (type, target) = processquestion(qwords)

    # Answer binary questions
    if type == "YESNO":
        binaryqs.answeryesno(article, qwords)
        continue

    # Get sentence keywords
    searchwords = set(target).difference(frequentwords)
    dict = collections.Counter()
        
    # Find relevant sentences
    for (i, sent) in enumerate(article):
        sentwords = nltk.word_tokenize(sent)
        wordmatches = set(filter(set(searchwords).__contains__, sentwords))
        dict[sent] = len(wordmatches)
    
    max_match = max(dict.values())
    for i in dict:
        if max_match == dict[i]:
            sent_list.append(i)
    if len(sent_list) == 1:
        tokens = nltk.word_tokenize(sent_list[0])
        target_stem = snow.stem(target[-1])
        for word in tokens:
            stemmed = snow.stem(word)
            if stemmed == target_stem:
                endidx = sent_list[0].index(word)
            else:
                endidx = sent_list[0].index(target[-1])                

        answer = sent_list[0]
        done = True

    else:
        # Choose from 10 relevant sentences
        for (sentence, matches) in dict.most_common(10):
            tokens = nltk.word_tokenize(sentence)
            parse = st.tag(tokens)
            sentencePOS = nltk.pos_tag(nltk.word_tokenize(sentence))

            # Attempt to find matching substrings
            searchstring = ' '.join(target)
            for each_target in target:
                if each_target in sentence:
                    cnt+=1
            if cnt == len(target):
                tokens = nltk.word_tokenize(sent_list[0])
                target_stem = snow.stem(target[-1])
                for word in tokens:
                    stemmed = snow.stem(word)
                if stemmed == target_stem:
                    endidx = sent_list[0].index(word)
                else:
                    endidx = sent_list[0].index(target[-1])                

                answer = sent_list[0]
                done  = True
            # Check if solution is found
            if done:
                continue

            # Check by question type
            answer = ""
            for worddata in parse:
                if worddata[0] in searchwords:
                    continue
            
                if type == "PERSON":
                    if worddata[1] == "PERSON":
                        answer = answer + " " + worddata[0]
                        done = True
                    elif done:
                        break

                if type == "PLACE":
                    if worddata[1] == "LOCATION":
                        answer = answer + " " + worddata[0]
                        done = True
                    elif done:
                        break

                if type == "QUANTITY":
                    if worddata[1] == "NUMBER":
                        answer = answer + " " + worddata[0]
                        done = True
                    elif done:
                        break

                if type == "TIME":
                    if worddata[1] == "NUMBER":
                        answer = answer + " " + worddata[0]
                        done = True
                    elif done:
                        answer = answer + " " + worddata[0]
                        break
            
    if done:
        print("Answer : ",answer)
        print("------------------------------------------------------------------------------------")
    if not done:
        (answer, matches) = dict.most_common(1)[0]
        #if len(answer) > 250:
        #    print("REDUCE")
        #    reduced_text = reduction.reduce(answer, summarization_factor)
        #    print(reduced_text, len(reduced_text))
        #else:
        print ("Answer : ", answer)
        print("-------------------------------------------------------------------------------------")
