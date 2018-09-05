#!/usr/bin/env python
import sys
import re
import os
import math
import numpy as np
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering

DIR, file_name = os.path.split(os.path.realpath(__file__))

token_docs = ""       # tokenized plant journals
corps_freq = ""       # frequency of each token in the journal.
titles = ""           # titles of each article in plant
stoplist = ""         # common uninteresting words

# doc_vector:
# An array of hashes, each array index indicating a particular
# query's weight "vector". (See more detailed below)

doc_vector = []

# docs_freq_hash
#
# dictionary which holds <token, frequency> pairs where
# docs_freq_hash[token] -> frequency
#   token     = a particular word or tag found in the cacm corpus
#   frequency = the total number of times the token appears in
#               the corpus.

docs_freq_hash = defaultdict(int)

# corp_freq_hash
#
# dictionary which holds <token, frequency> pairs where
# corp_freq_hash[token] -> frequency
#   token     = a particular word or tag found in the corpus
#   frequency = the total number of times the token appears per
#               document-- that is a token is counted only once
#               per document if it is present (even if it appears
#               several times within that document).

corp_freq_hash = defaultdict(int)

# stoplist_hash
#
# common list of uninteresting words which are likely irrelvant
# to any query.
# for given "word" you can do:   `if word in stoplist_hash` to check
# if word is in stop list
#
#   Note: this is an associative array to provide fast lookups
#         of these boring words

stoplist_hash = set()

# titles_vector
#
# vector of the cacm journal titles. Indexed in order of apperance
# within the corpus.

titles_vector = []

sensenum = []

v_profile1 = defaultdict(int)
v_profile2 = defaultdict(int)

sys.stderr.write("INITIALIZING VECTORS ... \n")

##########################################################
##  INIT_FILES
##
##  This function specifies the names and locations of
##  input files used by the program.
##
##  Parameter:  $type   ("stemmed" or "unstemmed")
##
##  If $type == "stemmed", the filenames are initialized
##  to the versions stemmed with the Porter stemmer, while
##  in the default ("unstemmed") case initializes to files
##  containing raw, unstemmed tokens.
##########################################################


def init_files(stemmed, name):
    global token_docs
    global corps_freq
    global stoplist
    global titles

    token_docs = DIR + "/" + name
    corps_freq = DIR + "/" + name
    stoplist = DIR + "/common_words"
    titles = name + ".titles"

    if stemmed == "stemmed":
        token_docs += ".stemmed"
        corps_freq += ".stemmed.hist"
        stoplist   += ".stemmed"
    else:
        token_docs += ".tokenized"
        corps_freq += ".tokenized.hist"

##########################################################
##  INIT_CORP_FREQ
##
##  This function reads in corpus and document frequencies from
##  the provided histogram file for both the document set
##  and the query set. This information will be used in
##  term weighting.
##
##  It also initializes the arrays representing the stoplist,
##  title list and relevance of document given query.
##########################################################


def init_corp_freq():
    global titles_vector
    titles_vector = []
    for line in open(corps_freq, 'r'):
        per_data = line.strip().split()
        if len(per_data) == 3:
            corp_freq, doc_freq, term = line.strip().split()
            docs_freq_hash[term] = int(doc_freq)
            corp_freq_hash[term] = int(corp_freq)

    for line in open(stoplist, 'r'):
        if line:
            stoplist_hash.add(line.strip())

    # push one empty value onto titles_vector
    # so that indices correspond with title numbers.
    # title looks like:
    # titles_vector[3195] = "Gorn    #  Reiteration of ACM Policy Toward Standardization"
    titles_vector.append("")

    for line in open(titles, 'r'):
        if line:
            titles_vector.append(line.strip())

##########################################################
##  INIT_DOC_VECTORS
##
##  This function reads in tokens from the document file.
##  When a .I token is encountered, indicating a document
##  break, a new vector is begun. When individual terms
##  are encountered, they are added to a running sum of
##  term frequencies. To save time and space, it is possible
##  to normalize these term frequencies by inverse document
##  frequency (or whatever other weighting strategy is
##  being used) while the terms are being summed or in
##  a posthoc pass.  The 2D vector array
##
##    doc_vector[ doc_num ][ term ]
##
##  stores these normalized term weights.
##
##  It is possible to weight different regions of the document
##  differently depending on likely importance to the classification.
##  The relative base weighting factors can be set when
##  different segment boundaries are encountered.
##
##  This function is currently set up for simple TF weighting.
##########################################################


def init_doc_vectors(tf="idf", weight="uniform", bag_of_words="none"):
    # Make sure doc_vector is emtpy
    global doc_vector
    doc_vector = []
    global sensenum
    sensenum = []

    doc_num = 0
    # push one empty value onto doc_vector and sensesum so that
    # indices correspond with doc numbers
    doc_vector.append(defaultdict(int))
    sensenum.append(-1)

    doc_line = []
    for word in open(token_docs, 'r'):
        word = word.strip()
        if not word or word == ".I 0":
            continue  # Skip empty line
        if word[:2] == ".I":
            if len(doc_line) != 0:
                get_weights(doc_line, new_doc_vec, weight, bag_of_words)

            doc_line = []
            new_doc_vec = defaultdict(int)
            doc_vector.append(new_doc_vec)
            sensenum.append(int((word.split()[2])))
            doc_num += 1
        elif word not in stoplist_hash and re.search("[a-zA-Z]", word):
            if docs_freq_hash[word] == 0:
                exit("ERROR: Document frequency of zero: " + word + \
                     " (check if token file matches corpus_freq file\n")
            doc_line.append(word)

    if len(doc_line) != 0:
        get_weights(doc_line, new_doc_vec, weight, bag_of_words)

    global total_docs
    total_docs = doc_num

    if tf == "idf":
        for docn in range(1, len(doc_vector)):
            for term in doc_vector[docn]:
                doc_vector[docn][term] *= math.log(float(total_docs) / corp_freq_hash[term])


def main():

    menu = \
    "============================================================\n"\
    "==      Welcome to the 600.466 Vector-based IR Engine       \n"\
    "============================================================\n"\
    "                                                            \n"\
    "OPTIONS:                                                    \n"\
    "  1 = Vector classification model accuracy (Part 1)           \n"\
    "  2 = Variations on the vector model (Part 2)                 \n"\
    "  3 = Table of vector classification models (Part 3)          \n"\
    "  4 = Extensions to the classification models (Part 4)        \n"\
    "  5 = Quit                                                    \n"\
    "                                                              \n"\
    "============================================================\n"

    table = \
        "===================================================================================================\n"\
        "   {0:10s}  |  {1:15}  |  {2:25}  |        {3:30}\n"\
        "               |                      |                               | {4:4} | {5:5} | {6:10}\n"\
        "===================================================================================================\n".\
        format("Stemming", "Position Weighting", "Local Collocation Modelling", "Accuracy", "tank",
               "plant", "pers/place")

    table2 = \
        "=================================================================================================================\n"\
        "   {0:10s}  |  {1:10s}  |  {2:15}  |  {3:25}  |        {4:30}\n"\
        "               |              |                      |                               | {5:4} | {6:5} | {7:10}\n"\
        "================================================================================================================= \n".\
        format("Model", "Stemming", "Position Weighting", "Local Collocation Modelling", "Accuracy", "tank",
               "plant", "pers/place")

    while True:
        sys.stderr.write(menu)
        option = raw_input("Enter Option: ")
        if option == "1":
            name = raw_input("Choose file (plant/tank/perplace): ")
            name = name.lower()
            if name != "plant" and name != "tank" and name != "perplace":
                exit("ERROR: File doesn't exist\n")
            stemming = raw_input("Choose stemmed or unstemmed: ")
            stemming = stemming.lower()
            if stemming != "stemmed" and stemming != "unstemmed":
                exit("ERROR: This option doesn't exist\n")
            init_files(stemming, name)
            init_corp_freq()
            init_doc_vectors()
            compute_centroid()
            classify_vectors(option)
        elif option == "2":
            name = raw_input("Choose file (plant/tank/perplace): ")
            name = name.lower()
            if name != "plant" and name != "tank" and name != "perplace":
                exit("ERROR: File doesn't exist\n")
            stemming = raw_input("Choose stemmed or unstemmed: ")
            stemming = stemming.lower()
            if stemming != "stemmed" and stemming != "unstemmed":
                exit("ERROR: This option doesn't exist\n")
            weighting = raw_input("Choose position weighting (exponential/stepped/custom): ")
            weighting = weighting.lower()
            if weighting != "exponential" and weighting != "stepped" and weighting != "custom":
                exit("ERROR: This weighting option doesn't exist\n")
            adjacency = raw_input("Include LR adjacency model (yes/no): ")
            adjacency = adjacency.lower()
            if adjacency != "yes" and adjacency != "no":
                exit("ERROR: This option doesn't exist\n")
            if adjacency == "yes":
                adjacency = "LR"
            else:
                adjacency = "none"
            init_files(stemming, name)
            init_corp_freq()
            init_doc_vectors("idf", weighting, adjacency)
            compute_centroid()
            classify_vectors(option)
        elif option == "3":
            counter = 1
            sys.stdout.write(table)
            print_table("unstemmed", option, "uniform", "none", counter)
            counter += 1
            print_table("stemmed", option, "exponential", "none", counter)
            counter += 1
            print_table("unstemmed", option, "exponential", "none", counter)
            counter += 1
            print_table("unstemmed", option, "exponential", "LR", counter)
            counter += 1
            print_table("unstemmed", option, "stepped", "none", counter)
            counter += 1
            print_table("unstemmed", option, "custom", "none", counter)
        elif option == "4":
            counter = 1
            sys.stdout.write(table2)
            print_gnb(counter)
            counter += 1
            print_knn(counter)
            counter += 1
            print_clustering(counter)
        elif option == "5":
            exit(0)
        else:
            sys.stderr.write("Input seems not right, try again\n")


def GNB(stemming, name):
    init_files(stemming, name)
    init_corp_freq()
    init_doc_vectors("none")

    unique_words = set()
    for vector in doc_vector:
        for token in vector.keys():
            unique_words.add(token)

    indexed_words = []
    for i in unique_words:
        indexed_words.append(i)

    dataset = []
    for vector in doc_vector:
        tmp = np.zeros(len(indexed_words))
        for token in vector.keys():
            if token in indexed_words:
                tmp[indexed_words.index(token)] = vector[token]
        dataset.append(np.array(tmp))

    train_data, test_data, train_labels, test_labels = train_test_split(dataset, sensenum, test_size=0.33)
    gnb = GaussianNB()
    gnb.fit(train_data, train_labels)
    predict_labels = gnb.predict(test_data)

    accuracy = round(accuracy_score(test_labels, predict_labels), 2)
    return accuracy


def KNN(stemming, name, weight, model):
    init_files(stemming, name)
    init_corp_freq()
    init_doc_vectors("idf", weight, model)

    unique_words = set()
    for vector in doc_vector:
        for token in vector.keys():
            unique_words.add(token)

    indexed_words = []
    for i in unique_words:
        indexed_words.append(i)

    dataset = []
    for vector in doc_vector:
        tmp = np.zeros(len(indexed_words))
        for token in vector.keys():
            if token in indexed_words:
                tmp[indexed_words.index(token)] = vector[token]
        dataset.append(np.array(tmp))

    train_data, test_data, train_labels, test_labels = train_test_split(dataset, sensenum, test_size=0.33)
    knn = KNeighborsClassifier()
    knn.fit(train_data, train_labels)
    predict_labels = knn.predict(test_data)

    accuracy = round(accuracy_score(test_labels, predict_labels), 2)
    return accuracy


def clustering(stemming, name, weight, model):
    init_files(stemming, name)
    init_corp_freq()
    init_doc_vectors("idf", weight, model)

    unique_words = set()
    for vector in doc_vector:
        for token in vector.keys():
            unique_words.add(token)

    indexed_words = []
    for i in unique_words:
        indexed_words.append(i)

    dataset = []
    for vector in doc_vector:
        tmp = np.zeros(len(indexed_words))
        for token in vector.keys():
            if token in indexed_words:
                tmp[indexed_words.index(token)] = vector[token]
        dataset.append(np.array(tmp))

    cluster = AgglomerativeClustering()
    predict_label = cluster.fit_predict(dataset)

    correct = 0
    incorrect = 0
    for i in range(len(sensenum)):
        if predict_label[i] + 1 == sensenum[i]:
            correct += 1
        else:
            incorrect += 1

    accuracy = float(correct) / (correct + incorrect)
    if accuracy < 0.5:
        accuracy = 1 - accuracy

    return accuracy


def get_weights(doc_line, new_doc_vec, weight, bag_of_words):
    center = 0
    for i in range(len(doc_line)):
        if doc_line[i][:3] == ".X-" or doc_line[i][:3] == ".x-":
            center = i
    for i in range(len(doc_line)):
        if bag_of_words == "LR":
            if i - center == 1:
                docs_freq_hash[doc_line[i]] -= 1
                if docs_freq_hash[doc_line[i]] == 0:
                    corp_freq_hash[doc_line[i]] = 0
                doc_line[i] = "R-" + doc_line[i]
                docs_freq_hash[doc_line[i]] += 1
                if corp_freq_hash[doc_line[i]] == 0:
                    corp_freq_hash[doc_line[i]] = 1
            elif i - center == -1:
                docs_freq_hash[doc_line[i]] -= 1
                if docs_freq_hash[doc_line[i]] == 0:
                    corp_freq_hash[doc_line[i]] = 0
                doc_line[i] = "L-" + doc_line[i]
                docs_freq_hash[doc_line[i]] += 1
                if corp_freq_hash[doc_line[i]] == 0:
                    corp_freq_hash[doc_line[i]] = 1
        if i != center:
            if weight == "exponential":
                new_doc_vec[doc_line[i]] += float(1) / abs(i - center)
            elif weight == "stepped":
                if abs(i - center) == 1:
                    new_doc_vec[doc_line[i]] += 6.0
                elif abs(i - center) == 2 or abs(i - center) == 3:
                    new_doc_vec[doc_line[i]] += 3.0
                else:
                    new_doc_vec[doc_line[i]] += 1.0
            elif weight == "custom":
                if abs(i - center) == 1:
                    new_doc_vec[doc_line[i]] += 100.0
                elif abs(i - center) == 2:
                    new_doc_vec[doc_line[i]] += 50.0
                else:
                    new_doc_vec[doc_line[i]] += 1.0
            elif weight == "uniform":
                new_doc_vec[doc_line[i]] += 1


def print_clustering(ind):
    t_numb = clustering("unstemmed", "tank", "stepped", "LR")
    t_numb2 = clustering("unstemmed", "plant", "stepped", "LR")
    t_numb3 = clustering("unstemmed", "perplace", "stepped", "LR")

    sys.stdout.write("{0:1d}  {1:10s}  |  {2:10s}  |  {3:15s}     |  {4:25s}    | {5:4.2f} | {6:4.2f}  | {7:7.2f}\n".
                     format(ind, "Clustering", "unstemmed", "#2-stepped", "#2-adjacent-separate-LR", t_numb, t_numb2, t_numb3))


def print_gnb(ind):
    t_numb = GNB("unstemmed", "tank")
    t_numb2 = GNB("unstemmed", "plant")
    t_numb3 = GNB("unstemmed", "perplace")

    sys.stdout.write("{0:1d}  {1:10s}  |  {2:10s}  |  {3:15s}     |  {4:25s}    | {5:4.2f} | {6:4.2f}  | {7:7.2f}\n".
                     format(ind, "Bayesian", "unstemmed", "#0-uniform", "#1-bag-of-words", t_numb, t_numb2, t_numb3))


def print_knn(ind):
    t_numb = KNN("unstemmed", "tank", "exponential", "none")
    t_numb2 = KNN("unstemmed", "plant", "exponential", "none")
    t_numb3 = KNN("unstemmed", "perplace", "exponential", "none")

    sys.stdout.write("{0:1d}  {1:10s}  |  {2:10s}  |  {3:15s}     |  {4:25s}    | {5:4.2f} | {6:4.2f}  | {7:7.2f}\n".
                     format(ind, "KNN", "unstemmed", "#1-expndecay", "#1-bag-of-words", t_numb, t_numb2, t_numb3))


def print_table(stemming, option, weighting, model, ind):
    init_files(stemming, "tank")
    init_corp_freq()
    init_doc_vectors("idf", weighting, model)
    compute_centroid()
    t_numb = classify_vectors(option)
    init_files(stemming, "plant")
    init_corp_freq()
    init_doc_vectors("idf", weighting, model)
    compute_centroid()
    t_numb2 = classify_vectors(option)
    init_files(stemming, "perplace")
    init_corp_freq()
    init_doc_vectors("idf", weighting, model)
    compute_centroid()
    t_numb3 = classify_vectors(option)

    if weighting == "uniform":
        weighting = "#0-uniform"
    elif weighting == "exponential":
        weighting = "#1-expndecay"
    elif weighting == "stepped":
        weighting = "#2-stepped"
    elif weighting == "custom":
        weighting = "#3-yours"

    if model == "none":
        model = "#1-bag-of-words"
    elif model == "LR":
        model = "#2-adjacent-separate-LR"

    sys.stdout.write("{0:1d}  {1:10s}  |  {2:15s}     |  {3:25s}    | {4:4.2f} | {5:4.2f}  | {6:7.2f}\n".
                     format(ind, stemming, weighting, model, t_numb, t_numb2, t_numb3))


def print_classification(result, accuracy):
    name, typo = token_docs.split("/")[-1].split(".")
    if typo == "tokenized":
        typo = "unstemmed"
    results = \
        "   {0:1s}  {1:1s}  {2:10s}  {3:5s}    {4:5s}   {5:50s}\n"\
        "========================================================="\
        "========================================================="\
        "==============\n".format("sense", "predict", "sim1-sim2", "sim1", "sim2", "doc#    sense    title")

    sys.stdout.write(results)
    for instance in result:
        sys.stdout.write("{0:1s}    {1:1d}    {2:3d}  {3:11.4f}    {4:5.4f}   {5:5.4f}   {6:47s}\n".
                         format(instance["symbol"], instance["label"], instance["predict"], instance["sim_diff"],
                                instance["sim1"], instance["sim2"], instance["title"]))
    sys.stdout.write("=========================================================" +
                     "=======================================================================\n")
    sys.stdout.write("\n" + name + " " + typo + " Prediction Accuracy: " + str(accuracy) + "\n")


def classify_vectors(option):
    result = []
    vec1_norm = sum(v * v for v in v_profile1.values())
    vec2_norm = sum(v * v for v in v_profile2.values())
    correct = 0
    incorrect = 0

    for i in range(3601, 4001):
        test = {}
        doc_norm = sum(v * v for v in doc_vector[i].values())
        sim1 = cosine_sim_a(doc_vector[i], v_profile1, doc_norm, vec1_norm)
        sim2 = cosine_sim_a(doc_vector[i], v_profile2, doc_norm, vec2_norm)
        sim_diff = sim1 - sim2
        label = sensenum[i]
        if sim1 > sim2:
            predict = 1
        else:
            predict = 2
        if label == predict:
            correct += 1
            symbol = "+"
        else:
            incorrect += 1
            symbol = "*"

        test["sim1"] = sim1
        test["sim2"] = sim2
        test["sim_diff"] = sim_diff
        test["label"] = label
        test["predict"] = predict
        test["symbol"] = symbol
        test["title"] = titles_vector[i]

        result.append(test)

    accuracy = float(correct) / (correct + incorrect)
    result = sorted(result, key=lambda k: -k["sim_diff"])
    if int(option) == 1 or int(option) == 2:
        print_classification(result, accuracy)

    return accuracy


def compute_centroid():
    compute_avg(1, v_profile1)
    compute_avg(2, v_profile2)


def compute_avg(sense, profile):
    temp = []
    for i in range(1, 3601):
        if sensenum[i] == sense:
            temp.append(doc_vector[i])
    for vector in temp:
        for token in vector.keys():
            if token not in profile:
                profile[token] = 0
            profile[token] += vector[token]
    n = len(temp)

    for token in profile.keys():
        profile[token] /= float(n)


########################################################
## COSINE_SIM_A
##
## Computes the cosine similarity for two vectors
## represented as associate arrays. You can also pass the
## norm as parameter
##
## Note: You may do it in a much efficient way like
## precomputing norms ahead or using packages like
## "numpy", below provide naive implementation of that
########################################################

def cosine_sim_a(vec1, vec2, vec1_norm=0.0, vec2_norm=0.0):
    if not vec1_norm:
        vec1_norm = sum(v * v for v in vec1.values())
    if not vec2_norm:
        vec2_norm = sum(v * v for v in vec2.values())

    # save some time of iterating over the shorter vec
    if len(vec1) > len(vec2):
        vec1, vec2 = vec2, vec1

    # calculate the cross product
    cross_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in vec1.keys())

    if vec1_norm == 0 or vec2_norm == 0:
        return 0
    else:
        return cross_product / math.sqrt(vec1_norm * vec2_norm)

########################################################
##  COSINE_SIM_B
##  Same thing, but to be consistant with original perl
##  script, we add this line
########################################################


def cosine_sim_b(vec1, vec2, vec1_norm=0.0, vec2_norm=0.0):
    return cosine_sim_a(vec1, vec2, vec1_norm, vec2_norm)


if __name__ == "__main__": main()
