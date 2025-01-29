import os
import string
import time
import matplotlib.pyplot as plt
import networkx as nx
import warnings

from wordcloud import WordCloud
from xgboost import XGBClassifier
from textblob import Word, TextBlob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from skimage.metrics import mean_squared_error

import gensim as gensim
from gensim import corpora
from gensim.models import LdaModel

import nltk
from nltk.stem import PorterStemmer as Stemmer
from nltk.stem.wordnet import WordNetLemmatizer as Lemmatizer
from nltk import word_tokenize, re, sent_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# nltk.download('punkt')  # one time execution
warnings.filterwarnings("ignore")

# GLOBAL VARIABLES
READED_FILE = 'parsed_files.txt'
PATH = 'txt_sentoken'
STEMMER = Stemmer()
LEMMATIZER = Lemmatizer()
VECTOR_SIZE = 50


# UTILITIES FUNCTIONS
def nested_to_dataframe(text_list):
    df = pd.DataFrame(columns=['Text'])
    index = 0
    for text in text_list:
        df.loc[index] = ' '.join(text)
        index += 1
    return df


# READER FUNCTIONS
def read_data():
    st = time.process_time()
    files_text = []
    for p in Path(PATH).glob('**/*.txt'):
        files_text.append(p.read_text())
    res = time.process_time() - st
    print('Process Time: {} seconds'.format(res))
    return files_text


def read_ready_file():
    words = []
    file = open(READED_FILE, 'r')
    for text_line in file.readlines():
        temp = text_line.split(',')
        temp[-1] = temp[-1][0:-1]
        words.append(temp)
    file.close()
    return words


def read_ready_file_text():
    text = ''
    file = open(READED_FILE, 'r')
    for text_line in file.readlines():
        temp = text_line.split(',')
        temp[-1] = temp[-1][0:-1]
        text += ' '.join(temp)
    file.close()
    return text


# WRITER FUNCTIONS
def write_to_file(text_files):
    file = open("parsed_files.txt", "w")
    for text_list in text_files:
        file.write(','.join(text_list))
        file.write('\n')
    print('File is created.\n')


def write_to_text(text_list):
    file = open("parsed.txt", "w")
    file.write(','.join(text_list))
    print('File is created.\n')


# FUNCTIONS
def tokenize_words(files_text):
    index = 0
    for text in files_text:
        files_text[index] = word_tokenize(text)
        index += 1
    return files_text


def remove_stopwords(text_list):
    stopword = nltk.corpus.stopwords.words('english')
    return_list = []
    for text in text_list:
        return_list.append([word for word in text if word not in stopword])
    return return_list


def stemming(text_list):
    return_list = []
    for text in text_list:
        temp_word_list = []
        for word in text:
            if word:
                temp_word_list.append(STEMMER.stem(word))
        return_list.append(temp_word_list)
    return return_list


def lemmatization(text_list, pos):
    return_list = []
    for text in text_list:
        temp_word_list = []
        for word in text:
            if word:
                temp_word_list.append(LEMMATIZER.lemmatize(word, pos=pos))
        return_list.append(temp_word_list)

    return return_list


def remove_punctuation(text_list):
    return_list = []
    for text in text_list:
        for remove in string.punctuation:
            if remove == '\'':
                text = text.replace(remove, '')
            else:
                text = text.replace(remove, ' ')
        return_list.append(text)
    return return_list


def remove_duplicate_letters(text_files):
    return_list = []
    for text in text_files:
        inner_list = []
        for word in text:
            rx = re.compile(r'([^\W\d_])\1{2,}')
            word = re.sub(r'[^\W\d_]+',
                          lambda x: Word(rx.sub(r'\1', x.group())).correct() if rx.search(x.group()) else x.group(),
                          word)
            inner_list.append(word)
        return_list.append(inner_list)
    return return_list


if __name__ == '__main__':
    ###################################################
    #  PART A.1
    ###################################################
    print('Files are reading...')
    text_files = read_data()
    print('Files readed.\n')

    print('Cleaning...')
    cleaned_text = remove_punctuation(text_files)
    print('Cleaned.\n')

    print('Words tokenizing...')
    tokenized_text_files = tokenize_words(cleaned_text)
    print('Words tokenized.\n')

    print('Removing Duplicate...')
    removed_text_files = remove_duplicate_letters(tokenized_text_files)
    print('Duplicate removed.\n')

    print('Stopwords removing...')
    text_files_no_stopwords = remove_stopwords(removed_text_files)
    print('Stopwords removed.\n')

    print('Lemmatizationing...')
    lemmatizatied_text_files = lemmatization(text_files_no_stopwords, 'v')
    lemmatizatied_text_files2 = lemmatization(lemmatizatied_text_files, 'n')
    print('Words lemmatizatied.\n')
    print('Dataset is ready.\n')
    write_to_file(tokenized_text_files)

    #  READING FROM FILE
    words = read_ready_file()


    ###################################################
    #  PART B.1
    ###################################################

    ###################################################
    #                WORD COUNT
    ###################################################
    def get_word_count(word_list):
        words_dict = dict()
        for text in word_list:
            for word in text:
                if word in words_dict:
                    words_dict[word] += 1
                else:
                    words_dict[word] = 1
        dataset = pd.DataFrame(words_dict.items(), columns=['Word', 'Count'])
        dataset.sort_values(by=['Count'], inplace=True, ascending=False)
        dataset = dataset.reset_index(drop=True)

        return dataset.loc[0]


    print(get_word_count(words).head(10))


    ###################################################
    #                LONGEST WORDS
    ###################################################

    def longest_word(words_list):
        words_dict = dict()
        for word in words_list:
            words_dict[word] = len(word)
        dataset = pd.DataFrame(words_dict.items(), columns=['Word', 'Length'])
        dataset.sort_values(by=['Length'], inplace=True, ascending=False)
        dataset = dataset.reset_index(drop=True)
        return dataset.loc[0]


    def get_longest_words(words_list):
        df = pd.DataFrame(columns=['Word', 'Length'])
        index = 0
        for words in words_list:
            df.loc[index] = longest_word(words)
            index += 1
        df.sort_values(by=['Length'], inplace=True, ascending=False)
        return df


    print(get_longest_words(words).head(10))


    ###################################################
    #                4 DIGIT DATE COUNT
    ###################################################

    def get_dates_in_word(word_list):
        words_dict = dict()
        for text in word_list:
            for word in text:
                if word.isdigit() and len(word) == 4:
                    if word in words_dict:
                        words_dict[word] += 1
                    else:
                        words_dict[word] = 1
        dataset = pd.DataFrame(words_dict.items(), columns=['Word', 'Count'])
        dataset.sort_values(by=['Count'], inplace=True, ascending=False)
        dataset = dataset.reset_index(drop=True)
        if not dataset.empty:
            return dataset.loc[0]


    print(get_dates_in_word(words).head(10))


    ###################################################
    #            SUBJECTIVITY
    ###################################################
    def print_subjectivity(text_list, count=10, type='h'):
        df = nested_to_dataframe(text_list)
        df['Subjectivity'] = df['Text'].apply(lambda x: TextBlob(str(x)).
                                              sentiment.subjectivity)
        df.sort_values(by=['Subjectivity'], inplace=True, ascending=False)
        if type == 'h':
            print(df.head(count))
        else:
            print(df.tail(count))


    print_subjectivity(words)


    ###################################################
    #  PART B.2
    ###################################################
    ###################################################
    #             BAG OF WORDS
    ###################################################
    def BoW(words_list, max_fea=10):
        bag_of_words_model = CountVectorizer(max_features=max_fea)
        bag_of_word_df = pd.DataFrame(bag_of_words_model.fit_transform([' '.join(word) for word in words_list]).
                                      todense())
        bag_of_word_df.columns = sorted(bag_of_words_model.vocabulary_)
        return [bag_of_words_model, bag_of_word_df]


    bow_model, linreg_df = BoW(words)
    print(linreg_df.head(10))


    ###################################################
    #             TF IDF
    ###################################################
    def tf_idf(words_list, max_fea=10):
        tfidf_model = TfidfVectorizer(max_features=max_fea)
        tfidf_df = pd.DataFrame(tfidf_model.fit_transform([' '.join(word) for word in words_list]).
                                todense())
        tfidf_df.columns = sorted(tfidf_model.vocabulary_)
        return [tfidf_model, tfidf_df]


    tfidf_model, tfidf_df = tf_idf(words)
    print(linreg_df.head(10))


    ###################################################
    #  PART B.3
    ###################################################
    ###################################################
    #             WORD CLOUD
    ###################################################
    def show_word_cloud(words_list):
        wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              max_words=50,
                              min_font_size=10).generate(words_list)
        plt.figure(figsize=(15, 15))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


    show_word_cloud(read_ready_file_text())


    ###################################################
    #             PART B.4
    ###################################################
    ###################################################
    #                LINEAR REG
    ###################################################
    def lin_reg(model_df, words_list):
        df = nested_to_dataframe(words_list)
        df['Target'] = [0 for x in range(0, 1000)] + [1 for x in range(0, 1000)]
        linreg = LinearRegression()
        linreg.fit(model_df, df['Target'])
        predicted_datas = linreg.predict(model_df)

        predicted_classes = []
        for i in predicted_datas:
            if i < 0.5:
                predicted_classes.append(0)
            else:
                predicted_classes.append(1)

        df['predicted'] = predicted_classes
        df['predicted_datas'] = predicted_datas

        return [df, linreg]


    linreg_df, linreg_model = lin_reg(tfidf_df, words)
    print(linreg_df.head(10))


    ###################################################
    #             LOGISTIC REG
    ###################################################
    def log_reg(model_df, words_list):
        df = nested_to_dataframe(words_list)
        df['Target'] = [0 for _ in range(0, 1000)] + [1 for _ in range(0, 1000)]
        logreg = LogisticRegression()
        logreg.fit(model_df, df['Target'])
        predicted_labels = logreg.predict(model_df)
        df['predicted'] = predicted_labels
        return [df, logreg]


    logreg_df, logreg_model = log_reg(tfidf_df, words)
    print(logreg_df.head(10))


    ###################################################
    #             ELBOW
    ###################################################
    def plot_elbow(model_df, max_cluster):
        distortions = []
        K = range(1, max_cluster + 1)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(model_df)
            distortions.append(
                sum(np.min(cdist(model_df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / model_df.shape[0])
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal number of clusters')
        plt.show()


    plot_elbow(tfidf_df, 15)


    ###################################################
    #             k MEANS
    ###################################################
    def k_means(model_df, cluster_number, words_list):
        df = nested_to_dataframe(words_list)
        df['Target'] = [0 for _ in range(0, 1000)] + [1 for _ in range(0, 1000)]
        kmeans = KMeans(n_clusters=cluster_number)
        kmeans.fit(model_df)
        y_kmeans = kmeans.predict(model_df)
        df['predicted'] = y_kmeans
        return [df, kmeans]


    kmeans_df, kmeans_model = k_means(tfidf_df, 2, words)
    print(kmeans_df.head(10))


    ###################################################
    #          TREE METHOD
    ###################################################
    # CREATE CLF MODEL
    def clf_model(model_type, X_train, y):
        model = model_type.fit(X_train, y)
        predicted_labels = model.predict(tfidf_df)
        return predicted_labels


    # CREATE XGB
    def XGB(model_df, words_list):
        df = nested_to_dataframe(words_list)
        df['Target'] = [0 for _ in range(0, 1000)] + [1 for _ in range(0, 1000)]
        xgb_clf = XGBClassifier(n_estimators=20, learning_rate=0.03, max_depth=5, subsample=0.6, colsample_bytree=0.6,
                                reg_alpha=10, seed=42)
        df['predicted'] = clf_model(xgb_clf, model_df, df['Target'])
        pd.crosstab(df['Target'], df['predicted'])
        return [df, xgb_clf]


    xgb_df, xgb_model = XGB(tfidf_df, words)
    print(xgb_df.head(10))


    # CREATE DTC
    def DTC(model_df, words_list):
        df = nested_to_dataframe(words_list)
        df['Target'] = [0 for _ in range(0, 1000)] + [1 for _ in range(0, 1000)]
        dtc = tree.DecisionTreeClassifier()
        dtc = dtc.fit(model_df, df['Target'])
        df['predicted'] = dtc.predict(tfidf_df)
        return [df, dtc]


    ######################################
    #         SHOWING RESULTS            #
    ######################################

    # SHOW ACCURACY
    def show_accuracy(tab):
        accuracy = (tab.iloc[0, 0] + tab.iloc[1, 1]) / tab.to_numpy().sum()
        print(f'Accuracy : %{(accuracy * 100):.2f}')


    # PRECISION
    def get_pre(tab):
        pre = tab.iloc[1, 1] / (tab.iloc[1, 1] + tab.iloc[0, 1])
        return pre


    # RECALL
    def get_rec(tab):
        rec = tab.iloc[1, 1] / (tab.iloc[1, 1] + tab.iloc[1, 0])
        return rec


    # SHOW PRECISION
    def show_pre(tab):
        print(f'Precision: {get_pre(tab):.2f}')


    # SHOW F1 SCORE
    def show_f1_score(tab):
        pre = get_pre(tab)
        rec = get_rec(tab)
        f1_score = 2 * pre * rec / (rec + pre)
        print(f'F1 Score : {f1_score:.2f}')


    # PLOT ROC CURVE
    def plot_roc(df_test, ml_model, numeric_df):
        y_pred_proba = ml_model.predict(numeric_df)
        fpr, tpr, _ = roc_curve(df_test['Target'], y_pred_proba)
        plt.plot(fpr, tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


    # PLOT ROC
    def plot_roc_lor(df_test, ml_model, numeric_df):
        y_pred_proba = ml_model.predict_proba(numeric_df)[::, 1]
        fpr, tpr, _ = roc_curve(df_test['Target'], y_pred_proba)
        plt.plot(fpr, tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


    # GET ROOT MEAN SQUARED ERROR
    def get_rmse(df):
        rmse = mean_squared_error(df['Target'], df['predicted'])
        return rmse


    # SHOW ROOT MEN SQUARED ERROR
    def show_rmse(df):
        rmse = get_rmse(df)
        print(f'Root Mean Squared Error : {rmse:.2f}')


    print('K-Means')
    kmeans_tab = pd.crosstab(kmeans_df['Target'], kmeans_df['predicted'])
    show_accuracy(kmeans_tab)
    show_pre(kmeans_tab)
    show_f1_score(kmeans_tab)
    show_rmse(kmeans_df)
    plot_roc(kmeans_df, kmeans_model, tfidf_df)
    print(kmeans_tab)

    # Showing Linear Regression Results
    print('Linear Regression')
    linreg_tab = pd.crosstab(linreg_df['Target'], linreg_df['predicted'])
    show_accuracy(linreg_tab)
    show_pre(linreg_tab)
    show_f1_score(linreg_tab)
    show_rmse(linreg_df)
    plot_roc(linreg_df, linreg_model, tfidf_df)
    print(linreg_tab)

    # Showing Logistic Regression Results
    print('Logistic Regression')
    logreg_tab = pd.crosstab(logreg_df['Target'], logreg_df['predicted'])
    show_accuracy(logreg_tab)
    show_pre(logreg_tab)
    show_f1_score(logreg_tab)
    show_rmse(logreg_df)
    plot_roc_lor(logreg_df, logreg_model, tfidf_df)
    print(logreg_tab)

    # Showing XGBoost Results
    print('XGBoost')
    xgb_tab = pd.crosstab(xgb_df['Target'], xgb_df['predicted'])
    show_accuracy(xgb_tab)
    show_pre(xgb_tab)
    show_f1_score(xgb_tab)
    show_rmse(xgb_df)
    plot_roc(xgb_df, xgb_model, tfidf_df)
    print(xgb_tab)


    #### If we want to print fingerprints of file

    # def topic_vector(topic_model:LdaModel, text:str):
    #     fingerprint = [0] * topic_model.num_topics
    #     for topic, prob in topic_model[dictionary.doc2bow(text)]:
    #         fingerprint[topic] = prob
    #     return fingerprint
    #
    # def show_fingerprint(topic_model, text:str):
    #     #display(text)
    #     vector = topic_vector(topic_model, text)
    #     plt.figure(figsize=(8,1))
    #     ax = plt.bar( range(len(vector)),vector,0.25,linewidth=1)
    #     plt.ylim(top=0.4)
    #     plt.tick_params(axis='both',which='both',left=False,bottom=False,top=False,labelleft=True,labelbottom=True)
    #     plt.grid(False)
    #
    # out
    # style.use('fivethirtyeight')
    # VECTOR_SIZE=100
    # %matplotlib inline
    # show_fingerprint(lda_model, words_list[0])

    ###################################################
    #             PART C
    ###################################################

    # GET DICT AND CORPUS VALUES
    def get_corpus(words_list):
        dictionary = gensim.corpora.Dictionary(words_list)
        corpus = [dictionary.doc2bow(text) for text in words_list]
        return [dictionary, corpus]


    # GET LDA MODEL
    def lda(corpus, words_list, dictionary):
        lda_model = LdaModel(corpus, num_topics=VECTOR_SIZE, passes=4)
        arr = []
        for words in words_list:
            arr += words
        bag_of_words = dictionary.doc2bow(arr)
        df = pd.DataFrame(lda_model[bag_of_words], columns=['Topic', 'Relevance']).set_index('Topic')
        return [lda_model, df]


    # SHOW LDA TOPICS
    def show_topics_lda(words_list, topic_number, word_number):
        dictionary = corpora.Dictionary(words_list)
        corpus = [dictionary.doc2bow(text) for text in words_list]
        ldamodel = LdaModel(corpus, num_topics=topic_number, id2word=dictionary, passes=15)
        topics = ldamodel.show_topics(num_words=word_number)

        # Create empty matrix
        list_topics = []
        for i in range(topic_number):  # For 3 topics => (1x3)matrix
            list_topics.append([])

        index = 0
        for topic in topics:
            for topic_word in (topic[1].split('"')):
                if topic_word.isalpha():
                    list_topics[index].append(topic_word)
            index += 1
        for topic_list in list_topics:
            print(topic_list)

        return list_topics


    # SHOW K_MEANS TOPICS
    def show_topics_kmeans(model, model_df, topic_number, word_number):
        kmeans = MiniBatchKMeans(n_clusters=topic_number)
        kmeans.fit(model_df)
        centers = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = model.get_feature_names_out()
        words_list = []
        for i in range(0, topic_number):
            word_list = []
            for j in centers[i, :word_number]:
                word_list.append(terms[j])
            print(word_list)
            words_list.append(word_list)
        return words_list


    # Creating LDA model and showing relevance values
    word_dic, corpus = get_corpus(words)
    lda_model, rel_df = lda(corpus, words, word_dic)
    rel_df.sort_values(by=['Relevance'], inplace=True, ascending=False)
    rel_df.head(10)

    # from sklearn.cluster import MiniBatchKMeans
    # K = range(1,15)
    # SSE = []
    # for k in K:
    #     kmeans = MiniBatchKMeans(n_clusters = k,batch_size = 300)
    #     kmeans.fit(tfidf_df)
    #     SSE.append(kmeans.inertia_)

    # import matplotlib.pyplot as plt
    # plt.plot(K,SSE,'bx-')
    # plt.title('Elbow Method')
    # plt.xlabel('cluster numbers')
    # plt.show()

    # We chose topic number according to number of relevance that is greater than 0.1

    # Printing LDA Topics
    print('LDA Topics:')
    lda_topics = show_topics_lda(words, topic_number=2, word_number=10)

    # Printing K-Means Topics
    from sklearn.cluster import MiniBatchKMeans

    print('K_Means Topics:')
    kmeans_topics = show_topics_kmeans(tfidf_model, tfidf_df, topic_number=2, word_number=10)

    # Showing similarity between model's topics
    acc_list = []
    for i in range(len(kmeans_topics)):
        acc = len(set(lda_topics[i]) & set(kmeans_topics[i])) / 10
        acc_list.append(acc)
    print('Similarity Accuracy:')
    print(acc_list)


    ###################################################
    #             PART D
    ###################################################
    ###################################################
    #             SUMMARIZING FUNCTIONS
    ###################################################
    def sentence_remove_stopwords(sentences):
        return_list = []
        stopword = nltk.corpus.stopwords.words('english')
        for sentence in sentences:
            return_list.append([word for word in sentence if word not in stopword])
        return return_list


    def sentence_verb_lemmatization(sentences):
        return_list = []
        for sentence in sentences:
            temp_word_list = []
            for word in sentence:
                if word:
                    temp_word_list.append(LEMMATIZER.lemmatize(word, pos='v'))
            return_list.append(temp_word_list)
        return return_list


    def sentence_noun_lemmatization(sentences):
        return_list = []
        for sentence in sentences:
            temp_word_list = []
            for word in sentence:
                if word:
                    temp_word_list.append(LEMMATIZER.lemmatize(word, pos='n'))
            return_list.append(temp_word_list)
        return return_list


    def sentence_remove_punctuation(sentences):
        return_list = []
        for sentence in sentences:
            for remove in string.punctuation:
                sentence = sentence.replace(remove, ' ')
            return_list.append(sentence)
        return return_list


    def sentence_remove_duplicate_letters(sentences):
        return_list = []
        for sentence in sentences:
            inner_list = []
            for word in sentence:
                rx = re.compile(r'([^\W\d_])\1{2,}')
                word = re.sub(r'[^\W\d_]+',
                              lambda x: Word(rx.sub(r'\1', x.group())).correct() if rx.search(x.group()) else x.group(),
                              word)
                inner_list.append(word)
            return_list.append(inner_list)
        return return_list


    def sentence_tokenize_words(sentences):
        temp_list = [word_tokenize(sentence) for sentence in sentences]
        return temp_list


    CLEAN_PATTERN = r'[^a-zA-z\s]'


    def clean(word):
        return re.sub(CLEAN_PATTERN, '', word)


    def clean_sentence(sentence):
        sentence = [clean(word) for word in sentence]
        return [word for word in sentence if word]


    def clean_sentences(sentences):
        temp_list = [clean_sentence(sentence) for sentence in sentences]
        return temp_list


    def load_glove_vectors(fn):
        st = time.process_time()
        print("Loading Glove Model")
        with open(fn, 'r', encoding='utf8') as glove_vector_file:
            model = {}
            for line in glove_vector_file:
                parts = line.split()
                word = parts[0]
                embedding = np.array([float(val) for val in parts[1:]])
                model[word] = embedding
            res = time.process_time() - st
            print("Loaded {} words. Processing Time: {} sec\n".format(len(model), res))
        return model


    glove_vectors = load_glove_vectors('glove.6B.50d.txt')

    EMPTY_VECTOR = np.zeros(VECTOR_SIZE)


    def sentence_vector(sentence):
        return sum([glove_vectors.get(word, EMPTY_VECTOR)
                    for word in sentence]) / len(sentence)


    def sentences_to_vectors(sentences):
        return [sentence_vector(sentence)
                for sentence in sentences if sentence]


    def similarity_matrix(sentence_vectors):
        sim_mat = np.zeros([len(sentence_vectors), len(sentence_vectors)])
        for i in range(len(sentence_vectors)):
            for j in range(len(sentence_vectors)):
                element_i = sentence_vectors[i].reshape(1, VECTOR_SIZE)
                element_j = sentence_vectors[j].reshape(1, VECTOR_SIZE)
                sim_mat[i][j] = cosine_similarity(element_i, element_j)[0, 0]
        return sim_mat


    def compute_graph(sim_matrix):
        nx_graph = nx.from_numpy_array(sim_matrix)
        # scores = nx.pagerank(nx_graph, max_iter=1000)
        scores = nx.pagerank_numpy(nx_graph)
        return scores


    def get_ranked_sentences(sentences, scores):
        temp_dict = dict()
        for i in range(min(len(sentences), len(scores))):
            temp_dict[scores[i]] = sentences[i]
        top_scores = dict(sorted(temp_dict.items(), key=lambda item: item[1]))
        sorted_sentences = [sentence for sentence in top_scores.values()]
        return " ".join(sorted_sentences)


    ###################################################
    #             NEG SUMMARIZED
    ###################################################

    def read_neg_files():
        print('Neg files are reading...')
        files_text = []
        for p in Path(PATH).glob('neg/*.txt'):
            files_text.append(p.read_text().replace('\n', ''))
        print('Neg files readed.\n')
        return files_text


    def write_neg_files():
        file = open("neg_summarized.txt", "w")
        for i in range(1000):
            file.write(neg_df['Summary'][i])
            file.write('\n')
        print('Summarized neg file is created.\n')


    neg_text = read_neg_files()
    neg_df = pd.DataFrame(neg_text, columns=['Files'])

    st = time.process_time()
    print('Sentences are creating...')
    neg_df['SentencesInFiles'] = neg_df.Files.apply(sent_tokenize)
    res = time.process_time() - st
    print('Sentences are created. Processing Time: {} sec\n'.format(res))

    st = time.process_time()
    print('Words are creating...')
    neg_df['WordsInSentence'] = neg_df.SentencesInFiles \
        .apply(sentence_remove_punctuation) \
        .apply(sentence_tokenize_words) \
        .apply(sentence_remove_duplicate_letters) \
        .apply(sentence_remove_stopwords) \
        .apply(sentence_verb_lemmatization) \
        .apply(sentence_noun_lemmatization) \
        .apply(clean_sentences)
    res = time.process_time() - st
    print('Words are created. Processing Time: {} sec\n'.format(res))

    st = time.process_time()
    print('Sentence vectors are creating...')
    neg_df['SentenceVector'] = neg_df.WordsInSentence.apply(sentences_to_vectors)
    res = time.process_time() - st
    print('Sentence vectors are created. Processing Time: {} sec\n'.format(res))

    st = time.process_time()
    print('Sim matrices are creating...')
    neg_df['SimMatrix'] = neg_df.SentenceVector.apply(similarity_matrix)
    res = time.process_time() - st
    print('Sim matrices are created. Processing Time: {} sec\n'.format(res))

    st = time.process_time()
    print('Graph is creating...')
    neg_df['Graph'] = neg_df.SimMatrix.apply(compute_graph)
    res = time.process_time() - st
    print('Graph is created. Processing Time: {} sec\n'.format(res))

    st = time.process_time()
    print('Summary is creating...')
    neg_df['Summary'] = neg_df.apply(lambda d: get_ranked_sentences(d.SentencesInFiles, d.Graph), axis=1)
    res = time.process_time() - st
    print('Summary is created. Processing Time: {} sec\n'.format(res))

    write_neg_files()


    # ###################################################
    # #             POS SUMMARIZED
    # ###################################################

    def read_pos_files():
        print('Pos files are reading...')
        files_text = []
        for p in Path(PATH).glob('pos/*.txt'):
            files_text.append(p.read_text().replace('\n', ''))
        print('Pos files readed.\n')
        return files_text


    def write_pos_files():
        file = open("pos_summarized.txt", "w")
        for i in range(1000):
            file.write(pos_df['Summary'][i])
            file.write('\n')
        print('Summarized pos file is created.\n')


    pos_text = read_pos_files()
    pos_df = pd.DataFrame(pos_text, columns=['Files'])

    st = time.process_time()
    print('Sentences are creating...')
    pos_df['SentencesInFiles'] = pos_df.Files.apply(sent_tokenize)
    res = time.process_time() - st
    print('Sentences are created. Processing Time: {} sec\n'.format(res))

    st = time.process_time()
    print('Words are creating...')
    pos_df['WordsInSentence'] = pos_df.SentencesInFiles \
        .apply(sentence_remove_punctuation) \
        .apply(sentence_tokenize_words) \
        .apply(sentence_remove_duplicate_letters) \
        .apply(sentence_remove_stopwords) \
        .apply(sentence_verb_lemmatization) \
        .apply(sentence_noun_lemmatization) \
        .apply(clean_sentences)
    res = time.process_time() - st
    print('Words are created. Processing Time: {} sec\n'.format(res))

    st = time.process_time()
    print('Sentence vectors are creating...')
    pos_df['SentenceVector'] = pos_df.WordsInSentence.apply(sentences_to_vectors)
    res = time.process_time() - st
    print('Sentence vectors are created. Processing Time: {} sec\n'.format(res))

    st = time.process_time()
    print('Sim matrices are creating...')
    pos_df['SimMatrix'] = pos_df.SentenceVector.apply(similarity_matrix)
    res = time.process_time() - st
    print('Sim matrices are created. Processing Time: {} sec\n'.format(res))

    st = time.process_time()
    print('Graph is creating...')
    pos_df['Graph'] = pos_df.SimMatrix.apply(compute_graph)
    res = time.process_time() - st
    print('Graph is created. Processing Time: {} sec\n'.format(res))

    st = time.process_time()
    print('Summary is creating...')
    pos_df['Summary'] = pos_df.apply(lambda d: get_ranked_sentences(d.SentencesInFiles, d.Graph), axis=1)
    res = time.process_time() - st
    print('Summary is created. Processing Time: {} sec\n'.format(res))

    write_pos_files()

    ###################################################
    #             PART E
    ###################################################
    # Creating Empty Data Frame
    df = pd.DataFrame(columns=['text', 'target'])


    # Read Datas
    def target_read(df, target):
        os.chdir(f'/txt_sentoken/{target}')
        for file in os.listdir():
            if file.endswith('.txt'):
                infile = open(file, 'r')
                contents = infile.read()
                df_new_row = pd.DataFrame.from_records([{'text': contents, 'target': target}])
                df = pd.concat([df, df_new_row], ignore_index=True)
        return df


    # Clean the datas with protect the emotions
    def clean_new(text):
        text = re.sub(r'[\W]+', ' ', text.lower())
        text = text.replace('hadn t', 'had not') \
            .replace('wasn t', 'was not') \
            .replace('didn t', 'did not')
        return text


    # Reading Part
    df = target_read(df, 'neg')
    df = target_read(df, 'pos')

    # Cleaning
    for i in range(len(df)):
        text = df.iloc[i]['text']  # Text
        clean_text = clean_new(text)
        df.loc[i, 'text'] = clean_text

    # Train test data split
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'],
                                                        test_size=0.25,
                                                        random_state=24, )

    # TF-IDF
    tfidf = TfidfVectorizer(strip_accents=None,
                            preprocessor=None,
                            lowercase=False, )

    # Bag Of Words
    bow = CountVectorizer(max_features=20)  # Bag

    # Logistic Regression with TFIDF
    log_reg = LogisticRegression(random_state=0, solver='lbfgs')
    log_tfidf = Pipeline([('vect', tfidf), ('clf', log_reg)])

    log_tfidf.fit(X_train, y_train)
    print("Log. Reg. TFIDF: ", cross_val_score(log_tfidf, X_train, y_train, cv=4).mean())
    #

    log_reg = LogisticRegression(random_state=0, solver='lbfgs')
    log_bow = Pipeline([('vect', bow), ('clf', log_reg)])

    log_bow.fit(X_train, y_train)
    print("Log. Reg. BoW: ", cross_val_score(log_bow, X_train, y_train, cv=4).mean())

    # Bag Of Words is less succesful than TFIDF so we will use TFIDF in other models
    rfc = RandomForestClassifier(criterion='gini')
    rfc_tfidf = Pipeline([('vect', tfidf), ('clf', rfc)])

    rfc_tfidf.fit(X_train, y_train)
    print("Random Forest TFIDF: ", cross_val_score(rfc_tfidf, X_train, y_train, cv=4).mean())

    tree = nltk.DecisionTreeClassifier()
    tree_tfidf = Pipeline([('vect', tfidf), ('clf', tree)])

    tree_tfidf.fit(X_train, y_train)
    print("DTC TFIDF: ", cross_val_score(tree_tfidf, X_train, y_train, cv=4).mean())

    # According to theese train validation values We choose logistic regression and search the best hyperparameters
    from sklearn.model_selection import GridSearchCV, cross_val_score

    tree_tfidf = Pipeline([('vect', tfidf), ('log_reg', log_reg)])

    params = {'log_reg__C': [0.01, 0.1, 1, 10, 100, 200],
              'log_reg__solver': ['lbfgs', 'liblinear'],
              }

    search = GridSearchCV(tree_tfidf, param_grid=params, scoring='accuracy', cv=4, refit=True)
    search.fit(X_train, y_train)
    print('Best Estimator = ', search.best_estimator_, '\nScore= ', search.best_score_)

    # Then we will test the best values
    test_accuracy = search.score(X_test, y_test)
    print('The model has a test accuracy of {:.2%}'.format(test_accuracy))
