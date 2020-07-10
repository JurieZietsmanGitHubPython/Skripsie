__author__ = ["Jurie Zietsman"]

import re
import sys
import warnings

import matplotlib
import nltk
import numpy as np
import pandas as pd
import qdarkgraystyle
import seaborn as sns
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
matplotlib.use('QT5Agg')
porter = nltk.PorterStemmer()


class window(QMainWindow):

    def __init__(self):
        super(window, self).__init__()
        uic.loadUi("C:/Users/Jurie/PycharmProjects/Skripsie/User Interface/main_window.ui", self)

        ##############################################################
        # Function calls for homepage and closing application
        ##############################################################

        self.stack.setCurrentIndex(0)
        self.startButton.clicked.connect(self.on_startButton_clicked)
        self.processingQuitButton.clicked.connect(self.close_application)
        self.resultsQuitButton.clicked.connect(self.close_application)

        ##############################################################
        # Function calls for homepage and closing application
        ##############################################################

        self.radioStopword.toggled.connect(lambda: self.toggleNormalisation(self.radioStopword))
        self.radioStopwordNone.toggled.connect(lambda: self.toggleNormalisation(self.radioStopwordNone))

        self.radioStemming.toggled.connect(lambda: self.toggleML(self.radioStemming))
        self.radioLemmatisation.toggled.connect(lambda: self.toggleML(self.radioLemmatisation))
        self.radioNonen.toggled.connect(lambda: self.toggleML(self.radioNonen))

        self.checkBoxUni.stateChanged.connect(self.toggleFC)
        self.checkBoxUniBi.stateChanged.connect(self.toggleFC)
        self.checkBoxUniTri.stateChanged.connect(self.toggleFC)
        self.checkBoxBi.stateChanged.connect(self.toggleFC)
        self.checkBoxBiTri.stateChanged.connect(self.toggleFC)
        self.checkBoxTri.stateChanged.connect(self.toggleFC)
        self.checkBoxUniBiTri.stateChanged.connect(self.toggleFC)

        self.checkBoxPres.stateChanged.connect(self.toggleFS)
        self.checkBoxCount.stateChanged.connect(self.toggleFS)
        self.checkBoxTDIDF.stateChanged.connect(self.toggleFS)

        self.feature_number.textChanged.connect(self.toggleTune)

        self.checkBoxIG.stateChanged.connect(self.toggleFeatureNumber)
        self.checkBoxDF.stateChanged.connect(self.toggleFeatureNumber)
        self.checkBoxFSNone.stateChanged.connect(self.toggleTune)

        self.checkBoxSVM.stateChanged.connect(self.toggleNgram)
        self.checkBoxNB.stateChanged.connect(self.toggleNgram)
        self.checkBoxME.stateChanged.connect(self.toggleNgram)

        self.comboBoxTune.currentIndexChanged.connect(self.toggleTuneFolds)

        self.comboBoxTuneFold.currentIndexChanged.connect(self.toggleEvaluation)

        self.comboBoxEvaluation.currentIndexChanged.connect(self.toggleFolds)

        self.comboBoxFolds.currentIndexChanged.connect(self.toggleButton)

        self.comboBoxTab.currentIndexChanged.connect(self.togglePicture)

        self.comboBoxFS.currentIndexChanged.connect(self.toggleFSList)
        self.comboBoxNgram.currentIndexChanged.connect(self.toggleFSList)

        ##############################################################
        # Function calls for processing
        ##############################################################
        self.btnInput.clicked.connect(self.import_csv)
        self.btnExecute.clicked.connect(self.toggleTab)
        self.btnExecute.clicked.connect(self.execute_classification)

    ##############################################################
    # Functions for homepage and closing application
    ##############################################################
    def on_startButton_clicked(self):
        self.stack.setCurrentIndex(1)
        self.tabWidget.setCurrentIndex(0)

    def close_application(self):
        choice = QMessageBox.question(self, 'Close application', "Quit the application?",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if choice == QMessageBox.Yes:
            sys.exit()
        else:
            pass

    ##############################################################
    # Functions for preprocessing
    ##############################################################

    def import_csv(self):
        """

        :return:
        """
        filePath, _ = QFileDialog.getOpenFileName(self, 'Open file', '/home', filter="csv(*.csv)")
        if filePath != "":
            global raw_data
            raw_data = pd.read_csv(str(filePath))
            global decontracted_raw_data
            decontracted_raw_data = self.decontract(raw_data)
            global test_series
            test_series = decontracted_raw_data.iloc[:, 0]
            global outcome_series
            outcome_series = raw_data.iloc[:, 1]
            test_list = test_series.tolist()
            test_string = str(test_list)
            test_tokens = nltk.wordpunct_tokenize(test_string)
            test_words = [word.lower() for word in test_tokens if word.isalpha()]
            global test_vocab
            test_vocab = test_words
            test_vocab_sorted = sorted(set(test_words), key=test_words.index)

            self.labelFeaturesDocument.setText('%d unique words in\nraw text douments' % (len(test_vocab_sorted)))
            self.labelFeaturesDocument.adjustSize()

            # Class Graph
            number_of_docs = len(raw_data)
            number_of_classes = len(raw_data.sentiment.value_counts())
            count_of_that_class = raw_data.sentiment.value_counts()
            probability_of_classes = count_of_that_class / number_of_docs

            class_df = pd.DataFrame()

            for i in range(0, number_of_classes):
                class_df.loc['', probability_of_classes.index[i]] = probability_of_classes[i] * 100

            my_colours = ['royalblue', 'peru', 'yellowgreen']
            Class_Graph = class_df.plot.barh(stacked=True, color=my_colours)
            Class_Graph.legend(loc='lower center', ncol=number_of_classes)
            Class_Graph.set_title('Class Distribution', fontsize=20)
            Class_Graph.axis('off')

            fig = Class_Graph.get_figure()
            fig.savefig("ClassGraph.png", bbox_inches="tight")
            self.labelClassGraph.resize(400, 300)

            pixmapclass = QPixmap('ClassGraph.png')
            self.labelClassGraph.setPixmap(pixmapclass)
            self.labelClassGraph.setScaledContents(True)

            self.labelPreprocessing.setEnabled(True)
            self.labelStopword.setEnabled(True)
            self.radioStopword.setEnabled(True)
            self.radioStopwordNone.setEnabled(True)

            return test_vocab

    def decontract(self, dataframe):
        """
        :param dataframe:
        :return:
        """
        for index, row in dataframe.iterrows():
            document = row[0]
            document = re.sub(r"won\'t", "will not", document)
            document = re.sub(r"can\'t", "can not", document)
            document = re.sub(r"n\'t", " not", document)
            document = re.sub(r"n\'s", " is", document)
            document = re.sub(r"\'re", " are", document)
            document = re.sub(r"\'d", " would", document)
            document = re.sub(r"\'ll", " will", document)
            document = re.sub(r"\'t", " not", document)
            document = re.sub(r"\'ve", " have", document)
            document = re.sub(r"\'m", " am", document)

            dataframe.iloc[index][0] = document

        return dataframe

    def stem_sentences(self, sentence):
        """

        :param sentence:
        :return:
        """
        tokens = sentence.split()
        stemmed_tokens = [porter.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)

    def lemma_sentences(self, sentence):
        """

        :param sentence:
        :return:
        """
        tokens = sentence.split()
        wnl = nltk.WordNetLemmatizer()
        lemma_tokens = [wnl.lemmatize(token, pos="v") for token in tokens]
        return ' '.join(lemma_tokens)

    ##############################################################
    # Functions for processing
    ##############################################################

    def execute_classification(self):
        """

        """
        global raw_data, test_vocab, test_series, outcome_series

        training_text = test_series
        training_text = training_text.to_frame()
        training_text['comment'] = training_text['comment'].str.lower()
        training_text['comment'] = training_text['comment'].str.replace('[^\w\s]', '')
        training_text = training_text.iloc[:, 0]

        training_outcome = outcome_series

        # Stopwords
        if self.radioStopword.isChecked():
            print('Success Stopwords')
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            not_stopwords = {'no', 'against', 'not', 'very', 'few', 'why', 'nor'}
            final_stop_words = set([word for word in stop_words if word not in not_stopwords])
            test_vocab = [word for word in test_vocab if word not in final_stop_words]

            training_text = training_text.to_frame()
            training_text['comment'] = training_text['comment'].apply(
                lambda x: ' '.join([word for word in x.split() if word not in (final_stop_words)]))
            training_text = training_text.iloc[:, 0]

        # Text Normalisation
        if self.radioStemming.isChecked():
            print('Success Stemming')
            porter = nltk.PorterStemmer()
            test_vocab = [porter.stem(t) for t in test_vocab]
            training_text = training_text.apply(self.stem_sentences)
        elif self.radioLemmatisation.isChecked():
            print('Success Lemma')
            wnl = nltk.WordNetLemmatizer()
            test_vocab = [wnl.lemmatize(t, pos="v") for t in test_vocab]
            training_text = training_text.apply(self.lemma_sentences)

        vocabulary_done = [" ".join(test_vocab)]

        # Feature Extraction
        # Presence
        if self.checkBoxPres.isChecked():
            print('Success Presence')
            if self.checkBoxUni.isChecked():
                print('Success Unigram')
                unipres_vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1, binary=True)
                unipres_vectorizer.fit(vocabulary_done)
                unipres_vector = unipres_vectorizer.transform(training_text).toarray()

            if self.checkBoxUniBi.isChecked():
                print('Success UniBi')
                unibipres_vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1, binary=True)
                unibipres_vectorizer.fit(vocabulary_done)
                unibipres_vector = unibipres_vectorizer.transform(training_text).toarray()

            if self.checkBoxUniTri.isChecked():
                print('Success UniTri')
                unipres_vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1, binary=True)
                unipres_vectorizer.fit(vocabulary_done)
                unipres_vector = unipres_vectorizer.transform(training_text).toarray()

                tripres_vectorizer = CountVectorizer(ngram_range=(3, 3), min_df=1, binary=True)
                tripres_vectorizer.fit(vocabulary_done)
                tripres_vector = tripres_vectorizer.transform(training_text).toarray()

                unitripres_vector = np.concatenate((unipres_vector, tripres_vector), axis=1)

            if self.checkBoxBi.isChecked():
                print('Success Bi')
                bipres_vectorizer = CountVectorizer(ngram_range=(2, 2), min_df=1, binary=True)
                bipres_vectorizer.fit(vocabulary_done)
                bipres_vector = bipres_vectorizer.transform(training_text).toarray()

            if self.checkBoxBiTri.isChecked():
                print('Success BiTri')
                bitripres_vectorizer = CountVectorizer(ngram_range=(2, 3), min_df=1, binary=True)
                bitripres_vectorizer.fit(vocabulary_done)
                bitripres_vector = bitripres_vectorizer.transform(training_text).toarray()

            if self.checkBoxTri.isChecked():
                print('Success Tri')
                tripres_vectorizer = CountVectorizer(ngram_range=(3, 3), min_df=1, binary=True)
                tripres_vectorizer.fit(vocabulary_done)
                tripres_vector = tripres_vectorizer.transform(training_text).toarray()

            if self.checkBoxUniBiTri.isChecked():
                print('Success UniBiTri')
                unibitripres_vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=1, binary=True)
                unibitripres_vectorizer.fit(vocabulary_done)
                unibitripres_vector = unibitripres_vectorizer.transform(training_text).toarray()

        # Count
        if self.checkBoxCount.isChecked() == True or self.checkBoxIG.isChecked() == True or self.checkBoxDF.isChecked() == True:
            print('Success Count')
            if self.checkBoxUni.isChecked():
                print('Success Unigram')
                unicount_vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1)
                unicount_vectorizer.fit(vocabulary_done)
                unicount_vector = unicount_vectorizer.transform(training_text).toarray()

            if self.checkBoxUniBi.isChecked():
                print('Success UniBi')
                unibicount_vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
                unibicount_vectorizer.fit(vocabulary_done)
                unibicount_vector = unibicount_vectorizer.transform(training_text).toarray()

            if self.checkBoxUniTri.isChecked():
                print('Success UniTri')
                unicount_vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1)
                unicount_vectorizer.fit(vocabulary_done)
                unicount_vector = unicount_vectorizer.transform(training_text).toarray()

                tricount_vectorizer = CountVectorizer(ngram_range=(3, 3), min_df=1)
                tricount_vectorizer.fit(vocabulary_done)
                tricount_vector = tricount_vectorizer.transform(training_text).toarray()

                unitricount_vector = np.concatenate((unicount_vector, tricount_vector), axis=1)

            if self.checkBoxBi.isChecked():
                print('Success Bi')
                bicount_vectorizer = CountVectorizer(ngram_range=(2, 2), min_df=1)
                bicount_vectorizer.fit(vocabulary_done)
                bicount_vector = bicount_vectorizer.transform(training_text).toarray()

            if self.checkBoxBiTri.isChecked():
                print('Success BiTri')
                bitricount_vectorizer = CountVectorizer(ngram_range=(2, 3), min_df=1)
                bitricount_vectorizer.fit(vocabulary_done)
                bitricount_vector = bitricount_vectorizer.transform(training_text).toarray()

            if self.checkBoxTri.isChecked():
                print('Success Tri')
                tricount_vectorizer = CountVectorizer(ngram_range=(3, 3), min_df=1)
                tricount_vectorizer.fit(vocabulary_done)
                tricount_vector = tricount_vectorizer.transform(training_text).toarray()

            if self.checkBoxUniBiTri.isChecked():
                print('Success UniBiTri')
                unibitricount_vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=1)
                unibitricount_vectorizer.fit(vocabulary_done)
                unibitricount_vector = unibitricount_vectorizer.transform(training_text).toarray()

        # TDIDF
        if self.checkBoxTDIDF.isChecked():
            print('Success TDIDF')

            if self.checkBoxUni.isChecked():
                print('Success Unigram')
                uniTDIDF_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
                uniTDIDF_vectorizer.fit(vocabulary_done)
                uniTDIDF_vector = uniTDIDF_vectorizer.transform(training_text).toarray()

            if self.checkBoxUniBi.isChecked():
                print('Success UniBi')
                unibiTDIDF_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
                unibiTDIDF_vectorizer.fit(vocabulary_done)
                unibiTDIDF_vector = unibiTDIDF_vectorizer.transform(training_text).toarray()

            if self.checkBoxUniTri.isChecked():
                print('Success UniTri')
                uniTDIDF_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
                uniTDIDF_vectorizer.fit(vocabulary_done)
                uniTDIDF_vector = uniTDIDF_vectorizer.transform(training_text).toarray()

                triTDIDF_vectorizer = TfidfVectorizer(ngram_range=(3, 3), min_df=1)
                triTDIDF_vectorizer.fit(vocabulary_done)
                triTDIDF_vector = triTDIDF_vectorizer.transform(training_text).toarray()

                unitriTDIDF_vector = np.concatenate((uniTDIDF_vector, triTDIDF_vector), axis=1)

            if self.checkBoxBi.isChecked():
                print('Success Bi')
                biTDIDF_vectorizer = TfidfVectorizer(ngram_range=(2, 2), min_df=1)
                biTDIDF_vectorizer.fit(vocabulary_done)
                biTDIDF_vector = biTDIDF_vectorizer.transform(training_text).toarray()

            if self.checkBoxBiTri.isChecked():
                print('Success BiTri')
                bitriTDIDF_vectorizer = TfidfVectorizer(ngram_range=(2, 3), min_df=1)
                bitriTDIDF_vectorizer.fit(vocabulary_done)
                bitriTDIDF_vector = bitriTDIDF_vectorizer.transform(training_text).toarray()

            if self.checkBoxTri.isChecked():
                print('Success Tri')
                triTDIDF_vectorizer = TfidfVectorizer(ngram_range=(3, 3), min_df=1)
                triTDIDF_vectorizer.fit(vocabulary_done)
                triTDIDF_vector = triTDIDF_vectorizer.transform(training_text).toarray()

            if self.checkBoxUniBiTri.isChecked():
                print('Success UniBiTri')
                unibitriTDIDF_vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
                unibitriTDIDF_vectorizer.fit(vocabulary_done)
                unibitriTDIDF_vector = unibitriTDIDF_vectorizer.transform(training_text).toarray()

        # Feature Selection

        global number_per_class
        global number_of_docs
        global number_of_classes
        global count_of_that_class
        global probability_of_classess
        global doc_clss_index
        number_per_class = raw_data.sentiment.value_counts()

        raw_data_num = raw_data.copy()
        raw_data_num['sentiment'] = raw_data_num['sentiment'].replace(['negative', 'positive', 'neutral'], [0, 1, 2])
        number_of_docs = len(raw_data_num)
        number_of_classes = len(raw_data_num.sentiment.value_counts())
        count_of_that_class = raw_data_num.sentiment.value_counts()
        probability_of_classess = count_of_that_class / number_of_docs
        doc_clss_index = raw_data_num.sentiment

        if self.checkBoxIG.isChecked():
            print('Success IG')

            if self.checkBoxPres.isChecked():

                if self.checkBoxUni.isChecked():
                    vocab_ig = self.performIG(unicount_vector, unicount_vectorizer)
                    uni_ig = vocab_ig[0:20, :]
                    print(uni_ig)
                    unipres_vectorizer = CountVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(1, 1), min_df=1,
                                                         binary=True)
                    unipres_vector_IG = unipres_vectorizer.transform(training_text).toarray()
                if self.checkBoxUniBi.isChecked():
                    vocab_ig = self.performIG(unibicount_vector, unibicount_vectorizer)
                    unibi_ig = vocab_ig[0:20, :]
                    # print(unibi_ig)
                    unibipres_vectorizer = CountVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(1, 2), min_df=1,
                                                           binary=True)
                    unibipres_vector_IG = unibipres_vectorizer.transform(training_text).toarray()
                    # print('unibipres')
                    # print(unibipres_vector_IG.shape)
                    # print(unibipres_vectorizer.get_feature_names())
                    # print(unibipres_vector_IG)
                if self.checkBoxUniTri.isChecked():
                    vocab_ig = self.performIG(unitricount_vector, unicount_vectorizer, tricount_vectorizer)
                    unitri_ig = vocab_ig[0:20, :]
                    unitripres_vectorizer = CountVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(1, 3), min_df=1,
                                                            binary=True)
                    unitripres_vector_IG = unitripres_vectorizer.transform(training_text).toarray()
                    # print(unitripres_vector_IG.shape)
                    # print(unitripres_vectorizer.get_feature_names())
                    # print(unitripres_vector_IG)
                if self.checkBoxBi.isChecked():
                    vocab_ig = self.performIG(bicount_vector, bicount_vectorizer)
                    bi_ig = vocab_ig[0:20, :]
                    bipres_vectorizer = CountVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(2, 2), min_df=1,
                                                        binary=True)
                    bipres_vector_IG = bipres_vectorizer.transform(training_text).toarray()
                if self.checkBoxBiTri.isChecked():
                    vocab_ig = self.performIG(bitricount_vector, bitricount_vectorizer)
                    bitri_ig = vocab_ig[0:20, :]
                    bitripres_vectorizer = CountVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(2, 3), min_df=1,
                                                           binary=True)
                    bitripres_vector_IG = bitripres_vectorizer.transform(training_text).toarray()
                if self.checkBoxTri.isChecked():
                    vocab_ig = self.performIG(tricount_vector, tricount_vectorizer)
                    tri_ig = vocab_ig[0:20, :]
                    tripres_vectorizer = CountVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(3, 3), min_df=1,
                                                         binary=True)
                    tripres_vector_IG = tripres_vectorizer.transform(training_text).toarray()
                if self.checkBoxUniBiTri.isChecked():
                    vocab_ig = self.performIG(unibitricount_vector, unibitricount_vectorizer)
                    unibitri_ig = vocab_ig[0:20, :]
                    unibitripres_vectorizer = CountVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(1, 3), min_df=1,
                                                              binary=True)
                    unibitripres_vector_IG = unibitripres_vectorizer.transform(training_text).toarray()
                    # print(unibitripres_vector_IG.shape)
                    # print(unibitripres_vectorizer.get_feature_names())
                    # print(unibitripres_vector_IG)

            if self.checkBoxCount.isChecked():

                if self.checkBoxUni.isChecked():
                    vocab_ig = self.performIG(unicount_vector, unicount_vectorizer)
                    uni_ig = vocab_ig[0:20, :]
                    unicount_vectorizer = CountVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(1, 1), min_df=1)
                    unicount_vector_IG = unicount_vectorizer.transform(training_text).toarray()
                if self.checkBoxUniBi.isChecked():
                    vocab_ig = self.performIG(unibicount_vector, unibicount_vectorizer)
                    unibi_ig = vocab_ig[0:20, :]
                    unibicount_vectorizer = CountVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(1, 2), min_df=1)
                    unibicount_vector_IG = unibicount_vectorizer.transform(training_text).toarray()
                if self.checkBoxUniTri.isChecked():
                    vocab_ig = self.performIG(unitricount_vector, unicount_vectorizer, tricount_vectorizer)
                    unitri_ig = vocab_ig[0:20, :]
                    unitricount_vectorizer = CountVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(1, 3), min_df=1)
                    unitricount_vector_IG = unitricount_vectorizer.transform(training_text).toarray()
                if self.checkBoxBi.isChecked():
                    vocab_ig = self.performIG(bicount_vector, bicount_vectorizer)
                    bi_ig = vocab_ig[0:20, :]
                    bicount_vectorizer = CountVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(2, 2), min_df=1)
                    bicount_vector_IG = bicount_vectorizer.transform(training_text).toarray()
                if self.checkBoxBiTri.isChecked():
                    vocab_ig = self.performIG(bitricount_vector, bitricount_vectorizer)
                    bitri_ig = vocab_ig[0:20, :]
                    bitricount_vectorizer = CountVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(2, 3), min_df=1)
                    bitricount_vector_IG = bitricount_vectorizer.transform(training_text).toarray()
                if self.checkBoxTri.isChecked():
                    vocab_ig = self.performIG(tricount_vector, tricount_vectorizer)
                    tri_ig = vocab_ig[0:20, :]
                    tricount_vectorizer = CountVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(3, 3), min_df=1)
                    tricount_vector_IG = tricount_vectorizer.transform(training_text).toarray()
                if self.checkBoxUniBiTri.isChecked():
                    vocab_ig = self.performIG(unibitricount_vector, unibitricount_vectorizer)
                    unibitri_ig = vocab_ig[0:20, :]
                    unibitricount_vectorizer = CountVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(1, 3), min_df=1)
                    unibitricount_vector_IG = unibitricount_vectorizer.transform(training_text).toarray()

            if self.checkBoxTDIDF.isChecked():

                if self.checkBoxUni.isChecked():
                    vocab_ig = self.performIG(unicount_vector, unicount_vectorizer)
                    uni_ig = vocab_ig[0:20, :]
                    uniTDIDF_vectorizer = TfidfVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(1, 1), min_df=1)
                    uniTDIDF_vectorizer.fit(vocab_ig[:, 1])
                    uniTDIDF_vector_IG = uniTDIDF_vectorizer.transform(training_text).toarray()

                if self.checkBoxUniBi.isChecked():
                    vocab_ig = self.performIG(unibicount_vector, unibicount_vectorizer)
                    unibi_ig = vocab_ig[0:20, :]
                    unibiTDIDF_vectorizer = TfidfVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(1, 2), min_df=1)
                    unibiTDIDF_vectorizer.fit(vocab_ig[:, 1])
                    unibiTDIDF_vector_IG = unibiTDIDF_vectorizer.transform(training_text).toarray()

                if self.checkBoxUniTri.isChecked():
                    vocab_ig = self.performIG(unitricount_vector, unicount_vectorizer, tricount_vectorizer)
                    unitri_ig = vocab_ig[0:20, :]
                    unitriTDIDF_vectorizer = TfidfVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(1, 3), min_df=1)
                    unitriTDIDF_vectorizer.fit(vocab_ig[:, 1])
                    unitriTDIDF_vector_IG = unitriTDIDF_vectorizer.transform(training_text).toarray()

                if self.checkBoxBi.isChecked():
                    vocab_ig = self.performIG(bicount_vector, bicount_vectorizer)
                    bi_ig = vocab_ig[0:20, :]
                    biTDIDF_vectorizer = TfidfVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(2, 2), min_df=1)
                    biTDIDF_vectorizer.fit(vocab_ig[:, 1])
                    biTDIDF_vector_IG = biTDIDF_vectorizer.transform(training_text).toarray()

                if self.checkBoxBiTri.isChecked():
                    vocab_ig = self.performIG(bitricount_vector, bitricount_vectorizer)
                    bitri_ig = vocab_ig[0:20, :]
                    bitriTDIDF_vectorizer = TfidfVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(2, 3), min_df=1)
                    bitriTDIDF_vectorizer.fit(vocab_ig[:, 1])
                    bitriTDIDF_vector_IG = bitriTDIDF_vectorizer.transform(training_text).toarray()

                if self.checkBoxTri.isChecked():
                    vocab_ig = self.performIG(tricount_vector, tricount_vectorizer)
                    tri_ig = vocab_ig[0:20, :]
                    triTDIDF_vectorizer = TfidfVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(3, 3), min_df=1)
                    triTDIDF_vectorizer.fit(vocab_ig[:, 1])
                    triTDIDF_vector_IG = triTDIDF_vectorizer.transform(training_text).toarray()

                if self.checkBoxUniBiTri.isChecked():
                    vocab_ig = self.performIG(unibitricount_vector, unibitricount_vectorizer)
                    unibitri_ig = vocab_ig[0:20, :]
                    unibitriTDIDF_vectorizer = TfidfVectorizer(vocabulary=vocab_ig[:, 1], ngram_range=(1, 3), min_df=1)
                    unibitriTDIDF_vectorizer.fit(vocab_ig[:, 1])
                    unibitriTDIDF_vector_IG = unibitriTDIDF_vectorizer.transform(training_text).toarray()

        if self.checkBoxDF.isChecked():
            print("Success DF")
            if self.checkBoxPres.isChecked():

                if self.checkBoxUni.isChecked():
                    vocab_df = self.performDF(unicount_vector, unicount_vectorizer)
                    uni_df = vocab_df[0:20, :]
                    unipres_vectorizer = CountVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(1, 1), min_df=1,
                                                         binary=True)
                    unipres_vector_DF = unipres_vectorizer.transform(training_text).toarray()
                if self.checkBoxUniBi.isChecked():
                    vocab_df = self.performDF(unibicount_vector, unibicount_vectorizer)
                    unibi_df = vocab_df[0:20, :]
                    unibipres_vectorizer = CountVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(1, 1), min_df=1,
                                                           binary=True)
                    unibipres_vector_DF = unibipres_vectorizer.transform(training_text).toarray()
                if self.checkBoxUniTri.isChecked():
                    vocab_df = self.performDF(unitricount_vector, unicount_vectorizer, tricount_vectorizer)
                    unitri_df = vocab_df[0:20, :]
                    unitripres_vectorizer = CountVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(1, 3), min_df=1,
                                                            binary=True)
                    unitripres_vector_DF = unitripres_vectorizer.transform(training_text).toarray()
                if self.checkBoxBi.isChecked():
                    vocab_df = self.performDF(bicount_vector, bicount_vectorizer)
                    bi_df = vocab_df[0:20, :]
                    bipres_vectorizer = CountVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(2, 2), min_df=1,
                                                        binary=True)
                    bipres_vector_DF = bipres_vectorizer.transform(training_text).toarray()
                if self.checkBoxBiTri.isChecked():
                    vocab_df = self.performDF(bitricount_vector, bitricount_vectorizer)
                    bitri_df = vocab_df[0:20, :]
                    bitripres_vectorizer = CountVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(2, 3), min_df=1,
                                                           binary=True)
                    bitripres_vector_DF = bitripres_vectorizer.transform(training_text).toarray()
                if self.checkBoxTri.isChecked():
                    vocab_df = self.performDF(tricount_vector, tricount_vectorizer)
                    tri_df = vocab_df[0:20, :]
                    tripres_vectorizer = CountVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(3, 3), min_df=1,
                                                         binary=True)
                    tripres_vector_DF = tripres_vectorizer.transform(training_text).toarray()
                    # print(np.sum(~tripres_vector.any(1)))
                if self.checkBoxUniBiTri.isChecked():
                    vocab_df = self.performDF(unibitricount_vector, unibitricount_vectorizer)
                    unibitri_df = vocab_df[0:20, :]
                    unibitripres_vectorizer = CountVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(1, 1), min_df=1,
                                                              binary=True)
                    unibitripres_vector_DF = unibitripres_vectorizer.transform(training_text).toarray()

            if self.checkBoxCount.isChecked():

                if self.checkBoxUni.isChecked():
                    vocab_df = self.performDF(unicount_vector, unicount_vectorizer)
                    uni_df = vocab_df[0:20, :]
                    unicount_vectorizer = CountVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(1, 1), min_df=1)
                    unicount_vector_DF = unicount_vectorizer.transform(training_text).toarray()
                if self.checkBoxUniBi.isChecked():
                    vocab_df = self.performDF(unibicount_vector, unibicount_vectorizer)
                    unibi_df = vocab_df[0:20, :]
                    unibicount_vectorizer = CountVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(1, 1), min_df=1)
                    unibicount_vector_DF = unibicount_vectorizer.transform(training_text).toarray()
                if self.checkBoxUniTri.isChecked():
                    vocab_df = self.performDF(unitricount_vector, unicount_vectorizer, tricount_vectorizer)
                    unitri_df = vocab_df[0:20, :]
                    unitricount_vectorizer = CountVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(1, 1), min_df=1)
                    unitricount_vector_DF = unitricount_vectorizer.transform(training_text).toarray()
                if self.checkBoxBi.isChecked():
                    vocab_df = self.performDF(bicount_vector, bicount_vectorizer)
                    bi_df = vocab_df[0:20, :]
                    bicount_vectorizer = CountVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(2, 2), min_df=1)
                    bicount_vector_DF = bicount_vectorizer.transform(training_text).toarray()
                if self.checkBoxBiTri.isChecked():
                    vocab_df = self.performDF(bitricount_vector, bitricount_vectorizer)
                    bitri_df = vocab_df[0:20, :]
                    bitricount_vectorizer = CountVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(2, 3), min_df=1)
                    bitricount_vector_DF = bitricount_vectorizer.transform(training_text).toarray()
                if self.checkBoxTri.isChecked():
                    vocab_df = self.performDF(tricount_vector, tricount_vectorizer)
                    tri_df = vocab_df[0:20, :]
                    tricount_vectorizer = CountVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(3, 3), min_df=1)
                    tricount_vector_DF = tricount_vectorizer.transform(training_text).toarray()
                if self.checkBoxUniBiTri.isChecked():
                    vocab_df = self.performDF(unibitricount_vector, unibitricount_vectorizer)
                    unibitri_df = vocab_df[0:20, :]
                    unibitricount_vectorizer = CountVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(1, 1), min_df=1)
                    unibitricount_vector_DF = unibitricount_vectorizer.transform(training_text).toarray()

            if self.checkBoxTDIDF.isChecked():

                if self.checkBoxUni.isChecked():
                    vocab_df = self.performDF(unicount_vector, unicount_vectorizer)
                    uni_df = vocab_df[0:20, :]
                    uniTDIDF_vectorizer = TfidfVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(1, 1), min_df=1)
                    uniTDIDF_vectorizer.fit(vocab_df[:, 1])
                    uniTDIDF_vector_DF = uniTDIDF_vectorizer.transform(training_text).toarray()

                if self.checkBoxUniBi.isChecked():
                    vocab_df = self.performDF(unibicount_vector, unibicount_vectorizer)
                    unibi_df = vocab_df[0:20, :]
                    unibiTDIDF_vectorizer = TfidfVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(1, 1), min_df=1)
                    unibiTDIDF_vectorizer.fit(vocab_df[:, 1])
                    unibiTDIDF_vector_DF = unibiTDIDF_vectorizer.transform(training_text).toarray()

                if self.checkBoxUniTri.isChecked():
                    vocab_df = self.performDF(unitricount_vector, unicount_vectorizer, tricount_vectorizer)
                    unitri_df = vocab_df[0:20, :]
                    unitriTDIDF_vectorizer = TfidfVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(1, 1), min_df=1)
                    unitriTDIDF_vectorizer.fit(vocab_df[:, 1])
                    unitriTDIDF_vector_DF = unitriTDIDF_vectorizer.transform(training_text).toarray()

                if self.checkBoxBi.isChecked():
                    vocab_df = self.performDF(bicount_vector, bicount_vectorizer)
                    bi_df = vocab_df[0:20, :]
                    biTDIDF_vectorizer = TfidfVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(2, 2), min_df=1)
                    biTDIDF_vectorizer.fit(vocab_df[:, 1])
                    biTDIDF_vector_DF = biTDIDF_vectorizer.transform(training_text).toarray()

                if self.checkBoxBiTri.isChecked():
                    vocab_df = self.performDF(bitricount_vector, bitricount_vectorizer)
                    bitri_df = vocab_df[0:20, :]
                    bitriTDIDF_vectorizer = TfidfVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(2, 3), min_df=1)
                    bitriTDIDF_vectorizer.fit(vocab_df[:, 1])
                    bitriTDIDF_vector_DF = bitriTDIDF_vectorizer.transform(training_text).toarray()

                if self.checkBoxTri.isChecked():
                    vocab_df = self.performDF(tricount_vector, tricount_vectorizer)
                    tri_df = vocab_df[0:20, :]
                    triTDIDF_vectorizer = TfidfVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(3, 3), min_df=1)
                    triTDIDF_vectorizer.fit(vocab_df[:, 1])
                    triTDIDF_vector_DF = triTDIDF_vectorizer.transform(training_text).toarray()

                if self.checkBoxUniBiTri.isChecked():
                    vocab_df = self.performDF(unibitricount_vector, unibitricount_vectorizer)
                    unibitri_df = vocab_df[0:20, :]
                    unibitriTDIDF_vectorizer = TfidfVectorizer(vocabulary=vocab_df[:, 1], ngram_range=(1, 1), min_df=1)
                    unibitriTDIDF_vectorizer.fit(vocab_df[:, 1])
                    unibitriTDIDF_vector_DF = unibitriTDIDF_vectorizer.transform(training_text).toarray()

        # ML algorithms

        if self.comboBoxFolds.currentIndex() == 1:
            n_folds = 5
        if self.comboBoxFolds.currentIndex() == 2:
            n_folds = 10

        if self.comboBoxTune.currentIndex() == 1:
            tune_metric = 'balanced_accuracy'
        if self.comboBoxTune.currentIndex() == 2:
            tune_metric = 'precision_weighted'
        if self.comboBoxTune.currentIndex() == 3:
            tune_metric = 'recall_weighted'
        if self.comboBoxTune.currentIndex() == 4:
            tune_metric = 'f1_weighted'

        # Evaluation
        if self.comboBoxEvaluation.currentIndex() == 1:
            y_label = 'Accuracy'
        if self.comboBoxEvaluation.currentIndex() == 2:
            y_label = 'Precision'
        if self.comboBoxEvaluation.currentIndex() == 3:
            y_label = 'Recall'
        if self.comboBoxEvaluation.currentIndex() == 4:
            y_label = 'F-Score'
        if self.comboBoxEvaluation.currentIndex() == 5:
            roc_auc_training_outcome = training_outcome
            y_label = 'ROC AUC Score'

        # Performance

        col_names = ['Data Representations', 'Fold Results', 'Classifier']
        performancePres = pd.DataFrame(columns=col_names)
        performanceCount = pd.DataFrame(columns=col_names)
        performanceTDIDF = pd.DataFrame(columns=col_names)
        resultsDF = pd.DataFrame(columns=col_names)

        if self.checkBoxSVM.isChecked():
            SVM = SVC(probability=True)
            SVMROC = OneVsRestClassifier(SVM)

            if self.comboBoxEvaluation.currentIndex() != 5:
                parametersSVM = [
                    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                    {'C': [1, 10, 100, 1000], 'kernel': ['rbf']},
                    {'C': [1, 10, 100, 1000], 'degree': [2, 3, 4, 5], 'kernel': ['poly']}
                ]
            if self.comboBoxEvaluation.currentIndex() == 5:
                parametersSVM = [
                    {'estimator__C': [1, 10, 100, 1000], 'estimator__kernel': ['linear']},
                    {'estimator__C': [1, 10, 100, 1000], 'estimator__kernel': ['rbf']},
                    {'estimator__C': [1, 10, 100, 1000], 'estimator__degree': [2, 3, 4, 5],
                     'estimator__kernel': ['poly']}
                ]

            print('Success SVM')

            # Uni
            if self.checkBoxUni.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unipres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)

                    if self.checkBoxCount.isChecked():
                        training_featurevector = unicount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)

                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = uniTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unipres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)

                    if self.checkBoxCount.isChecked():
                        training_featurevector = unicount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)

                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = uniTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unipres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)

                    if self.checkBoxCount.isChecked():
                        training_featurevector = unicount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)

                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = uniTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # UniBi
            if self.checkBoxUniBi.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibipres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)

                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibicount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibiTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibipres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibicount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibiTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibipres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibicount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibiTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # UniTri
            if self.checkBoxUniTri.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unitripres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unitricount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unitriTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unitripres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unitricount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unitriTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unitripres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unitricount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unitriTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # Bi
            if self.checkBoxBi.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bipres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bicount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = biTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bipres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bicount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = biTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bipres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bicount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = biTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # BiTri
            if self.checkBoxBiTri.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bitripres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bitricount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = bitriTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bitripres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bitricount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = bitriTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bitripres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bitricount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = bitriTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # Tri
            if self.checkBoxTri.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = tripres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = tricount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = triTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = tripres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = tricount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = triTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = tripres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = tricount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = triTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # UniBiTri
            if self.checkBoxUniBiTri.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibitripres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibitricount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibitriTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibitripres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibitricount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibitriTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (IG)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (IG)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibitripres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibitricount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibitriTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (DF)', 'SVM', SVM, parametersSVM,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (DF)', 'SVM', SVMROC, parametersSVM,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

        if self.checkBoxNB.isChecked():
            print('Success NB')
            NB_pres = BernoulliNB()
            NB_count = MultinomialNB()
            NB_TFIDF = GaussianNB()

            NBROC_pres = OneVsRestClassifier(NB_pres)
            NBROC_count = OneVsRestClassifier(NB_count)
            NBROC_TFIDF = OneVsRestClassifier(NB_TFIDF)

            if self.comboBoxEvaluation.currentIndex() != 5:
                parametersNB_pres = [{'alpha': np.linspace(0.5, 1.5, 6), 'fit_prior': [True, False]}]
                parametersNB_count = [{'alpha': np.linspace(0.5, 1.5, 6), 'fit_prior': [True, False]}]
                parametersNB_TFIDF = [{'var_smoothing': [1e-06]}]
            if self.comboBoxEvaluation.currentIndex() == 5:
                parametersNB_pres = [
                    {'estimator__alpha': np.linspace(0.5, 1.5, 6), 'estimator__fit_prior': [True, False]}]
                parametersNB_count = [
                    {'estimator__alpha': np.linspace(0.5, 1.5, 6), 'estimator__fit_prior': [True, False]}]
                parametersNB_TFIDF = [{'estimator__var_smoothing': [1e-09]}]

            # Uni
            if self.checkBoxUni.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unipres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)

                    if self.checkBoxCount.isChecked():
                        training_featurevector = unicount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram', 'NB', NB_count, parametersNB_count,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram', 'NB', NBROC_count, parametersNB_count,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceCount = performanceCount.append(resultsDF)

                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = uniTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram', 'NB', NB_TFIDF, parametersNB_TFIDF,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram', 'NB', NBROC_TFIDF, parametersNB_TFIDF,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unipres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (IG)', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (IG)', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)

                    if self.checkBoxCount.isChecked():
                        training_featurevector = unicount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (IG)', 'NB', NB_count, parametersNB_count,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (IG)', 'NB', NBROC_count, parametersNB_count,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceCount = performanceCount.append(resultsDF)

                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = uniTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (IG)', 'NB', NB_TFIDF, parametersNB_TFIDF,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (IG)', 'NB', NBROC_TFIDF, parametersNB_TFIDF,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unipres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (DF)', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (DF)', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)

                    if self.checkBoxCount.isChecked():
                        training_featurevector = unicount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (DF)', 'NB', NB_count, parametersNB_count,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (DF)', 'NB', NBROC_count, parametersNB_count,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceCount = performanceCount.append(resultsDF)

                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = uniTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (DF)', 'NB', NB_TFIDF, parametersNB_TFIDF,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (DF)', 'NB', NBROC_TFIDF, parametersNB_TFIDF,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # UniBi
            if self.checkBoxUniBi.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibipres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibicount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram', 'NB', NB_count, parametersNB_count,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram', 'NB', NBROC_count, parametersNB_count,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibiTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram', 'NB', NB_TFIDF, parametersNB_TFIDF,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram', 'NB', NBROC_TFIDF, parametersNB_TFIDF,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibipres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (IG)', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (IG)', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibicount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (IG)', 'NB', NB_count, parametersNB_count,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (IG)', 'NB', NBROC_count, parametersNB_count,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibiTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (IG)', 'NB', NB_TFIDF, parametersNB_TFIDF,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (IG)', 'NB', NBROC_TFIDF, parametersNB_TFIDF,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibipres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (DF)', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (DF)', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibicount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (DF)', 'NB', NB_count, parametersNB_count,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (DF)', 'NB', NBROC_count, parametersNB_count,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibiTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (DF)', 'NB', NB_TFIDF, parametersNB_TFIDF,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (DF)', 'NB', NBROC_TFIDF, parametersNB_TFIDF,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # UniTri
            if self.checkBoxUniTri.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unitripres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unitricount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram', 'NB', NB_count, parametersNB_count,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram', 'NB', NBROC_count, parametersNB_count,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unitriTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram', 'NB', NB_TFIDF, parametersNB_TFIDF,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram', 'NB', NBROC_TFIDF, parametersNB_TFIDF,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unitripres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (IG)', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (IG)', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unitricount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (IG)', 'NB', NB_count,
                                                                   parametersNB_count, training_featurevector,
                                                                   training_outcome, n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (IG)', 'NB', NBROC_count,
                                                                parametersNB_count, training_featurevector,
                                                                training_outcome, n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unitriTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (IG)', 'NB', NB_TFIDF,
                                                                   parametersNB_TFIDF, training_featurevector,
                                                                   training_outcome, n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (IG)', 'NB', NBROC_TFIDF,
                                                                parametersNB_TFIDF, training_featurevector,
                                                                training_outcome, n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unitripres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (DF)', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (DF)', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unitricount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (DF)', 'NB', NB_count,
                                                                   parametersNB_count, training_featurevector,
                                                                   training_outcome, n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (DF)', 'NB', NBROC_count,
                                                                parametersNB_count, training_featurevector,
                                                                training_outcome, n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unitriTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (DF)', 'NB', NB_TFIDF,
                                                                   parametersNB_TFIDF, training_featurevector,
                                                                   training_outcome, n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (DF)', 'NB', NBROC_TFIDF,
                                                                parametersNB_TFIDF, training_featurevector,
                                                                training_outcome, n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # Bi
            if self.checkBoxBi.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bipres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bicount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram', 'NB', NB_count, parametersNB_count,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram', 'NB', NBROC_count, parametersNB_count,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = biTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram', 'NB', NB_TFIDF, parametersNB_TFIDF,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram', 'NB', NBROC_TFIDF, parametersNB_TFIDF,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bipres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (IG)', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (IG)', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bicount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (IG)', 'NB', NB_count, parametersNB_count,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (IG)', 'NB', NBROC_count, parametersNB_count,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = biTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (IG)', 'NB', NB_TFIDF, parametersNB_TFIDF,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (IG)', 'NB', NBROC_TFIDF, parametersNB_TFIDF,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bipres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (DF)', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (DF)', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bicount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (DF)', 'NB', NB_count, parametersNB_count,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (DF)', 'NB', NBROC_count, parametersNB_count,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = biTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (DF)', 'NB', NB_TFIDF, parametersNB_TFIDF,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (DF)', 'NB', NBROC_TFIDF, parametersNB_TFIDF,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # BiTri
            if self.checkBoxBiTri.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bitripres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bitricount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram', 'NB', NB_count, parametersNB_count,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram', 'NB', NBROC_count, parametersNB_count,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = bitriTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram', 'NB', NB_TFIDF, parametersNB_TFIDF,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram', 'NB', NBROC_TFIDF, parametersNB_TFIDF,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bitripres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (IG)', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (IG)', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bitricount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (IG)', 'NB', NB_count, parametersNB_count,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (IG)', 'NB', NBROC_count, parametersNB_count,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = bitriTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (IG)', 'NB', NB_TFIDF, parametersNB_TFIDF,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (IG)', 'NB', NBROC_TFIDF, parametersNB_TFIDF,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bitripres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (DF)', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (DF)', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bitricount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (DF)', 'NB', NB_count, parametersNB_count,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (DF)', 'NB', NBROC_count, parametersNB_count,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = bitriTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (DF)', 'NB', NB_TFIDF, parametersNB_TFIDF,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (DF)', 'NB', NBROC_TFIDF, parametersNB_TFIDF,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # Tri
            if self.checkBoxTri.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = tripres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = tricount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram', 'NB', NB_count, parametersNB_count,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram', 'NB', NBROC_count, parametersNB_count,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = triTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram', 'NB', NB_TFIDF, parametersNB_TFIDF,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram', 'NB', NBROC_TFIDF, parametersNB_TFIDF,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = tripres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (IG)', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram (IG)', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = tricount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (IG)', 'NB', NB_count, parametersNB_count,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram (IG)', 'NB', NBROC_count, parametersNB_count,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = triTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (IG)', 'NB', NB_TFIDF, parametersNB_TFIDF,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram (IG)', 'NB', NBROC_TFIDF, parametersNB_TFIDF,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = tripres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (DF)', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram (DF)', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = tricount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (DF)', 'NB', NB_count, parametersNB_count,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Triigram (DF)', 'NB', NBROC_count, parametersNB_count,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = triTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (DF)', 'NB', NB_TFIDF, parametersNB_TFIDF,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram (DF)', 'NB', NBROC_TFIDF, parametersNB_TFIDF,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # UniBiTri
            if self.checkBoxUniBiTri.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibitripres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram', 'NB', NB_pres, parametersNB_pres,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram', 'NB', NBROC_pres, parametersNB_pres,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibitricount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram', 'NB', NB_count, parametersNB_count,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram', 'NB', NBROC_count, parametersNB_count,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibitriTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram', 'NB', NB_TFIDF, parametersNB_TFIDF,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram', 'NB', NBROC_TFIDF, parametersNB_TFIDF,
                                                                training_featurevector, training_outcome, n_folds,
                                                                tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibitripres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (IG)', 'NB', NB_pres,
                                                                   parametersNB_pres, training_featurevector,
                                                                   training_outcome, n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (IG)', 'NB', NBROC_pres,
                                                                parametersNB_pres, training_featurevector,
                                                                training_outcome, n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibitricount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (IG)', 'NB', NB_count,
                                                                   parametersNB_count, training_featurevector,
                                                                   training_outcome, n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (IG)', 'NB', NBROC_count,
                                                                parametersNB_count, training_featurevector,
                                                                training_outcome, n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibitriTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (IG)', 'NB', NB_TFIDF,
                                                                   parametersNB_TFIDF, training_featurevector,
                                                                   training_outcome, n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (IG)', 'NB', NBROC_TFIDF,
                                                                parametersNB_TFIDF, training_featurevector,
                                                                training_outcome, n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibitripres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (DF)', 'NB', NB_pres,
                                                                   parametersNB_pres, training_featurevector,
                                                                   training_outcome, n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (DF)', 'NB', NBROC_pres,
                                                                parametersNB_pres, training_featurevector,
                                                                training_outcome, n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibitricount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (DF)', 'NB', NB_count,
                                                                   parametersNB_count, training_featurevector,
                                                                   training_outcome, n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (DF)', 'NB', NBROC_count,
                                                                parametersNB_count, training_featurevector,
                                                                training_outcome, n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibitriTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (DF)', 'NB', NB_TFIDF,
                                                                   parametersNB_TFIDF, training_featurevector,
                                                                   training_outcome, n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (DF)', 'NB', NBROC_TFIDF,
                                                                parametersNB_TFIDF, training_featurevector,
                                                                training_outcome, n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

        if self.checkBoxME.isChecked():
            ME = LogisticRegression()
            MEROC = OneVsRestClassifier(ME)
            if self.comboBoxEvaluation.currentIndex() != 5:
                parametersME = [
                    {'solver': ['newton-cg', 'sag', 'lbfgs', 'saga'], 'penalty': ['l2'], 'C': [1, 10, 100, 1000],
                     'multi_class': ['auto'], 'max_iter': [150]},
                    {'solver': ['liblinear'], 'penalty': ['l1'], 'C': [1, 10, 100, 1000], 'multi_class': ['auto'],
                     'max_iter': [150]}
                ]
            if self.comboBoxEvaluation.currentIndex() == 5:
                parametersME = [
                    {'estimator__solver': ['newton-cg', 'sag', 'lbfgs', 'saga'], 'estimator__penalty': ['l2'],
                     'estimator__C': [1, 10, 100, 1000], 'estimator__multi_class': ['auto'],
                     'estimator__max_iter': [150]},
                    {'estimator__solver': ['liblinear'], 'estimator__penalty': ['l1'],
                     'estimator__C': [1, 10, 100, 1000], 'estimator__multi_class': ['auto'],
                     'estimator__max_iter': [150]}
                ]

            print('Success ME')

            # Uni
            if self.checkBoxUni.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unipres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)

                    if self.checkBoxCount.isChecked():
                        training_featurevector = unicount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)

                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = uniTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unipres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)

                    if self.checkBoxCount.isChecked():
                        training_featurevector = unicount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)

                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = uniTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unipres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)

                    if self.checkBoxCount.isChecked():
                        training_featurevector = unicount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)

                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = uniTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Unigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Unigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # UniBi
            if self.checkBoxUniBi.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibipres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibicount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibiTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibipres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibicount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibiTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibipres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibicount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibiTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # UniTri
            if self.checkBoxUniTri.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unitripres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unitricount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unitriTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unitripres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unitricount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unitriTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unitripres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unitricount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unitriTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniTrigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniTrigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # Bi
            if self.checkBoxBi.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bipres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bicount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = biTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bipres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bicount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = biTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bipres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bicount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = biTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Bigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Bigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # BiTri
            if self.checkBoxBiTri.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bitripres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bitricount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = bitriTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bitripres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bitricount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = bitriTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = bitripres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = bitricount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = bitriTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('BiTrigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('BiTrigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # Tri
            if self.checkBoxTri.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = tripres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = tricount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Triigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = triTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = tripres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Triigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = tricount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = triTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = tripres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = tricount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = triTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('Trigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('Trigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

            # UniBiTri
            if self.checkBoxUniBiTri.isChecked():
                if self.checkBoxFSNone.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibitripres_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibitricount_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibitriTDIDF_vector
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxIG.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibitripres_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibitricount_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibitriTDIDF_vector_IG
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (IG)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (IG)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

                if self.checkBoxDF.isChecked():
                    if self.checkBoxPres.isChecked():
                        training_featurevector = unibitripres_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performancePres = performancePres.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performancePres = performancePres.append(resultsDF)
                    if self.checkBoxCount.isChecked():
                        training_featurevector = unibitricount_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceCount = performanceCount.append(resultsDF)
                    if self.checkBoxTDIDF.isChecked():
                        training_featurevector = unibitriTDIDF_vector_DF
                        if self.comboBoxEvaluation.currentIndex() != 5:
                            resultsDF = self.CrossValidationNormal('UniBiTrigram (DF)', 'ME', ME, parametersME,
                                                                   training_featurevector, training_outcome, n_folds,
                                                                   tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)
                        elif self.comboBoxEvaluation.currentIndex() == 5:
                            resultsDF = self.CrossValidationAUC('UniBiTrigram (DF)', 'ME', MEROC, parametersME,
                                                                training_featurevector, roc_auc_training_outcome,
                                                                n_folds, tune_metric)
                            performanceTDIDF = performanceTDIDF.append(resultsDF)

        export_csv_pres = performancePres.to_csv(
            r'C:\Users\Jurie\Documents\Akademie\Jaar 4\Skripsie\Python Course\Presence_Dataframe.csv', header=True)
        export_csv_count = performanceCount.to_csv(
            r'C:\Users\Jurie\Documents\Akademie\Jaar 4\Skripsie\Python Course\Count_Dataframe.csv', header=True)
        export_csv_TFIDF = performanceTDIDF.to_csv(
            r'C:\Users\Jurie\Documents\Akademie\Jaar 4\Skripsie\Python Course\tfidf_Dataframe.csv', header=True)

        self.comboBoxTab.clear()

        self.comboBoxTab.addItem('')
        if self.checkBoxPres.isChecked():
            self.comboBoxTab.addItem('Presence Vectorisation')
        if self.checkBoxCount.isChecked():
            self.comboBoxTab.addItem('Count Vectorisation')
        if self.checkBoxTDIDF.isChecked():
            self.comboBoxTab.addItem('TFIDF Vectorisation')
        self.comboBoxTab.adjustSize()
        self.comboBoxTab.setEnabled(True)

        self.comboBoxFS.clear()

        self.comboBoxFS.addItem('')
        if self.checkBoxIG.isChecked():
            self.comboBoxFS.addItem('Information Gain')
        if self.checkBoxDF.isChecked():
            self.comboBoxFS.addItem('Document Frequency')
        self.comboBoxFS.adjustSize()
        self.comboBoxFS.setEnabled(True)

        from pathlib import Path
        import os
        self.comboBoxNgram.clear()

        self.comboBoxNgram.addItem('')
        if self.checkBoxIG.isChecked():
            if self.checkBoxUni.isChecked():
                self.comboBoxNgram.addItem('Unigram')
                path_instance = Path("uni_words_ig.png")
                if path_instance.is_file():
                    os.remove("uni_words_ig.png")
                uni_ig = pd.DataFrame(uni_ig)
                uni_ig.columns = ['IG', 'Words']
                uni_ig['IG'] = uni_ig['IG'].astype(float)
                uni_words_ig = sns.barplot(x='IG', y='Words', data=uni_ig, color=(0.2, 0.4, 0.6, 0.6))
                uni_words_ig = uni_words_ig.get_figure()
                uni_words_ig.savefig("uni_words_ig.png", bbox_inches="tight")
            if self.checkBoxBi.isChecked():
                self.comboBoxNgram.addItem('Bigram')
                bi_ig = pd.DataFrame(bi_ig)
                bi_ig.columns = ['IG', 'Words']
                bi_ig['IG'] = bi_ig['IG'].astype(float)
                bi_words_ig = sns.barplot(x='IG', y='Words', data=bi_ig, color=(0.2, 0.4, 0.6, 0.6))
                bi_words_ig = bi_words_ig.get_figure()
                bi_words_ig.savefig("bi_words_ig.png", bbox_inches="tight")
            if self.checkBoxTri.isChecked():
                self.comboBoxNgram.addItem('Trigram')
                tri_ig = pd.DataFrame(tri_ig)
                tri_ig.columns = ['IG', 'Words']
                tri_ig['IG'] = tri_ig['IG'].astype(float)
                tri_words_ig = sns.barplot(x='IG', y='Words', data=tri_ig, color=(0.2, 0.4, 0.6, 0.6))
                tri_words_ig = tri_words_ig.get_figure()
                tri_words_ig.savefig("tri_words_ig.png", bbox_inches="tight")
            if self.checkBoxUniBi.isChecked():
                self.comboBoxNgram.addItem('Unigram + Bigram')
                unibi_ig = pd.DataFrame(unibi_ig)
                unibi_ig.columns = ['IG', 'Words']
                unibi_ig['IG'] = unibi_ig['IG'].astype(float)
                unibi_words_ig = sns.barplot(x='IG', y='Words', data=unibi_ig, color=(0.2, 0.4, 0.6, 0.6))
                unibi_words_ig = unibi_words_ig.get_figure()
                unibi_words_ig.savefig("unibi_words_ig.png", bbox_inches="tight")
            if self.checkBoxUniTri.isChecked():
                self.comboBoxNgram.addItem('Unigram + Trigram')
                unitri_ig = pd.DataFrame(unitri_ig)
                unitri_ig.columns = ['IG', 'Words']
                unitri_ig['IG'] = unitri_ig['IG'].astype(float)
                unitri_words_ig = sns.barplot(x='IG', y='Words', data=unitri_ig, color=(0.2, 0.4, 0.6, 0.6))
                unitri_words_ig = unitri_words_ig.get_figure()
                unitri_words_ig.savefig("unitri_words_ig.png", bbox_inches="tight")
            if self.checkBoxBiTri.isChecked():
                self.comboBoxNgram.addItem('Bigram + Trigram')
                bitri_ig = pd.DataFrame(bitri_ig)
                bitri_ig.columns = ['IG', 'Words']
                bitri_ig['IG'] = bitri_ig['IG'].astype(float)
                bitri_words_ig = sns.barplot(x='IG', y='Words', data=bitri_ig, color=(0.2, 0.4, 0.6, 0.6))
                bitri_words_ig = bitri_words_ig.get_figure()
                bitri_words_ig.savefig("bitri_words_ig.png", bbox_inches="tight")
            if self.checkBoxUniBiTri.isChecked():
                self.comboBoxNgram.addItem('Unigram + Bigram + Trigram')
                unibitri_ig = pd.DataFrame(unibitri_ig)
                unibitri_ig.columns = ['IG', 'Words']
                unibitri_ig['IG'] = unibitri_ig['IG'].astype(float)
                unibitri_words_ig = sns.barplot(x='IG', y='Words', data=unibitri_ig, color=(0.2, 0.4, 0.6, 0.6))
                unibitri_words_ig = unibitri_words_ig.get_figure()
                unibitri_words_ig.savefig("unibitri_words_ig.png", bbox_inches="tight")

        if self.checkBoxDF.isChecked():
            if self.checkBoxUni.isChecked():
                if not self.checkBoxIG.isChecked():
                    self.comboBoxNgram.addItem('Unigram')
                uni_df = pd.DataFrame(uni_df)
                uni_df.columns = ['DF', 'Words']
                uni_df['DF'] = uni_df['DF'].astype(float)
                uni_words_df = sns.barplot(x='DF', y='Words', data=uni_df, color=(0.2, 0.4, 0.6, 0.6))
                uni_words_df = uni_words_df.get_figure()
                uni_words_df.savefig("uni_words_df.png", bbox_inches="tight")
            if self.checkBoxBi.isChecked():
                if not self.checkBoxIG.isChecked():
                    self.comboBoxNgram.addItem('Bigram')
                bi_df = pd.DataFrame(bi_df)
                bi_df.columns = ['DF', 'Words']
                bi_df['DF'] = bi_df['DF'].astype(float)
                bi_words_df = sns.barplot(x='DF', y='Words', data=bi_df, color=(0.2, 0.4, 0.6, 0.6))
                bi_words_df = bi_words_df.get_figure()
                bi_words_df.savefig("bi_words_df.png", bbox_inches="tight")
            if self.checkBoxTri.isChecked():
                if not self.checkBoxIG.isChecked():
                    self.comboBoxNgram.addItem('Trigram')
                tri_df = pd.DataFrame(tri_df)
                tri_df.columns = ['DF', 'Words']
                tri_df['DF'] = tri_df['DF'].astype(float)
                tri_words_df = sns.barplot(x='DF', y='Words', data=tri_df, color=(0.2, 0.4, 0.6, 0.6))
                tri_words_df = tri_words_df.get_figure()
                tri_words_df.savefig("tri_words_df.png", bbox_inches="tight")
            if self.checkBoxUniBi.isChecked():
                if not self.checkBoxIG.isChecked():
                    self.comboBoxNgram.addItem('Unigram + Bigram')
                unibi_df = pd.DataFrame(unibi_df)
                unibi_df.columns = ['DF', 'Words']
                unibi_df['DF'] = unibi_df['DF'].astype(float)
                unibi_words_df = sns.barplot(x='DF', y='Words', data=unibi_df, color=(0.2, 0.4, 0.6, 0.6))
                unibi_words_df = unibi_words_df.get_figure()
                unibi_words_df.savefig("unibi_words_df.png", bbox_inches="tight")
            if self.checkBoxUniTri.isChecked():
                if not self.checkBoxIG.isChecked():
                    self.comboBoxNgram.addItem('Unigram + Trigram')
                unitri_df = pd.DataFrame(unitri_df)
                unitri_df.columns = ['DF', 'Words']
                unitri_df['DF'] = unitri_df['DF'].astype(float)
                unitri_words_df = sns.barplot(x='DF', y='Words', data=unitri_df, color=(0.2, 0.4, 0.6, 0.6))
                unitri_words_df = unitri_words_df.get_figure()
                unitri_words_df.savefig("unitri_words_df.png", bbox_inches="tight")
            if self.checkBoxBiTri.isChecked():
                if not self.checkBoxIG.isChecked():
                    self.comboBoxNgram.addItem('Bigram + Trigram')
                bitri_df = pd.DataFrame(bitri_df)
                bitri_df.columns = ['DF', 'Words']
                bitri_df['DF'] = bitri_df['DF'].astype(float)
                bitri_words_df = sns.barplot(x='DF', y='Words', data=bitri_df, color=(0.2, 0.4, 0.6, 0.6))
                bitri_words_df = bitri_words_df.get_figure()
                bitri_words_df.savefig("bitri_words_df.png", bbox_inches="tight")
            if self.checkBoxUniBiTri.isChecked():
                if not self.checkBoxIG.isChecked():
                    self.comboBoxNgram.addItem('Unigram + Bigram + Trigram')
                unibitri_df = pd.DataFrame(unibitri_df)
                unibitri_df.columns = ['DF', 'Words']
                unibitri_df['DF'] = unibitri_df['DF'].astype(float)
                unibitri_words_df = sns.barplot(x='DF', y='Words', data=unibitri_df, color=(0.2, 0.4, 0.6, 0.6))
                unibitri_words_df = unibitri_words_df.get_figure()
                unibitri_words_df.savefig("unibitri_words_df.png", bbox_inches="tight")

        self.comboBoxNgram.adjustSize()
        self.comboBoxNgram.setEnabled(True)

        if self.checkBoxIG.isChecked():
            if self.checkBoxUni.isChecked():
                pixmapwords = QPixmap('uni_words_ig.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif self.checkBoxUniBi.isChecked():
                pixmapwords = QPixmap('unibi_words_ig.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif self.checkBoxUniTri.isChecked():
                pixmapwords = QPixmap('unitri_words_ig.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif self.checkBoxBi.isChecked():
                pixmapwords = QPixmap('bi_words_ig.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif self.checkBoxBiTri.isChecked():
                pixmapwords = QPixmap('bitri_words_ig.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif self.checkBoxTri.isChecked():
                pixmapwords = QPixmap('tri_words_ig.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif self.checkBoxUniBiTri.isChecked():
                pixmapwords = QPixmap('unibitri_words_ig.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif self.checkBoxDF.isChecked():
                if self.checkBoxUni.isChecked():
                    pixmapwords = QPixmap('uni_words_df.png')
                    self.labelFS.setPixmap(pixmapwords)
                    self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
                elif self.checkBoxUniBi.isChecked():
                    pixmapwords = QPixmap('unibi_words_df.png')
                    self.labelFS.setPixmap(pixmapwords)
                    self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
                elif self.checkBoxUniTri.isChecked():
                    pixmapwords = QPixmap('unitri_words_df.png')
                    self.labelFS.setPixmap(pixmapwords)
                    self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
                elif self.checkBoxBi.isChecked():
                    pixmapwords = QPixmap('bi_words_df.png')
                    self.labelFS.setPixmap(pixmapwords)
                    self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
                elif self.checkBoxBiTri.isChecked():
                    pixmapwords = QPixmap('bitri_words_df.png')
                    self.labelFS.setPixmap(pixmapwords)
                    self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
                elif self.checkBoxTri.isChecked():
                    pixmapwords = QPixmap('tri_words_df.png')
                    self.labelFS.setPixmap(pixmapwords)
                    self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
                elif self.checkBoxUniBiTri.isChecked():
                    pixmapwords = QPixmap('unibitri_words_df.png')
                    self.labelFS.setPixmap(pixmapwords)
                    self.labelFS.resize(pixmapwords.width(), pixmapwords.height())

        sns.set(style="ticks")
        custom_palette = ["red", "blue", "yellow"]
        sns.set_palette(custom_palette)

        if self.checkBoxPres.isChecked():
            pres_plot = sns.catplot(x='Data Representations', y="Fold Results", hue="Classifier", kind="box",
                                    data=performancePres)
            pres_plot.set(ylabel=y_label, title=y_label + ' For Different Representations')
            pres_plot.set(ylim=(0.45, 1))
            for axes in pres_plot.axes.flat:
                axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
            pres_plot.savefig("MainGraphPres.png", bbox_inches="tight")

        if self.checkBoxCount.isChecked():
            count_plot = sns.catplot(x='Data Representations', y="Fold Results", hue="Classifier", kind="box",
                                     data=performanceCount)
            count_plot.set(ylabel=y_label, title=y_label + ' For Different Representations')
            count_plot.set(ylim=(0.45, 1))
            for axes in count_plot.axes.flat:
                axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
            count_plot.savefig("MainGraphCount.png", bbox_inches="tight")

        if self.checkBoxTDIDF.isChecked():
            TDIDF_plot = sns.catplot(x='Data Representations', y="Fold Results", hue="Classifier", kind="box",
                                     data=performanceTDIDF)
            TDIDF_plot.set(ylabel=y_label, title=y_label + ' For Different Representations')
            TDIDF_plot.set(ylim=(0.45, 1))
            for axes in TDIDF_plot.axes.flat:
                axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
            TDIDF_plot.savefig("MainGraphTDIDF.png", bbox_inches="tight")

        self.labelGraph.resize(500, 1000)
        if self.checkBoxPres.isChecked():
            pixmappres = QPixmap('MainGraphPres.png')
            self.labelGraph.setPixmap(pixmappres)
            self.labelGraph.setScaledContents(True)
        elif self.checkBoxCount.isChecked():
            pixmapcount = QPixmap('MainGraphCount.png')
            self.labelGraph.setPixmap(pixmapcount)
            self.labelGraph.setScaledContents(True)
        elif self.checkBoxTDIDF.isChecked():
            pixmapTDIDF = QPixmap('MainGraphTDIDF.png')
            self.labelGraph.setPixmap(pixmapTDIDF)
            self.labelGraph.setScaledContents(True)

        self.comboBoxTab.setCurrentIndex(1)
        self.comboBoxFS.setCurrentIndex(1)
        self.comboBoxNgram.setCurrentIndex(1)

    ##############################################################
    # Functions for sequential flow
    ##############################################################

    def togglePicture(self):
        if self.comboBoxTab.currentIndex() == 1:
            if self.checkBoxPres.isChecked():
                pixmappres = QPixmap('MainGraphPres.png')
                self.labelGraph.setPixmap(pixmappres)
                self.labelGraph.resize(pixmappres.width(), pixmappres.height())
            elif self.checkBoxCount.isChecked():
                pixmapcount = QPixmap('MainGraphCount.png')
                self.labelGraph.setPixmap(pixmapcount)
                self.labelGraph.resize(pixmapcount.width(), pixmapcount.height())
            elif self.checkBoxTDIDF.isChecked():
                pixmapTDIDF = QPixmap('MainGraphTDIDF.png')
                self.labelGraph.setPixmap(pixmapTDIDF)
                self.labelGraph.resize(pixmapTDIDF.width(), pixmapTDIDF.height())

        if self.comboBoxTab.currentIndex() == 2:
            if (self.checkBoxPres.isChecked() == True and self.checkBoxCount.isChecked() == True):
                pixmapcount = QPixmap('MainGraphCount.png')
                self.labelGraph.setPixmap(pixmapcount)
                self.labelGraph.resize(pixmapcount.width(), pixmapcount.height())
            elif (self.checkBoxPres.isChecked() == True and self.checkBoxTDIDF.isChecked() == True):
                pixmapcount = QPixmap('MainGraphTDIDF.png')
                self.labelGraph.setPixmap(pixmapcount)
                self.labelGraph.resize(pixmapcount.width(), pixmapcount.height())
            elif (self.checkBoxCount.isChecked() == True and self.checkBoxTDIDF.isChecked() == True):
                pixmapcount = QPixmap('MainGraphTDIDF.png')
                self.labelGraph.setPixmap(pixmapcount)
                self.labelGraph.resize(pixmapcount.width(), pixmapcount.height())

        if self.comboBoxTab.currentIndex() == 3:
            pixmapcount = QPixmap('MainGraphTDIDF.png')
            self.labelGraph.setPixmap(pixmapcount)
            self.labelGraph.resize(pixmapcount.width(), pixmapcount.height())

    def toggleFSList(self):
        if str(self.comboBoxFS.currentText()) == 'Information Gain':
            if str(self.comboBoxNgram.currentText()) == 'Unigram':
                pixmapwords = QPixmap('uni_words_ig.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif str(self.comboBoxNgram.currentText()) == 'Bigram':
                pixmapwords = QPixmap('bi_words_ig.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif str(self.comboBoxNgram.currentText()) == 'Trigram':
                pixmapwords = QPixmap('tri_words_ig.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif str(self.comboBoxNgram.currentText()) == 'Unigram + Bigram':
                pixmapwords = QPixmap('unibi_words_ig.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif str(self.comboBoxNgram.currentText()) == 'Unigram + Trigram':
                pixmapwords = QPixmap('unitri_words_ig.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif str(self.comboBoxNgram.currentText()) == 'Bigram + Trigram':
                pixmapwords = QPixmap('bitri_words_ig.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif str(self.comboBoxNgram.currentText()) == 'Unigram + Bigram + Trigram':
                pixmapwords = QPixmap('unibitri_words_ig.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())

        if str(self.comboBoxFS.currentText()) == 'Document Frequency':
            if str(self.comboBoxNgram.currentText()) == 'Unigram':
                pixmapwords = QPixmap('uni_words_df.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif str(self.comboBoxNgram.currentText()) == 'Bigram':
                pixmapwords = QPixmap('bi_words_df.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif str(self.comboBoxNgram.currentText()) == 'Trigram':
                pixmapwords = QPixmap('tri_words_df.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif str(self.comboBoxNgram.currentText()) == 'Unigram + Bigram':
                pixmapwords = QPixmap('unibi_words_df.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif str(self.comboBoxNgram.currentText()) == 'Unigram + Trigram':
                pixmapwords = QPixmap('unitri_words_df.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif str(self.comboBoxNgram.currentText()) == 'Bigram + Trigram':
                pixmapwords = QPixmap('bitri_words_df.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())
            elif str(self.comboBoxNgram.currentText()) == 'Unigram + Bigram + Trigram':
                pixmapwords = QPixmap('unibitri_words_df.png')
                self.labelFS.setPixmap(pixmapwords)
                self.labelFS.resize(pixmapwords.width(), pixmapwords.height())

    def toggleNormalisation(self, b):
        if b.isChecked() == True:
            self.labelStemLem.setEnabled(True)
            self.radioStemming.setEnabled(True)
            self.radioLemmatisation.setEnabled(True)
            self.radioNonen.setEnabled(True)

    def toggleML(self, b):
        if b.isChecked() == True:
            self.labelML.setEnabled(True)
            self.checkBoxSVM.setEnabled(True)
            self.checkBoxNB.setEnabled(True)
            self.checkBoxME.setEnabled(True)

    def toggleNgram(self):
        if self.checkBoxSVM.isChecked() == True or self.checkBoxNB.isChecked() == True or self.checkBoxME.isChecked() == True:
            self.labelNgram.setEnabled(True)
            self.checkBoxUni.setEnabled(True)
            self.checkBoxUniBi.setEnabled(True)
            self.checkBoxUniTri.setEnabled(True)
            self.checkBoxBi.setEnabled(True)
            self.checkBoxBiTri.setEnabled(True)
            self.checkBoxTri.setEnabled(True)
            self.checkBoxUniBiTri.setEnabled(True)

    def toggleFC(self):
        if (
                self.checkBoxUni.isChecked() == True or self.checkBoxUniBi.isChecked() == True or self.checkBoxUniTri.isChecked() == True
                or self.checkBoxBi.isChecked() == True or self.checkBoxBiTri.isChecked() == True or self.checkBoxTri.isChecked() == True or self.checkBoxUniBiTri.isChecked() == True):
            self.labelConstruction.setEnabled(True)
            self.checkBoxPres.setEnabled(True)
            self.checkBoxCount.setEnabled(True)
            self.checkBoxTDIDF.setEnabled(True)

    def toggleFS(self):
        if self.checkBoxPres.isChecked() is True or self.checkBoxCount.isChecked() is True or self.checkBoxTDIDF.isChecked() is True:
            self.labelSelection.setEnabled(True)
            self.checkBoxIG.setEnabled(True)
            self.checkBoxDF.setEnabled(True)
            self.checkBoxFSNone.setEnabled(True)

    def toggleFeatureNumber(self):
        if self.checkBoxIG.isChecked() is True or self.checkBoxDF.isChecked() is True:
            self.feature_number.setEnabled(True)
            self.labelFeatures.setEnabled(True)
            self.labelFeaturesDocument.setEnabled(True)

    def toggleTune(self):
        if len(self.feature_number.text()) > 0 or self.checkBoxFSNone.isChecked() is True:
            self.labelTuningandCV.setEnabled(True)
            self.comboBoxTune.setEnabled(True)
            self.labelTune.setEnabled(True)
            self.labelMetric.setEnabled(True)

    def toggleEvaluation(self):
        self.comboBoxEvaluation.setEnabled(True)
        self.labelPerformance.setEnabled(True)

    def toggleTuneFolds(self):
        self.comboBoxTuneFold.setEnabled(True)
        self.labelFolds.setEnabled(True)

    def toggleFolds(self):
        self.comboBoxFolds.setEnabled(True)

    def toggleButton(self):
        self.btnExecute.setEnabled(True)

    def toggleTab(self):
        self.tabWidget.setCurrentIndex(1)
        self.results_tab.setEnabled(True)
        self.results_frame.setEnabled(True)

    ##############################################################
    # Functions for feature selection
    ##############################################################

    def performIG(self, the_vector, the_vectorizer, second_vectorizer=None):
        global number_per_class
        global number_of_docs
        global number_of_classes
        global count_of_that_class
        global probability_of_classess
        global doc_clss_index

        number_of_features = int(self.feature_number.text())

        if second_vectorizer is not None:
            vocab1 = the_vectorizer.get_feature_names()
            vocab2 = second_vectorizer.get_feature_names()
            vocab = vocab1 + vocab2
        if second_vectorizer is None:
            vocab = the_vectorizer.get_feature_names()

        vocab_size = len(vocab)
        word_occurance_frequency = np.zeros(vocab_size, dtype=int)
        word_occurance_frequency_vs_class = np.zeros((vocab_size, number_of_classes), dtype=int)

        vocab_list = np.asarray(vocab)

        for j in range(0, vocab_size):
            word_occurance_frequency[j] = the_vector[:, j].sum()
        word_occurance_frequency[word_occurance_frequency == 0] = 1

        for j in range(0, (vocab_size)):
            for i in range(0, number_of_docs):
                if the_vector[i, j] != 0:
                    word_occurance_frequency_vs_class[j][doc_clss_index[i]] += the_vector[i, j]

                    # probabilities
        p_w = word_occurance_frequency / number_of_docs
        p_w_not = 1 - p_w
        p_c = probability_of_classess

        p_class_condition_on_w = np.zeros((number_of_classes, vocab_size), dtype=float)
        tmp = word_occurance_frequency_vs_class.T
        for i in range(0, number_of_classes):
            p_class_condition_on_w[i] = tmp[i] / word_occurance_frequency

        p_class_condition_on_not_w = np.zeros((number_of_classes, vocab_size), dtype=float)
        for i in range(0, number_of_classes):
            p_class_condition_on_not_w[i] = (count_of_that_class[i] - tmp[i]) / (
                    number_of_docs - word_occurance_frequency)

        # compute information gain

        word_ig_information = []

        number_per_class = raw_data.sentiment.value_counts()
        Corpus_entropy = entropy(number_per_class, qk=None, base=2)
        # Corpus_entropy

        for j in range(0, vocab_size):
            e_1 = 0.0
            for classno in range(0, number_of_classes):
                tmp1 = p_class_condition_on_w[classno][j]
                # if tmp1 !=0:
                if tmp1 > 0:
                    e_1 += p_w[j] * tmp1 * np.log2(tmp1)
                tmp2 = p_class_condition_on_not_w[classno][j]
                # if tmp2 !=0:
                if tmp2 > 0:
                    e_1 += (1 - p_w[j]) * (tmp2 * np.log2(tmp2))
            e_1 = -e_1

            information_gain = Corpus_entropy - e_1

            word_ig_information.append([information_gain, vocab_list[j]])

        word_ig_information = np.array(sorted(word_ig_information, key=lambda x: x[0], reverse=True))
        reduced_features_ig_vocab = word_ig_information[0:number_of_features, :]
        return reduced_features_ig_vocab

    def performDF(self, the_vector, the_vectorizer, second_vectorizer=None):
        global number_per_class
        global number_of_docs
        global number_of_classes
        global count_of_that_class
        global probability_of_classess
        global doc_clss_index

        number_of_features = int(self.feature_number.text())

        if second_vectorizer is not None:
            vocab1 = the_vectorizer.get_feature_names()
            vocab2 = second_vectorizer.get_feature_names()
            vocab = vocab1 + vocab2
        if second_vectorizer is None:
            vocab = the_vectorizer.get_feature_names()

        vocab_size = len(vocab)
        word_occurance_frequency = np.zeros(vocab_size, dtype=int)
        word_occurance_frequency_vs_class = np.zeros((vocab_size, number_of_classes), dtype=int)

        vocab_list = np.asarray(vocab)

        for j in range(0, vocab_size):
            word_occurance_frequency[j] = the_vector[:, j].sum()
        word_occurance_frequency[word_occurance_frequency == 0] = 1

        for j in range(0, (vocab_size)):
            for i in range(0, number_of_docs):
                if the_vector[i, j] != 0:
                    word_occurance_frequency_vs_class[j][doc_clss_index[i]] += the_vector[i, j]

                    # probabilities
        p_w = word_occurance_frequency / number_of_docs
        p_w_not = 1 - p_w
        p_c = probability_of_classess

        p_class_condition_on_w = np.zeros((number_of_classes, vocab_size), dtype=float)
        tmp = word_occurance_frequency_vs_class.T
        for i in range(0, number_of_classes):
            p_class_condition_on_w[i] = tmp[i] / word_occurance_frequency

        p_class_condition_on_not_w = np.zeros((number_of_classes, vocab_size), dtype=float)
        for i in range(0, number_of_classes):
            p_class_condition_on_not_w[i] = (count_of_that_class[i] - tmp[i]) / (
                    number_of_docs - word_occurance_frequency)

        # compute document frequency

        word_df_information = []

        for j in range(0, vocab_size):
            document_frequency = word_occurance_frequency[j] / number_of_docs
            word_df_information.append([document_frequency, vocab_list[j]])

        word_df_information = np.array(sorted(word_df_information, key=lambda x: x[0], reverse=True))
        reduced_features_df_vocab = word_df_information[0:number_of_features, :]
        return reduced_features_df_vocab

    ##############################################################
    # Functions for performance evaluation
    ##############################################################

    def CrossValidationNormal(self, data_representation, classifier_name, classifier, parameters, features, labels,
                              folds, tune_metric):
        """
        Performs cross validation for some specified classifier using AUC as performance metric
        :param data_representation:
        :param classifier_name:
        :param classifier:
        :param parameters:
        :param features:
        :param labels:
        :param folds:
        :param tune_metric:
        :return:
        """
        if self.comboBoxTuneFold.currentIndex() == 1:
            tune_fold = 3
        if self.comboBoxTuneFold.currentIndex() == 2:
            tune_fold = 5
        if self.comboBoxTuneFold.currentIndex() == 3:
            tune_fold = 10

        col_names = ['Data Representations', 'Fold Results', 'Classifier']
        resultsDF = pd.DataFrame(columns=col_names)

        fold_scores = []

        skf = StratifiedKFold(n_splits=folds, random_state=9)

        for train_index, test_index in skf.split(features, labels):
            x_train, x_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            best_classifier_gscv = GridSearchCV(classifier, parameters, scoring=tune_metric, cv=tune_fold, n_jobs=-1,
                                                refit=True)
            best_classifier_gscv.fit(x_train, y_train)
            best_classifier = best_classifier_gscv.best_estimator_

            predicted_labels = best_classifier.predict(x_test)

            # Chosen metric
            if self.comboBoxEvaluation.currentIndex() == 1:
                fold_scores.append(accuracy_score(y_true=y_test, y_pred=predicted_labels))
            if self.comboBoxEvaluation.currentIndex() == 2:
                fold_scores.append(precision_score(y_true=y_test, y_pred=predicted_labels, average='micro'))
            if self.comboBoxEvaluation.currentIndex() == 3:
                fold_scores.append(recall_score(y_true=y_test, y_pred=predicted_labels, average='micro'))
            if self.comboBoxEvaluation.currentIndex() == 4:
                fold_scores.append(f1_score(y_true=y_test, y_pred=predicted_labels, average='micro'))

        fold_scores = np.asarray(fold_scores)
        resultsDF['Fold Results'] = fold_scores
        resultsDF['Data Representations'] = data_representation
        resultsDF['Classifier'] = classifier_name

        return resultsDF

    def CrossValidationAUC(self, data_representation, classifier_name, classifier, parameters, features, labels, folds,
                           tune_metric):
        """
        Performs cross validation for some specified classifier using AUC as performance metric
        :param data_representation:
        :param classifier_name:
        :param classifier:
        :param parameters:
        :param features:
        :param labels:
        :param folds:
        :param tune_metric:
        :return:
        """
        if self.comboBoxTuneFold.currentIndex() == 1:
            tune_fold = 3
        if self.comboBoxTuneFold.currentIndex() == 2:
            tune_fold = 5
        if self.comboBoxTuneFold.currentIndex() == 3:
            tune_fold = 10

        col_names = ['Data Representations', 'Fold Results', 'Classifier']
        resultsDF = pd.DataFrame(columns=col_names)

        fold_scores = []

        skf = StratifiedKFold(n_splits=folds, random_state=9)
        for train_index, test_index in skf.split(features, labels):
            x_train, x_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            best_classifier_gscv = GridSearchCV(classifier, parameters, scoring=tune_metric, cv=tune_fold, n_jobs=-1,
                                                refit=True)
            best_classifier_gscv.fit(x_train, y_train)
            best_classifier = best_classifier_gscv.best_estimator_

            predicted_labels = best_classifier.predict_proba(x_test)
            np.nan_to_num(predicted_labels, False)
            fold_scores.append(self.multi_roc(y_test, predicted_labels))

        fold_scores = np.asarray(fold_scores)
        resultsDF['Fold Results'] = fold_scores
        resultsDF['Data Representations'] = data_representation
        resultsDF['Classifier'] = classifier_name

        return resultsDF

    def multi_roc(self, true, pred):
        ''' Multiclass ROC value
        :param true: vector of true labels of shape [n_samples]
        :param pred: matrix of predicted scores/probabilities for each label of shape [n_samples],[n_samples, n_classes]
               (output of predict_proba)
        :return: macro and micro-averaged AUC scores for a multi class problem
        '''
        classes = true.unique()
        classes.sort()
        num_classes = len(classes)

        i = 0
        auc = {}
        macro_auc = 0
        micro_auc = 0

        for class_i in classes:
            y_i = true == class_i  # convert to just one class
            p_i = []
            for j in range(len(pred)):
                # extract numerical predictions for class i (necessary due to them not being in separate columns)
                p_i.append(pred[j][i])
            i += 1

            auc_i = roc_auc_score(y_i, p_i)
            auc[class_i] = auc_i
            macro_auc += auc_i
            micro_auc += auc_i * sum(y_i)

        macro_auc /= num_classes
        micro_auc /= len(true)  # num_samples

        return micro_auc


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app_window = window()
    app.setStyleSheet(qdarkgraystyle.load_stylesheet())
    app_window.show()
    sys.exit(app.exec_())
