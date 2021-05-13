# Implement your own version of the Naive Bayes Classifier

# open the data files from github
neg_reviews = list(open('cnn-text-classification-tf/data/rt-polaritydata/rt-polarity.neg', 'r'))
pos_reviews = list(open('cnn-text-classification-tf/data/rt-polaritydata/rt-polarity.pos', 'r'))

# clean the text by removing newline characters and punctuation
def clean_text(old_list):

    new_list = []
    punctuation = ['[', ']', '"', ',', '.', '(', ')', '?', '!', '&', '--']

    for review in old_list:
        review = review.rstrip('\n')
        review = review.rstrip(' .')
        review = review.rstrip(' ')
        words = review.split(' ')
        for i in range(0, len(words)):
            words[i] = words[i].lstrip("'")
            words[i] = words[i].rstrip("'")
            words[i] = words[i].lstrip('/')
            words[i] = words[i].rstrip('/')
            for item in punctuation:
                words[i] = words[i].replace(item, '')
        review = ' '.join(words)
        review = review.replace('  ', ' ')
        new_list.append(review)

    return new_list

neg_reviews = clean_text(neg_reviews)
pos_reviews = clean_text(pos_reviews)

# split negative and positive reviews into training, development, and test sets
num_neg = len(neg_reviews)
num_pos = len(pos_reviews)
num_negtrain = int(0.7*num_neg)
num_postrain = int(0.7*num_pos)
negdev_ind = num_negtrain + int(0.15*num_neg)
posdev_ind = num_postrain + int(0.15*num_pos)
neg_train = neg_reviews[0:num_negtrain]
pos_train = pos_reviews[0:num_postrain]
neg_dev = neg_reviews[num_negtrain:negdev_ind]
pos_dev = pos_reviews[num_postrain:posdev_ind]
neg_test = neg_reviews[negdev_ind:]
pos_test = pos_reviews[posdev_ind:]

def train_classifier(neg_train, pos_train): 
    # set probability of a document falling into the negative and positive classes
    neg_prior = len(neg_train)/(len(neg_train) + len(pos_train))
    pos_prior = len(pos_train)/(len(neg_train) + len(pos_train))

    # negative vocabulary
    neg_vocab = []
    for review in neg_train:
        for word in review.split(' '):
            if word not in neg_vocab:
                neg_vocab.append(word)

    # positive vocabulary
    pos_vocab = []
    for review in pos_train:
        for word in review.split(' '):
            if word not in pos_vocab:
                pos_vocab.append(word)

    # negative word counts
    neg_counts = {}
    for word in neg_vocab:
        neg_counts[word] = str(neg_train).count(word)

    # positive word counts
    pos_counts = {}
    for word in pos_vocab:
        pos_counts[word] = str(pos_train).count(word)

    return neg_prior, pos_prior, neg_vocab, pos_vocab, neg_counts, pos_counts


# classify each review as positive or negative by calculating the probability that it falls into 
# each class based on the words contained in the review
def classify(document):
    # assign probabilities
    global pos_prior
    global neg_prior

    pos_prob = pos_prior
    neg_prob = neg_prior

    negWordProbs = {}
    numNegWords = sum(neg_counts.values())
    for word in neg_counts:
        negWordProbs[word] = (neg_counts[word] + 1)/(numNegWords + len(neg_vocab))

    posWordProbs = {}
    numPosWords = sum(pos_counts.values())
    for word in pos_counts:
        posWordProbs[word] = (pos_counts[word] + 1)/(numPosWords + len(pos_vocab))

    # split the document and calculate the probability that it is negative or positive based on each
    # of its words
    document = document.split(' ')

    for word in document:
        if word in negWordProbs.keys():
            neg_prob *= negWordProbs[word]
        else:
            neg_prob *= (1/(numNegWords + len(neg_vocab)))
        if word in posWordProbs.keys():
            pos_prob *= posWordProbs[word]
        else:
            pos_prob *= (1/(numPosWords + len(pos_vocab)))
    
    # determine if the negative or positive probability is greater and classify the document as follows
    if neg_prob > pos_prob:
        doc_class = "negative"
    elif pos_prob > neg_prob:
        doc_class = "positive"
    else: 
        doc_class = "unknown" 
    
    certainty = pos_prob - neg_prob
    
    return doc_class, certainty


# check the accuracy of the classifier on the data
def check_accuracy(neg_data, pos_data, certainty = False):
    accuracy = 0
    neg_certainty = {}
    pos_certainty = {}

    for review in neg_data:
        doc_class, neg_certainty[review] = classify(review)
        if doc_class == "negative":
            accuracy += 1
    
    for review in pos_data:
        doc_class, pos_certainty[review] = classify(review)
        if doc_class == "positive":
            accuracy += 1
    
    if certainty == True:
        return accuracy/(len(neg_data) + len(pos_data)), neg_certainty, pos_certainty


    return accuracy/(len(neg_data) + len(pos_data))

# train and check accuracy
# neg_prior, pos_prior, neg_vocab, pos_vocab, neg_counts, pos_counts = train_classifier(neg_train, pos_train)
# print(check_accuracy(neg_train, pos_train))
# 0.9313856874832485

# check initial development set accuracy
# print(check_accuracy(neg_dev, pos_dev))
# 0.7478097622027534

# tune the vocabulary by adding and removing words
def tune_vocab():

    global neg_vocab
    global pos_vocab
    global neg_counts
    global pos_counts

    # add words from negative and positive lists to vocabulary and upweight them

    # cloned from the following link: https://gist.github.com/4289441.git
    neg_words = list(open('4289441/negative-words.txt', 'r'))
    del neg_words[0:35]

    # cloned from the following link: https://gist.github.com/4289437.git 
    pos_words = list(open('4289437/positive-words.txt', 'r'))
    del pos_words[0:35]

    neg_words = clean_text(neg_words)
    pos_words = clean_text(pos_words)

    for word in neg_words:
        if word in neg_counts.keys():
            neg_counts[word] += 1
        else: 
            neg_counts[word] = 1
            neg_vocab.append(word)

    for word in pos_words:
        if word in pos_counts.keys():
            pos_counts[word] += 1
        else:
            pos_counts[word] = 1
            pos_vocab.append(word)


    # remove stop words
    import spacy
    sp = spacy.load('en_core_web_sm')

    stop_words = sp.Defaults.stop_words

    # create new vocabularies without stop words
    neg_vocab = [word for word in neg_vocab if not word in stop_words]
    pos_vocab = [word for word in pos_vocab if not word in stop_words]

    # remove words that are only one letter
    for word in neg_vocab:
        if len(list(word)) <= 1:
            neg_vocab.remove(word)

    for word in pos_vocab:
        if len(list(word)) <= 1:
            pos_vocab.remove(word)

    # remove words that are no longer in the vocabulary from the word counts
    rm_keys = [word for word in neg_counts.keys() if not word in neg_vocab]

    for key in rm_keys:
        del(neg_counts[key])

    rm_keys = [word for word in pos_counts.keys() if not word in pos_vocab]

    for key in rm_keys:
        del(pos_counts[key])

    return neg_vocab, pos_vocab, neg_counts, pos_counts

# check accuracy after tuning
# neg_vocab, pos_vocab, neg_counts, pos_counts = tune_vocab()
# print(check_accuracy(neg_train, pos_train))
# 0.9394264272313053
# print(check_accuracy(neg_dev, pos_dev))
# 0.7515644555694618

# train again on concatenation of training and development sets
neg_train2 = neg_train + neg_dev
pos_train2 = pos_train + pos_dev
neg_prior, pos_prior, neg_vocab, pos_vocab, neg_counts, pos_counts = train_classifier(neg_train2, pos_train2)
neg_vocab, pos_vocab, neg_counts, pos_counts = tune_vocab()

# check accuracy after training the second time
# print(check_accuracy(neg_train2, pos_train2))
# 0.9324503311258279

# check and report performance on the test set
performance, neg_certainty, pos_certainty = check_accuracy(neg_test, pos_test, certainty = True)
print(f'Final CLassifier Performance: {round(performance, 3)}')
# 0.7659176029962547

# write the classifier's certainty for each negative and positive review to a file
# certainty = pos_prob - neg_prob so + means it was classified as positive and - means it was classified as negative
# smaller magnitude = less certainty
f = open('neg_certainty.txt', 'w')
f.write(str(neg_certainty))
f.close()

g = open('pos_certainty.txt', 'w')
g.write(str(pos_certainty))
g.close()


'''
# Additional Attempted Tuning

# only keep 1000 most frequent negative and positive words in vocab
from operator import itemgetter

del(neg_counts[''])
freqNegWords = dict(sorted(neg_counts.items(), key = itemgetter(1), reverse = True)[:1000])
neg_vocab = list(freqNegWords.keys())

del(pos_counts[''])
freqPosWords = dict(sorted(pos_counts.items(), key = itemgetter(1), reverse = True)[:1000])
pos_vocab = list(freqPosWords.keys())

# adjust counts to match vocabulary
neg_counts, pos_counts = counts()
# calculate the probability of each word in the vocabulary
negWordProbs, posWordProbs, numNegWords, numPosWords = probs()

# check accuracy after removing infrequent words
print(check_accuracy(neg_train, pos_train))
# 0.7042347896006432
print(check_accuracy(neg_dev, pos_dev))
# 0.6589486858573217
# decreased accuracy significantly -- not choosing this option


# remove words that only occur once from vocabulary
for word in neg_counts.keys():
    if neg_counts[word] == 1:
        neg_vocab.remove(word)

for word in pos_counts.keys():
    if pos_counts[word] == 1:
        pos_vocab.remove(word)

# adjust counts to match vocabulary 
# neg_counts, pos_counts = counts()
# calculate the probability of each word in the vocabulary
# negWordProbs, posWordProbs, numNegWords, numPosWords = probs()

# check accuracy after removing words that only occur once
# print(check_accuracy(neg_train, pos_train))
# 0.8976145805414099
# print(check_accuracy(neg_dev, pos_dev))
# 0.7396745932415519
# training set accuracy decreased but development set accuracy decreased only slightly -- potential 
# option to avoid overfitting?
# after checking the accuracy of the other changes with and without removing these words, it appears 
# best not to remove them
'''