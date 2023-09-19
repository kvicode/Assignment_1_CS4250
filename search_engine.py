#importing some Python libraries
import csv
import math

documents = []
labels = []

#reading the data in a csv file
with open('C:/Users/bobb1/OneDrive/Desktop/CPP Stuff/CS4250\collection.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            documents.append(row[0])
            labels.append(row[1])

# Conduct stopword removal.
stopWords = {'I', 'and', 'She', 'They', 'her', 'their'}
documents = [' '.join([word for word in doc.split() if word.lower() not in stopWords]) for doc in documents]

# Conduct stemming.
stemming = {
    "cats": "cat",
    "dogs": "dog",
    "loves": "love",
}
documents = [' '.join([stemming.get(word, word) for word in doc.split()]) for doc in documents]

# Tokenize and build the list of index terms.
terms = []
for doc in documents:
    for word in doc.split():
        if word not in terms:
            terms.append(word)

# Build the tf-idf term weights matrix.
docMatrix = []
for doc in documents:
    tfidf_weights = []
    for term in terms:
        tf = doc.split().count(term)
        idf = math.log(len(documents) / (sum([1 for d in documents if term in d]) + 1))
        tfidf = tf * idf
        tfidf_weights.append(tfidf)
    docMatrix.append(tfidf_weights)

# Calculate the document scores (ranking) using document weights (tf-idf) and query weights (binary - have or not the term).
query = 'cat and dog'
query_terms = query.split()

docScores = []
for doc_weights in docMatrix:
    score = sum([doc_weights[terms.index(term)] if term in doc.split() else 0 for term in query_terms])
    docScores.append(score)

# Calculate and print the precision and recall of the model by considering that the search engine will return all documents with scores >= 0.1.
threshold = 0.1
#relevant_docs = [i for i, score in enumerate(docScores) if score >= threshold]
#relevant_labels = [labels[i] for i in relevant_docs]
total_relevant = labels.count('R')

true_positives = sum([1 for i, score in enumerate(docScores) if score >= threshold and labels[i] == 'R'])
false_positives = sum([1 for i, score in enumerate(docScores) if score >= threshold and labels[i] == 'I'])
false_negatives = total_relevant - true_positives

if true_positives + false_positives > 0:
    precision = true_positives / (true_positives + false_positives)
else:
    precision = 0.0

if true_positives + false_negatives > 0:
    recall = true_positives / (true_positives + false_negatives)
else:
    recall = 0.0

print("Precision:", precision)
print("Recall:", recall)
