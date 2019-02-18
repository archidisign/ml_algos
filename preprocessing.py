
# Read csv file
with open(data_path, 'r', encoding="utf-8") as f:
    data = f.readlines()
data = [x.strip().split("\t") for x in data]


n = len(data)
X = np.zeros((n), dtype = object)
y = np.zeros((n))
for i in range(n):
    X[i] = data[i][0].replace("<br />", " ").replace("/", " ").lower().translate(table).translate(dig_table)
    y[i] = int(data[i][1])


# --------------------------------------------
# Natural Language Processing
from sklearn.feature_extraction.text import CountVectorizer

vectorizer_binary = CountVectorizer(max_features = 10000, binary=True)
vectorizer_frequency = CountVectorizer(max_features = 10000, binary=False)
yelp_bbow_train = vectorizer_binary.fit_transform(X)

temp = vectorizer_frequency.fit_transform(X).toarray()
yelp_fbow_train = [temp[i]/sum(temp[i]) if sum(temp[i])>0 else temp[i] for i in range(len(temp))]


def bag(s):
    arr = s.split()
    new_str = ""
    for i in range(len(arr)):
        try:
            j = words.index(arr[i])
            new_str += str(j+1) + " "
        except:
            continue
    return new_str


def generate_DB(db0,db1):
    n = len(db0)
    strings = ['']*n
    for i in range(n):
        strings[i] = bag(db0[i])[:-1] + '\t' + str(db1[i])
    return strings


yelp_dataset_train = generate_DB(X, y)