import numpy as np
import pandas as pd
from ampligraph.datasets import load_wn18
from ampligraph.latent_features import ComplEx
from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score

X = load_wn18()

def load_definitions(path):
    import regex as re
    def_line = re.compile(r"""(?P<id>\d+)\s+__(?P<tok>(?:[a-z\d`'\.\-\/]+_)+)([A-Z]+)_+(\d)+\s+([`\w \(\).,;\"']+)""")
    with open(path, 'r+') as f:
        defs = f.readlines();
    """
    for l in defs:
        
        m = def_line.match(l)
        if not m:
            print ('no match for ', l)
            raise ValueError("dsdsds")
        print (l, m)
    """
    defsm = [def_line.match(l).groupdict() for l in defs]
    id2tok = {g['id']:g['tok'].replace("_", " ").strip() for g in defsm}
    tok2id = {g['tok'].replace("_", " ").strip():g['id'] for g in defsm}
    return tok2id, id2tok

tok2id, id2tok = load_definitions('wordnet-mlj12-definitions.txt')



X_train, X_valid = X['train'], X['valid']

print('Train set size: ', X_train.shape)
print('Test set size: ', X_valid.shape)

from ampligraph.latent_features import ComplEx

import os
from ampligraph.utils import save_model, restore_model

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from ampligraph.evaluation import evaluate_performance




ke_model_path = "./models/wikirr_ke.amplimodel"
if not os.path.isfile(ke_model_path):
    model = ComplEx(batches_count=10, seed=0, epochs=20, k=150, eta=10,
                    # Use adam optimizer with learning rate 1e-3
                    optimizer='adam', optimizer_params={'lr': 1e-3},
                    # Use pairwise loss with margin 0.5
                    loss='pairwise', loss_params={'margin': 0.5},
                    # Use L2 regularizer with regularizer weight 1e-5
                    regularizer='LP', regularizer_params={'p': 2, 'lambda': 1e-5},
                    # Enable stdout messages (set to false if you don't want to display)
                    verbose=True)


    print ("Training...")
    model.fit(X_train)
    save_model(model, model_name_path=ke_model_path)

    filter_triples = np.concatenate((X_train, X_valid))

    filter = np.concatenate((X['train'], X['valid'], X['test']))

    ranks = evaluate_performance(X['test'],
                                 model=model,
                                 filter_triples=filter,
                                 use_default_protocol=True,  # corrupt subj and obj separately while evaluating
                                 verbose=True)

    mrr = mrr_score(ranks)
    hits_10 = hits_at_n_score(ranks, n=10)
    print("MRR: %f, Hits@10: %f" % (mrr, hits_10))
    # Output: MRR: 0.886406, Hits@10: 0.935000
else:
    model = restore_model(model_name_path=ke_model_path)

w1 = 'tree'
w2 = 'plant'
pred = model.predict(np.array([[tok2id[w1], '_hyponym', tok2id[w2]],
                               [tok2id[w2], '_hyponym', tok2id[w1]],
                               [tok2id[w1], '_hypernym', tok2id[w2]],
                               [tok2id[w2], '_hypernym', tok2id[w1]]]))

print (w1,w2, pred)

w1 = 'spoon'
w2 = 'kitchen'
pred = model.predict(np.array([[tok2id[w1], '_hyponym', tok2id[w2]],
                               [tok2id[w2], '_hyponym', tok2id[w1]],
                               [tok2id[w1], '_hypernym', tok2id[w2]],
                               [tok2id[w2], '_hypernym', tok2id[w1]]]))

print (w1,w2, pred)

w1 = 'fish'
w2 = 'shark'
pred = model.predict(np.array([[tok2id[w1], '_hyponym', tok2id[w2]],
                               [tok2id[w2], '_hyponym', tok2id[w1]],
                               [tok2id[w1], '_hypernym', tok2id[w2]],
                               [tok2id[w2], '_hypernym', tok2id[w1]]]))

print (w1,w2, pred)

w1 = 'dog'
w2 = 'bark'
pred = model.predict(np.array([[tok2id[w1], '_hyponym', tok2id[w2]],
                               [tok2id[w2], '_hyponym', tok2id[w1]],
                               [tok2id[w1], '_hypernym', tok2id[w2]],
                               [tok2id[w2], '_hypernym', tok2id[w1]]]))

print (w1,w2, pred)

w1 = 'radio'
w2 = 'podcast'
pred = model.predict(np.array([[tok2id[w1], '_hyponym', tok2id[w2]],
                               [tok2id[w2], '_hyponym', tok2id[w1]],
                               [tok2id[w1], '_hypernym', tok2id[w2]],
                               [tok2id[w2], '_hypernym', tok2id[w1]]]))

print (w1,w2, pred)


w1 = 'street'
w2 = 'rue'
pred = model.predict(np.array([[tok2id[w1], '_hyponym', tok2id[w2]],
                               [tok2id[w2], '_hyponym', tok2id[w1]],
                               [tok2id[w1], '_hypernym', tok2id[w2]],
                               [tok2id[w2], '_hypernym', tok2id[w1]]]))

print (w1,w2, pred)


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from incf.countryutils import transformations

print("Extracting Embeddings..")

id_to_name_map = {**dict(zip(df.home_team_id, df.home_team)), **dict(zip(df.away_team_id, df.away_team))}

teams = pd.concat((df.home_team_id[df["train"]], df.away_team_id[df["train"]])).unique()
team_embeddings = dict(zip(teams, model.get_embeddings(teams)))


embeddings_2d = PCA(n_components=2).fit_transform(np.array([i for i in team_embeddings.values()]))

print (embeddings_2d)
first_embeddings = list(team_embeddings.values())[0]
print (first_embeddings)
print (first_embeddings.shape)
print (embeddings_2d.shape)
from ampligraph.discovery import find_clusters
from sklearn.cluster import KMeans

print("Clustering..")

clustering_algorithm = KMeans(n_clusters=6, n_init=50, max_iter=500, random_state=0)
clusters = find_clusters(teams, model, clustering_algorithm, mode='entity')

print (clusters)
print("Visualize..")

def cn_to_ctn(country):
    try:
        return transformations.cn_to_ctn(id_to_name_map[country])
    except KeyError:
        return "unk"

plot_df = pd.DataFrame({"teams": teams,
                        "embedding1": embeddings_2d[:, 0],
                        "embedding2": embeddings_2d[:, 1],
                        "continent": pd.Series(teams).apply(cn_to_ctn),
                        "cluster": "cluster" + pd.Series(clusters).astype(str)})

top20teams = ["TeamBelgium", "TeamFrance", "TeamBrazil", "TeamEngland", "TeamPortugal", "TeamCroatia", "TeamSpain",
              "TeamUruguay", "TeamSwitzerland", "TeamDenmark", "TeamArgentina", "TeamGermany", "TeamColombia",
              "TeamItaly", "TeamNetherlands", "TeamChile", "TeamSweden", "TeamMexico", "TeamPoland", "TeamIran"]

def plot_clusters(hue):
    print(hue)

    np.random.seed(0)
    plt.figure(figsize=(12, 12))
    plt.title("{} embeddings".format(hue).capitalize())
    ax = sns.scatterplot(data=plot_df[plot_df.continent!="unk"], x="embedding1", y="embedding2", hue=hue)
    texts = []
    for i, point in plot_df.iterrows():
        if point["teams"] in top20teams or np.random.random() < 0.1:
            texts.append(plt.text(point['embedding1']+0.02, point['embedding2']+0.01, str(point["teams"])))
    adjust_text(texts)
    plt.show()

plot_clusters("continent")


plot_clusters("cluster")


from sklearn import metrics
metrics.adjusted_rand_score(plot_df.continent, plot_df.cluster)

df["results"] = (df.home_score > df.away_score).astype(int) + \
                (df.home_score == df.away_score).astype(int)*2 + \
                (df.home_score < df.away_score).astype(int)*3 - 1

df.results.value_counts(normalize=True)


def get_features_target(mask):
    def get_embeddings(team):
        return team_embeddings.get(team, np.full(list(team_embeddings.values())[0].shape[0], np.nan))

    X = np.hstack((np.vstack(df[mask].home_team_id.apply(get_embeddings).values),
                   np.vstack(df[mask].away_team_id.apply(get_embeddings).values)))
    y = df[mask].results.values
    return X, y

clf_X_train, y_train = get_features_target((df["train"]))
clf_X_test, y_test = get_features_target((~df["train"]))

clf_X_train.shape, clf_X_test.shape

np.isnan(clf_X_test).sum()/clf_X_test.shape[1]

from xgboost import XGBClassifier

clf_model = XGBClassifier(n_estimators=550, max_depth=5, objective="multi:softmax")

clf_model.fit(clf_X_train, y_train)

print (df[~df["train"]].results.value_counts(normalize=True))

print (metrics.accuracy_score(y_test, clf_model.predict(clf_X_test)))




