from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def test_single_model(data,label):
	# CLASSIFICATIONS 
	scoring = ['accuracy', 'f1_weighted','precision_micro','recall_micro']
	scores = {}

	# LOGISTIC REGRESSION
	engine = LogisticRegression()
	scores['LR'] = cross_validate(engine, data, label, scoring=scoring)

	# NAIVE BAYES
	engine = GaussianNB()
	scores['NB'] = cross_validate(engine, data, label, scoring=scoring)

	# KNN
	engine = KNeighborsClassifier()
	scores['KNN'] = cross_validate(engine, data, label, scoring=scoring)

	# DECISION TREE
	engine = DecisionTreeClassifier()
	scores['DT'] = cross_validate(engine, data, label, scoring=scoring)

	# SVD
	engine = SVC()
	scores['SVD'] = cross_validate(engine, data, label, scoring=scoring)
	return scores