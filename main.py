import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = load_iris()
X,y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200, random_state=42)

model.fit(X_train,y_train)

score = model.score(X_test,y_test)
print('acuracia no conjunto de teste:', score)

joblib.dump(model,'model_iris.pkl')
print('salvo em model_iris.pkl')