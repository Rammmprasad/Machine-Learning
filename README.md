# Machine-Learning
1.### FIND-S Algorithm Implementation

Introduction
The FIND-S algorithm is a simple machine learning algorithm used to find the most specific hypothesis that fits all the positive examples in a given dataset. It is one of the foundational algorithms in the field of concept learning.

Installation
To run the code in this repository, you need to have Python installed along with the following libraries:

pandas

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

License
This project is licensed under the MIT License.


**2
CODE**

Step 1: Create the tennis.csv file
csv_content = """Outlook,Temperature,Humidity,Windy,PlayTennis
Sunny,Hot,High,False,No
Sunny,Hot,High,True,No
Overcast,Hot,High,False,Yes
Rain,Mild,High,False,Yes
Rain,Cool,Normal,False,Yes
Rain,Cool,Normal,True,No
Overcast,Cool,Normal,True,Yes
Sunny,Mild,High,False,No
Sunny,Cool,Normal,False,Yes
Rain,Mild,Normal,False,Yes
Sunny,Mild,Normal,True,Yes
Overcast,Mild,High,True,Yes
Overcast,Hot,Normal,False,Yes
Rain,Mild,High,True,No"""

with open('tennis.csv', 'w') as file:
file.write(csv_content)

print("tennis.csv file created successfully.")

Step 2: Implement the Candidate-Elimination algorithm
import pandas as pd

def load_csv(filename):
return pd.read_csv(filename)

def candidate_elimination(data):
# Initialize G to the most general hypotheses
G = set([("?", "?", "?", "?", "?")])
# Initialize S to the most specific hypotheses
S = ["ϕ", "ϕ", "ϕ", "ϕ", "ϕ"]

for index, instance in data.iterrows():
    instance = instance.tolist()  # Convert series to list for easier handling
    if instance[-1] == 'Yes':  # Positive instance
        G = {g for g in G if consistent(g, instance)}
        S = generalize_S(S, instance)
    else:  # Negative instance
        S = {s for s in S if consistent(s, instance)}
        G = specialize_G(G, instance, data)

    # Remove hypotheses from G that are more specific than any hypothesis in S
    G = {g for g in G if any(more_general(g, s) for s in S)}

return S, G
def consistent(hypothesis, instance):
for h, i in zip(hypothesis, instance):
if h != "?" and h != i:
return False
return True

def more_general(h1, h2):
more_general_parts = []
for x, y in zip(h1, h2):
mg = x == "?" or (x != "ϕ" and (x == y or y == "ϕ"))
more_general_parts.append(mg)
return all(more_general_parts)

def generalize_S(S, instance):
S_new = list(S)
for i in range(len(S_new)):
if S_new[i] == "ϕ":
S_new[i] = instance[i]
elif S_new[i] != instance[i]:
S_new[i] = "?"
return S_new

def specialize_G(G, instance, data):
G_new = set(G)
for g in G:
for i in range(len(g)):
if g[i] == "?":
for val in set(data.iloc[:, i]):
if val != instance[i]:
g_new = list(g)
g_new[i] = val
G_new.add(tuple(g_new))
G_new.remove(g)
return G_new

Load data
data = load_csv('tennis.csv')

Apply Candidate-Elimination algorithm
S, G = candidate_elimination(data)

print("S: ", S)
print("G: ", G)


**Experiment 3**
Create the tennis.csv file
csv_content = """Outlook,Temperature,Humidity,Windy,PlayTennis
Sunny,Hot,High,False,No
Sunny,Hot,High,True,No
Overcast,Hot,High,False,Yes
Rain,Mild,High,False,Yes
Rain,Cool,Normal,False,Yes
Rain,Cool,Normal,True,No
Overcast,Cool,Normal,True,Yes
Sunny,Mild,High,False,No
Sunny,Cool,Normal,False,Yes
Rain,Mild,Normal,False,Yes
Sunny,Mild,Normal,True,Yes
Overcast,Mild,High,True,Yes
Overcast,Hot,Normal,False,Yes
Rain,Mild,High,True,No"""

with open('tennis.csv', 'w') as file:
file.write(csv_content)

print("tennis.csv file created successfully.")
import pandas as pd
import numpy as np

Load the dataset
data = pd.read_csv('tennis.csv')

Function to calculate entropy
def entropy(target_col):
elements, counts = np.unique(target_col, return_counts=True)
entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
return entropy

Function to calculate information gain
def InfoGain(data, split_attribute_name, target_name="PlayTennis"):
total_entropy = entropy(data[target_name])
vals, counts = np.unique(data[split_attribute_name], return_counts=True)
weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
Information_Gain = total_entropy - weighted_entropy
return Information_Gain

Function to implement the ID3 algorithm
def ID3(data, original_data, features, target_attribute_name="PlayTennis", parent_node_class=None):
if len(np.unique(data[target_attribute_name])) <= 1:
return np.unique(data[target_attribute_name])[0]
elif len(data) == 0:
return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
elif len(features) == 0:
return parent_node_class
else:
parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
best_feature_index = np.argmax(item_values)
best_feature = features[best_feature_index]
tree = {best_feature: {}}
features = [i for i in features if i != best_feature]
for value in np.unique(data[best_feature]):
value = value
sub_data = data.where(data[best_feature] == value).dropna()
subtree = ID3(sub_data, data, features, target_attribute_name, parent_node_class)
tree[best_feature][value] = subtree
return tree

Function to classify a new sample
def classify(sample, tree):
for attribute in list(sample.keys()):
if attribute in list(tree.keys()):
try:
result = tree[attribute][sample[attribute]]
except:
return None
result = tree[attribute][sample[attribute]]
if isinstance(result, dict):
return classify(sample, result)
else:
return result

Run ID3 algorithm
features = list(data.columns[:-1])
tree = ID3(data, data, features)
print("Decision Tree:\n", tree)

Classify a new sample
sample = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Windy': 'True'}
classification = classify(sample, tree)
print("Sample:", sample)
print("Classification:", classification)


**Experiment 4**
import numpy as np

Sigmoid activation function
def sigmoid(x):
return 1 / (1 + np.exp(-x))

Derivative of the sigmoid function
def sigmoid_derivative(x):
return x * (1 - x)

Train the neural network using backpropagation
def train(X, y, epochs, learning_rate):
input_layer_neurons = X.shape[1] # Number of features in input data
hidden_layer_neurons = 2 # Number of neurons in the hidden layer
output_neurons = 1 # Number of neurons in the output layer

# Random weight and bias initialization
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bh = np.random.uniform(size=(1, hidden_layer_neurons))
wout = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_input = np.dot(X, wh) + bh
    hidden_layer_activation = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_activation, wout) + bout
    output = sigmoid(output_layer_input)

    # Compute the loss
    loss = y - output
    if (epoch+1) % 1000 == 0:
        print(f"Epoch {epoch+1}, Loss: {np.mean(np.abs(loss))}")

    # Backpropagation
    d_output = loss * sigmoid_derivative(output)

    error_hidden_layer = d_output.dot(wout.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)

    # Update weights and biases
    wout += hidden_layer_activation.T.dot(d_output) * learning_rate
    bout += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    wh += X.T.dot(d_hidden_layer) * learning_rate
    bh += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

return wh, bh, wout, bout
Predict function
def predict(X, wh, bh, wout, bout):
hidden_layer_input = np.dot(X, wh) + bh
hidden_layer_activation = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_activation, wout) + bout
output = sigmoid(output_layer_input)
return output
XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

Training parameters
epochs = 10000
learning_rate = 0.1

Train the neural network
wh, bh, wout, bout = train(X, y, epochs, learning_rate)

Predict on the XOR dataset
predictions = predict(X, wh, bh, wout, bout)
print("Predictions:")
print(predictions)

Classify the predictions as 0 or 1
classifications = np.round(predictions)
print("Classifications:")
print(classifications)

**
Experiment 5
**
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

Standardize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Function to calculate the Euclidean distance between two points
def euclidean_distance(x1, x2):
return np.sqrt(np.sum((x1 - x2) ** 2))

Function to find the k nearest neighbors
def get_neighbors(X_train, y_train, test_instance, k):
distances = []
for i in range(len(X_train)):
dist = euclidean_distance(X_train[i], test_instance)
distances.append((y_train[i], dist))
distances.sort(key=lambda x: x[1])
neighbors = [distances[i][0] for i in range(k)]
return neighbors

Function to make a prediction based on the nearest neighbors
def predict(X_train, y_train, test_instance, k):
neighbors = get_neighbors(X_train, y_train, test_instance, k)
prediction = max(set(neighbors), key=neighbors.count)
return prediction

Function to evaluate the K-NN algorithm
def evaluate_knn(X_train, y_train, X_test, y_test, k):
predictions = []
for test_instance in X_test:
prediction = predict(X_train, y_train, test_instance, k)
predictions.append(prediction)
accuracy = np.sum(predictions == y_test) / len(y_test)
return accuracy

Set the value of k
k = 3

Evaluate the K-NN algorithm
accuracy = evaluate_knn(X_train, y_train, X_test, y_test, k)
print(f"Accuracy: {accuracy * 100:.2f}%")

Test the K-NN algorithm on a new sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
new_sample = scaler.transform(new_sample)
prediction = predict(X_train, y_train, new_sample[0], k)
print(f"Prediction for the new sample: {iris.target_names[prediction]}")

**
EXPERIMENT 6
**
6import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

Standardize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Train the Naïve Bayes classifier
nb = GaussianNB()
nb.fit(X_train, y_train)

Make predictions on the test set
y_pred = nb.predict(X_test)

Calculate the confusion matrix and accuracy
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

Display the accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")

**
Experiment 7
**
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

Standardize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Train the Logistic Regression classifier
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

Make predictions on the test set
y_pred = lr.predict(X_test)

Calculate the confusion matrix and accuracy
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

Display the accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")

**
Experiment 8
**
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

Load the California Housing dataset
california = fetch_california_housing()
X, y = california.data, california.target

Standardize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

Make predictions on the test set
y_pred = lr.predict(X_test)

Calculate the mean squared error and R² score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

Display the results
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

**
Experiment 9
**
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

Load the California Housing dataset
california = fetch_california_housing()
X, y = california.data, california.target

Use only one feature for simplicity in polynomial regression
X = X[:, 0].reshape(-1, 1) # Using the first feature (MedInc)

Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Standardize the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

Polynomial Regression
poly = PolynomialFeatures(degree=3) # Polynomial features of degree 3
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)

Calculate metrics for Linear Regression
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

Calculate metrics for Polynomial Regression
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

Display the results
print(f"Linear Regression Mean Squared Error: {mse_lr:.2f}")
print(f"Linear Regression R² Score: {r2_lr:.2f}")
print(f"Polynomial Regression Mean Squared Error: {mse_poly:.2f}")
print(f"Polynomial Regression R² Score: {r2_poly:.2f}")

Plot the results
plt.figure(figsize=(14, 6))

Linear Regression Plot
plt.subplot(1, 2, 1)
plt.scatter(X_test_scaled, y_test, color='red', edgecolors='k', label='Actual')
plt.plot(X_test_scaled, y_pred_lr, color='blue', label='Linear Regression')
plt.title('Linear Regression')
plt.xlabel('Median Income (scaled)')
plt.ylabel('House Price')
plt.legend()

Polynomial Regression Plot
plt.subplot(1, 2, 2)
X_range = np.linspace(X_test_scaled.min(), X_test_scaled.max(), 100).reshape(-1, 1)
X_range_poly = poly.transform(scaler.transform(X_range))
y_range_poly = poly_reg.predict(X_range_poly)
plt.scatter(X_test_scaled, y_test, color='red', edgecolors='k', label='Actual')
plt.plot(X_range, y_range_poly, color='blue', label='Polynomial Regression')
plt.title('Polynomial Regression (Degree 3)')
plt.xlabel('Median Income (scaled)')
plt.ylabel('House Price')
plt.legend()

plt.tight_layout()
plt.show()
**
Experiment 10
**
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

Load the California Housing dataset
california = fetch_california_housing()
X, y = california.data, california.target

Use only one feature for simplicity in polynomial regression
X = X[:, 0].reshape(-1, 1) # Using the first feature (MedInc)

Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Standardize the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

Polynomial Regression
poly = PolynomialFeatures(degree=3) # Polynomial features of degree 3
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)

Calculate metrics for Linear Regression
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

Calculate metrics for Polynomial Regression
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

Display the results
print(f"Linear Regression Mean Squared Error: {mse_lr:.2f}")
print(f"Linear Regression R² Score: {r2_lr:.2f}")
print(f"Polynomial Regression Mean Squared Error: {mse_poly:.2f}")
print(f"Polynomial Regression R² Score: {r2_poly:.2f}")

Plot the results
plt.figure(figsize=(14, 6))

Linear Regression Plot
plt.subplot(1, 2, 1)
plt.scatter(X_test_scaled, y_test, color='red', edgecolors='k', label='Actual')
plt.plot(X_test_scaled, y_pred_lr, color='blue', label='Linear Regression')
plt.title('Linear Regression')
plt.xlabel('Median Income (scaled)')
plt.ylabel('House Price')
plt.legend()

Polynomial Regression Plot
plt.subplot(1, 2, 2)
X_range = np.linspace(X_test_scaled.min(), X_test_scaled.max(), 100).reshape(-1, 1)
X_range_poly = poly.transform(scaler.transform(X_range))
y_range_poly = poly_reg.predict(X_range_poly)
plt.scatter(X_test_scaled, y_test, color='red', edgecolors='k', label='Actual')
plt.plot(X_range, y_range_poly, color='blue', label='Polynomial Regression')
plt.title('Polynomial Regression (Degree 3)')
plt.xlabel('Median Income (scaled)')
plt.ylabel('House Price')
plt.legend()

plt.tight_layout()
plt.show()

**
Exp 11
**
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

Generate synthetic dataset
np.random.seed(0)
n_samples = 1000

data = {
'age': np.random.randint(18, 70, size=n_samples),
'income': np.random.randint(20000, 120000, size=n_samples),
'credit_score': np.random.randint(300, 850, size=n_samples),
'loan_amount': np.random.randint(1000, 50000, size=n_samples),
'class': np.random.choice(['Good', 'Bad'], size=n_samples) # Target variable
}

df = pd.DataFrame(data)

Prepare features and target variable
X = df[['age', 'income', 'credit_score', 'loan_amount']]
y = df['class']

Encode target variable
y = y.map({'Good': 1, 'Bad': 0})

Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Standardize the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

Make predictions on the test set
y_pred = model.predict(X_test_scaled)

Calculate metrics
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

Display the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)

Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

**
Exp 12
**
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Train the KNN model
k = 5 # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

Make predictions on the test set
y_pred = knn.predict(X_test_scaled)

Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_names)

Display the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)

Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
