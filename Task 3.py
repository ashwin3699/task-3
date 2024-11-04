import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, accuracy_score
import graphviz

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'
df = pd.read_csv(url, compression='zip', sep=';')

# Display basic information
print("Dataset Preview:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nDataset Description:")
print(df.describe())

# Handle categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split the dataset into features and target variable
X = df.drop('y', axis=1)  # Features
y = df['y']               # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the classifier
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize the decision tree
dot_data = export_graphviz(clf, out_file=None,
                           feature_names=X.columns,
                           class_names=['no', 'yes'],  # Assuming 'no' and 'yes' are the classes in 'y'
                           filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Save the visualization to a file
graph.view()  # Open the visualization

