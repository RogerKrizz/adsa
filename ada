X_train = df_enc.drop("Heart_Disease", axis = 1)
y_train = df_enc["Heart_Disease"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

model = AdaBoostClassifier(
    n_estimators = 100,
    random_state = 42
)

model.fit(X_train, y_train)
results = model.predict(X_test)

print("Ada Boost Accuracy:", accuracy_score(y_test, results))
print("\nClassification Report:\n", classification_report(y_test, results))
