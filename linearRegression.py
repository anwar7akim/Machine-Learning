import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Manually create data for study hours and test scores
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 10, 10])
test_scores = np.array([40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 48, 53, 58, 63, 68, 73, 78, 83, 61, 66, 71, 76, 81, 86, 91, 92])  # Aligned with study effort

# Prepare the data for linear regression
X = study_hours.reshape(-1, 1)
y = test_scores

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Function to predict test score based on input study hours
def predict_score(hours):
    predicted_score = model.predict([[hours]])[0]
    print(f"Predicted Test Score for {hours} hours of study: {predicted_score:.2f}%")

    # Plot the regression line and the predicted point
    plt.scatter(study_hours, test_scores, color='blue', label='Data Points')
    plt.plot(study_hours, model.predict(X), color='red', label='Regression Line')
    plt.scatter([hours], [predicted_score], color='green', label='Your Input', zorder=5)
    plt.axvline(x=hours, color='green', linestyle='--', label='Study Hours Line')
    plt.title('Study Hours vs Test Scores (Prediction)')
    plt.xlabel('Study Hours')
    plt.ylabel('Test Scores')
    plt.legend()
    plt.show()

# User input for study hours
user_input = float(input("Enter the number of hours studied: "))
predict_score(user_input)