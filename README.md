<h1 align="center"><strong>Contextual Diet ChatBot with Recommendation System</strong></h1>

Welcome to the **Contextual Diet ChatBot with Recommendation System** repository. This project is designed to provide personalized diet recommendations based on a user's basal metabolic rate (BMR), using a combination of machine learning techniques and a feed-forward neural network. The chatbot takes into account factors like age, weight, and height to suggest appropriate diet plans through a Tkinter-based graphical user interface.

## Repository Structure

This repository contains the following files:

1. **Presentation.pdf**: The presentation file used to present the final project.
2. **Report.pdf**: The final project report in PDF format.
3. **app.py**: The GUI implementation using Tkinter.
4. **chat.py**: Generates responses for the user based on the model's predictions.
5. **data.pth**: The saved neural network model file.
6. **intents.json**: Contains user prompts and associated intents (e.g., "hi", "hello" for greetings).
7. **meals.csv**: Dataset with meal information.
8. **nutrients.csv**: Dataset with nutrient information.
9. **model.py**: Defines the feed-forward neural network architecture.
10. **mychatbot.py**: Includes methods for model training using TF-IDF, bag-of-words, and other techniques.
11. **nltk_utils.py**: Helper functions for tokenization, stemming, and bag-of-words.
12. **recsystem.py**: Implements the recommendation system for meal suggestions.
13. **trainmodel.py**: Detailed model training processes including alternative models like Random Forest.

## Project Overview

### Objective

The primary objective of this project is to offer personalized diet recommendations by analyzing the user's basal metabolic rate (BMR) and other health metrics. The chatbot uses machine learning techniques to predict the most suitable diet plans, helping users to manage their nutrition more effectively.

### Machine Learning Techniques Used

- **Feed-Forward Neural Network**: The core model used to make predictions based on user input.
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: A text vectorization technique used during model training.
- **Bag-of-Words**: Another text vectorization technique employed for training the model.
- **Random Forest (Exploratory)**: Alternative model explored in the training process to compare results.

### GUI Implementation

The chatbot uses a Tkinter-based graphical user interface (GUI) to interact with users. The GUI collects user information such as age, weight, and height, and uses this data to recommend diet plans through the trained model.

## Getting Started

To explore the contents of this repository and run the application, follow these steps:

1. **Clone the Repository**:
   - Clone this repository to your local machine using Git:

     ```bash
     git clone https://github.com/mehmetalpayy/Contextual-Diet-ChatBot-with-Recommendation-System.git
     ```

2. **Install Dependencies**:
   - Ensure you have Python installed on your system. Then, manually install the required libraries using pip:

     ```bash
     pip install torch nltk pandas scikit-learn
     ```

   > **Note**: `tkinter` is usually included with Python, but if needed, you can install it separately.

3. **Run the Application**:
   - Execute the following command to launch the chatbot application:

     ```bash
     python app.py
     ```

4. **Interact with the ChatBot**:
   - Once the application is running, you can interact with the chatbot through the GUI, entering your personal information to receive diet recommendations.


## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your modifications and commit them.
4. Submit a pull request with a detailed description of the changes.

## Contact

For any questions or feedback, please contact me at [mehmetcompeng@gmail.com](mailto:mehmetcompeng@gmail.com).

---

Thank you for visiting the Contextual Diet ChatBot with Recommendation System repository. I hope you find the project useful and informative!
