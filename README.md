# Virtual Data Science Internship at Bharat-Intern

# Task 1
## Problem statement :

## TO PREDICT THE STOCK PRICE OF ANY COMPANY USING LSTM.
### ABOUT DATASET :
This dataset contains historical data of Adani Enterprise stock prices and related attributes. It consists of 7 columns and a smaller subset of 745 rows. Each column represents a specific attribute, and each row contains the corresponding values for that attribute.

The columns in the dataset are as follows:

    Symbol: The name of the company, which is Adani Enterprise (ADANIENT) in this case.
    Date: The year and date of the stock data.
    Close: The closing price of ADANIENT's stock on a particular day.
    High: The highest value reached by ADANIENT's stock on the given day.
    Low: The lowest value reached by ADANIENT's stock on the given day.
    Open: The opening value of ADANIENT's stock on the given day.
    Volume: The trading volume of ADANIENT's stock on the given day, i.e., the number of shares traded.
    adjClose: The adjusted closing price of ADANIENT's stock, considering factors such as dividends and stock splits.

Dataset : https://finance.yahoo.com/quote/ADANIENT.NS/history/                                                     
Solution : https://github.com/ansariparvej/Data-Science-Internship_Bharat-Intern/blob/main/Data%20Science/Task_1%20Stock%20Price%20Prediction%20Using%20LSTM/Task_1%20Stock%20Price%20Prediction%20Using%20LSTM.ipynb

------------------------------------------------------------------------------------------------

# Task 2
## Problem statement :

## Titanic Classification : Algorithm which tells whether the person will be save from sinking or not
### About Dataset :
Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we have to a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

PassengerId: Passenger Identity	
Survived: Whether passenger survived or not	(0 = No, 1 = Yes)
Pclass:	Class of ticket, a proxy for socio-economic status (SES)	(1 = 1st, 2 = 2nd, 3 = 3rd)
Name:	Name of passenger	
Sex:	Sex of passenger	
Age:	Age of passenger in years	
SibSp: Number of sibling and/or spouse travelling with passenger	
Parch:	Number of parent and/or children travelling with passenger	
Ticket: Ticket number	
Fare: Price of ticket	
Cabin: Cabin number	
Embarked: Port of embarkation	(C = Cherbourg, Q = Queenstown, S = Southampton)

Dataset : https://www.kaggle.com/datasets/rahulsah06/titanic                                                                                   
Solution : https://github.com/ansariparvej/Data-Science-Internship_Bharat-Intern/blob/main/Data%20Science/Task_2%20Titanic%20Classification/Task_2%20Titanic%20Classification.ipynb

------------------------------------------------------------------------------------------------

# Task 3
## Problem statement :

## Number Recognition : MNIST Handwritten Digits Recognition using Convolution Neural Network (CNN):

### About Dataset :
Applying a Convolutional Neural Network (CNN) on the MNIST dataset is a popular way to learn about and demonstrate the capabilities of CNNs for image classification tasks. The MNIST dataset consists of 28×28 grayscale images of hand-written digits (0-9), with a training set of 60,000 examples and a test set of 10,000 examples.
Here is a basic approach to applying a CNN on the MNIST dataset using the Python programming language and the Keras library:

    1. Load and preprocess the data: The MNIST dataset can be loaded using the Keras library, and the images can be normalized to have pixel values between 0 and 1.
    2. Define the model architecture: The CNN can be constructed using the Keras Sequential API, which allows for easy building of sequential models layer-by-layer. The architecture should typically include convolutional layers, pooling layers, and fully-connected layers.
    3. Compile the model: The model needs to be compiled with a loss function, an optimizer, and a metric for evaluation.
    4. Train the model: The model can be trained on the training set using the Keras fit() function. It is important to monitor the training accuracy and loss to ensure the model is converging properly.
    5. Evaluate the model: The trained model can be evaluated on the test set using the Keras evaluate() function. The evaluation metric typically used for classification tasks is accuracy.

References:

1. MNIST dataset: http://yann.lecun.com/exdb/mnist/
2. Keras documentation: https://keras.io/
3. “Deep Learning with Python” by Francois Chollet (https://www.manning.com/books/deep-learning-with-python)


Dataset : http://yann.lecun.com/exdb/mnist/                                                                                   
Solution : https://github.com/ansariparvej/Data-Science-Internship_Bharat-Intern/blob/main/Data%20Science/Task_3%20Handwritten_digit_recognition/Task_3%20Handwritten_digit_recognition.ipynb
