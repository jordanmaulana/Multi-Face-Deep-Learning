# Multi-Face-Deep-Learning

to proceed this work, **MAKE SURE YOU'VE TRIED THIS**

https://github.com/jordanmaulana/YourOwnFaceRecognition

to know what is going on and to know the basic requirements

## THE GOAL

The goal is to classify names of people that exist in your own dataset

## REQUIREMENTS (addition from previous work)

- **pickle**

  run `pip install pickle`

  It allows you to save your program's variable into a file. Even it is not readable like txt or csv files, it is easier to save and load the data. No need to make certain algorithm to load what you've saved.

## Algorithm Differences

- This version allows your program to train and recognize as many people as you want. Do it with the same step as previous version.

- It saves names in python `List` and use it's index as category, and load it as you run the last program `deploy2.py`

- `one hot` label is now inside tflearn config, while the previous version was defined directly at training and deploy program. Check this link to understand what one hot encoding is.

  https://towardsdatascience.com/choosing-the-right-encoding-method-label-vs-onehot-encoder-a4434493149b



Happy Learning! ^_^
