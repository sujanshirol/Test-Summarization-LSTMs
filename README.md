# Test Summarization using LSTM Encoder-Decoder

Text summarization is the problem of creating a short, accurate, and fluent summary of a longer text document. It is a Natural Language Processing application which produces short and meaningful summary of a lengthy paragraph thereby helping us to understand the essence of the topic in an efficient way.

Automatic text summarization methods are greatly needed to address the ever-growing amount of text data available online to both better help discover relevant information and to consume relevant information faster.

Types of Text Summarization
1. **Abstrative Based**: In Abstractive based, we generate new sentences from the original text. The sentences generated through abstractive summarization might not be present in the original text.

2. **Extractive Based**: In Extractive based, we identify the important sentences or phrases from the original text and extract only those from the text. Those extracted sentences would be our summary.

## Problem Statement
Customer reviews can be lengthy and detailed. Manually analysing these reviews, as you might guess, takes a long time. This is where Natural Language Processing's application can be put to use to develop a short summary for lengthy reviews.

Our objective here is to generate a summary for the "Amazon Fine Food reviews" using the abstraction-based and as well as extraction-based text summarization approaches.

Data Scource: [Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews?select=Reviews.csv)

## Project pipeline
1. Understanding Text Summarization,
2. Text pre-processing,
3. Abstractive Text Summarization using LSTM, ENCODER-DECODER architecture,
4. Web scrape an article using BS4.
5. Extractive Text Summarization using Transformer,

## Results
### Abstractive Text Summarization 

Original text: ``gluten free want crackers one also delicious second order``<br>
Predicted summary:  ``great gluten free baking``


Original text: ``coffee tastes like regular coffee good weak used little less water regular strength coffee taste great smell better whether works stomach needed sure ``<br>
Predicted summary:  ``good coffee``


Original text: ``first time italy wife actually give gifts well good sucks seasonal definitely makes look forward holidays``<br>
Predicted summary:  ``the best``

### Extractive Text Summarization

Original text:<br>
```Artificial intelligence AI is intelligence demonstrated by machines as opposed to the natural intelligence displayed by humans or animals Leading AI textbooks define the field as the study of intelligent agents any system that perceives its environment and takes actions that maximize its chance of achieving its goals a Some popular accounts use the term artificial intelligence to describe machines that mimic cognitive functions that humans associate with the human mind such as learning and prob```

Predicted summary:<br>
```artificial intelligence (AI) is intelligence demonstrated by machines as opposed to the natural intelligence displayed by humans or animals . leading AI textbooks define the field as the study of intelligent agents.```





Thanks to Aravind Pai's [article](https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/)
