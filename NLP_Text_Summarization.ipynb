{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# TEXT SUMMARIZATION"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Problem Statement"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Customer reviews can be lengthy and detailed. Manually analysing these reviews, as you might guess, takes a long time. This is where Natural Language Processing's application can be put to use to develop a short summary for lengthy reviews.\n",
    "\n",
    "Our objective here is to generate a summary for the **\"Amazon Fine Food reviews\"** using the **abstraction-based** and as well as **extraction-based** text summarization approaches."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Project pipeline\n",
    "\n",
    "1. Understanding Text Summarization\n",
    "2. Text pre-processing\n",
    "3. Abstractive Text Summarization using LSTM, ENCODER-DECODER architecture\n",
    "4. Extractive Text Summarization using Transformer"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Understanding Text Summarization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Text summarization** is a Natural Language Processing application which produces short and meaningful summary of a lengthy paragraph thereby helping us to understand the essence of the topic in an efficient way.\n",
    "\n",
    "**Types of Text Summarization**\n",
    "\n",
    "1. Abstrative Based\n",
    "2. Extractive Based\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In Abstractive based, we generate new sentences from the original text. The sentences generated through abstractive summarization might not be present in the original text.\n",
    "\n",
    "In Extractive based, we identify the important sentences or phrases from the original text and extract only those from the text. Those extracted sentences would be our summary. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Custom Attention Layer: Keras does not officially support attention layer. We will use a third-party implementation."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "JUValOzcHtEK"
   },
   "source": [
    "## Import the Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:19.127820Z",
     "iopub.status.busy": "2021-07-18T10:06:19.127344Z",
     "iopub.status.idle": "2021-07-18T10:06:19.142478Z",
     "shell.execute_reply": "2021-07-18T10:06:19.141188Z",
     "shell.execute_reply.started": "2021-07-18T10:06:19.127775Z"
    },
    "id": "_Jpu8qLEFxcY",
    "outputId": "95968e01-faac-4911-c802-9c008a4e62cf"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd \n",
    "import re\n",
    "from bs4 import BeautifulSoup\n",
    "from keras.preprocessing.text import Tokenizer\n",
    "from attention import AttentionLayer \n",
    "from keras.preprocessing.sequence import pad_sequences\n",
    "from nltk.corpus import stopwords\n",
    "from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed\n",
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.callbacks import EarlyStopping\n",
    "import warnings\n",
    "pd.set_option(\"display.max_colwidth\", 200)\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "UVakjZ3oICgx"
   },
   "source": [
    "## Read the dataset\n",
    "\n",
    "This dataset consists of reviews of fine foods from Amazon. The data spans a period of more than 10 years, including all ~500,000 reviews up to October 2012. These reviews include product and user information, ratings, plain text review, and summary. It also includes reviews from all other Amazon categories.\n",
    "\n",
    "We’ll take a sample of 50,000 reviews to reduce the training time of our model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:19.147403Z",
     "iopub.status.busy": "2021-07-18T10:06:19.147019Z",
     "iopub.status.idle": "2021-07-18T10:06:19.528279Z",
     "shell.execute_reply": "2021-07-18T10:06:19.526625Z",
     "shell.execute_reply.started": "2021-07-18T10:06:19.147372Z"
    },
    "id": "wnK5o4Z1Fxcj"
   },
   "outputs": [],
   "source": [
    "data = pd.read_csv(\"../input/amazon-fine-food-reviews/Reviews.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "kGNQKvCaISIn"
   },
   "source": [
    "## Drop Duplicates and NA values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:19.532154Z",
     "iopub.status.busy": "2021-07-18T10:06:19.531613Z",
     "iopub.status.idle": "2021-07-18T10:06:19.657223Z",
     "shell.execute_reply": "2021-07-18T10:06:19.655957Z",
     "shell.execute_reply.started": "2021-07-18T10:06:19.532104Z"
    },
    "id": "Cjul88oOFxcr"
   },
   "outputs": [],
   "source": [
    "data.drop_duplicates(subset=['Text'],inplace=True)\n",
    "data.dropna(axis=0,inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "r0xLYACiFxdJ"
   },
   "source": [
    "## 2. Text Pre-processing\n",
    "\n",
    "Before we start developing the model, we must first complete some basic preprocessing tasks. Using messy and sloppy text data can be devastating. As a result, in this stage, we will remove all unneeded symbols, characters, and other elements from the text that do not affect the problem's goal.\n",
    "\n",
    "Here is the dictionary that we will use for expanding the contractions:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:19.707818Z",
     "iopub.status.busy": "2021-07-18T10:06:19.707224Z",
     "iopub.status.idle": "2021-07-18T10:06:19.725252Z",
     "shell.execute_reply": "2021-07-18T10:06:19.723806Z",
     "shell.execute_reply.started": "2021-07-18T10:06:19.707748Z"
    },
    "id": "0s6IY-x2FxdL"
   },
   "outputs": [],
   "source": [
    "contraction_mapping = {\"ain't\": \"is not\", \"aren't\": \"are not\",\"can't\": \"cannot\", \"'cause\": \"because\", \"could've\": \"could have\", \"couldn't\": \"could not\",\n",
    "                           \"didn't\": \"did not\",  \"doesn't\": \"does not\", \"don't\": \"do not\", \"hadn't\": \"had not\", \"hasn't\": \"has not\", \"haven't\": \"have not\",\n",
    "                           \"he'd\": \"he would\",\"he'll\": \"he will\", \"he's\": \"he is\", \"how'd\": \"how did\", \"how'd'y\": \"how do you\", \"how'll\": \"how will\", \"how's\": \"how is\",\n",
    "                           \"I'd\": \"I would\", \"I'd've\": \"I would have\", \"I'll\": \"I will\", \"I'll've\": \"I will have\",\"I'm\": \"I am\", \"I've\": \"I have\", \"i'd\": \"i would\",\n",
    "                           \"i'd've\": \"i would have\", \"i'll\": \"i will\",  \"i'll've\": \"i will have\",\"i'm\": \"i am\", \"i've\": \"i have\", \"isn't\": \"is not\", \"it'd\": \"it would\",\n",
    "                           \"it'd've\": \"it would have\", \"it'll\": \"it will\", \"it'll've\": \"it will have\",\"it's\": \"it is\", \"let's\": \"let us\", \"ma'am\": \"madam\",\n",
    "                           \"mayn't\": \"may not\", \"might've\": \"might have\",\"mightn't\": \"might not\",\"mightn't've\": \"might not have\", \"must've\": \"must have\",\n",
    "                           \"mustn't\": \"must not\", \"mustn't've\": \"must not have\", \"needn't\": \"need not\", \"needn't've\": \"need not have\",\"o'clock\": \"of the clock\",\n",
    "                           \"oughtn't\": \"ought not\", \"oughtn't've\": \"ought not have\", \"shan't\": \"shall not\", \"sha'n't\": \"shall not\", \"shan't've\": \"shall not have\",\n",
    "                           \"she'd\": \"she would\", \"she'd've\": \"she would have\", \"she'll\": \"she will\", \"she'll've\": \"she will have\", \"she's\": \"she is\",\n",
    "                           \"should've\": \"should have\", \"shouldn't\": \"should not\", \"shouldn't've\": \"should not have\", \"so've\": \"so have\",\"so's\": \"so as\",\n",
    "                           \"this's\": \"this is\",\"that'd\": \"that would\", \"that'd've\": \"that would have\", \"that's\": \"that is\", \"there'd\": \"there would\",\n",
    "                           \"there'd've\": \"there would have\", \"there's\": \"there is\", \"here's\": \"here is\",\"they'd\": \"they would\", \"they'd've\": \"they would have\",\n",
    "                           \"they'll\": \"they will\", \"they'll've\": \"they will have\", \"they're\": \"they are\", \"they've\": \"they have\", \"to've\": \"to have\",\n",
    "                           \"wasn't\": \"was not\", \"we'd\": \"we would\", \"we'd've\": \"we would have\", \"we'll\": \"we will\", \"we'll've\": \"we will have\", \"we're\": \"we are\",\n",
    "                           \"we've\": \"we have\", \"weren't\": \"were not\", \"what'll\": \"what will\", \"what'll've\": \"what will have\", \"what're\": \"what are\",\n",
    "                           \"what's\": \"what is\", \"what've\": \"what have\", \"when's\": \"when is\", \"when've\": \"when have\", \"where'd\": \"where did\", \"where's\": \"where is\",\n",
    "                           \"where've\": \"where have\", \"who'll\": \"who will\", \"who'll've\": \"who will have\", \"who's\": \"who is\", \"who've\": \"who have\",\n",
    "                           \"why's\": \"why is\", \"why've\": \"why have\", \"will've\": \"will have\", \"won't\": \"will not\", \"won't've\": \"will not have\",\n",
    "                           \"would've\": \"would have\", \"wouldn't\": \"would not\", \"wouldn't've\": \"would not have\", \"y'all\": \"you all\",\n",
    "                           \"y'all'd\": \"you all would\",\"y'all'd've\": \"you all would have\",\"y'all're\": \"you all are\",\"y'all've\": \"you all have\",\n",
    "                           \"you'd\": \"you would\", \"you'd've\": \"you would have\", \"you'll\": \"you will\", \"you'll've\": \"you will have\",\n",
    "                           \"you're\": \"you are\", \"you've\": \"you have\"}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "2JFRXFHmI7Mj"
   },
   "source": [
    "We will perform the below pre-processing tasks for our data:\n",
    "\n",
    "1.Convert everything to lowercase\n",
    "\n",
    "2.Remove HTML tags\n",
    "\n",
    "3.Contraction mapping\n",
    "\n",
    "4.Remove (‘s)\n",
    "\n",
    "5.Remove any text inside the parenthesis ( )\n",
    "\n",
    "6.Eliminate punctuations and special characters\n",
    "\n",
    "7.Remove stopwords\n",
    "\n",
    "8.Remove single characters\n",
    "\n",
    "**Let’s define the function for performing the above pre-processing steps**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "stop_words = set(stopwords.words('english')) \n",
    "\n",
    "def text_cleaner(text,num):\n",
    "    # lower\n",
    "    newString = text.lower()\n",
    "    # remove HTML\n",
    "    newString = BeautifulSoup(newString, \"lxml\").text\n",
    "    # Remove any text inside the parenthesis\n",
    "    newString = re.sub(r'\\([^)]*\\)', '', newString)\n",
    "    # remove double quotes\n",
    "    newString = re.sub('\"','', newString)\n",
    "    # contraction mapping\n",
    "    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(\" \")])  \n",
    "    # remove 's\n",
    "    newString = re.sub(r\"'s\\b\",\"\",newString)\n",
    "    # Eliminate punctuations and special characters\n",
    "    newString = re.sub(\"[^a-zA-Z]\", \" \", newString)\n",
    "    # Remove stopwords\n",
    "    if(num==0):\n",
    "        tokens = [w for w in newString.split() if not w in stop_words]\n",
    "    else:\n",
    "        tokens=newString.split()\n",
    "    long_words=[]\n",
    "    # Remove short words\n",
    "    for i in tokens:\n",
    "        if len(i)>1:                                                 \n",
    "            long_words.append(i)   \n",
    "    return (\" \".join(long_words)).strip()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:19.746650Z",
     "iopub.status.busy": "2021-07-18T10:06:19.746323Z",
     "iopub.status.idle": "2021-07-18T10:06:40.791858Z",
     "shell.execute_reply": "2021-07-18T10:06:40.790710Z",
     "shell.execute_reply.started": "2021-07-18T10:06:19.746619Z"
    },
    "id": "A2QAeCHWFxdY"
   },
   "outputs": [],
   "source": [
    "# Cleaning the \"Text\" Column\n",
    "\n",
    "cleaned_text = []\n",
    "for t in data['Text']:\n",
    "    cleaned_text.append(text_cleaner(t,0))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "snRZY8wjLao2"
   },
   "source": [
    "Let us look at the first 2 preprocessed reviews"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:40.794101Z",
     "iopub.status.busy": "2021-07-18T10:06:40.793640Z",
     "iopub.status.idle": "2021-07-18T10:06:40.805621Z",
     "shell.execute_reply": "2021-07-18T10:06:40.804247Z",
     "shell.execute_reply.started": "2021-07-18T10:06:40.794048Z"
    },
    "id": "NCAIkhWbFxdh",
    "outputId": "c2da1a36-4488-4e32-ef9e-fcfe496e374d"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['bought several vitality canned dog food products found good quality product looks like stew processed meat smells better labrador finicky appreciates product better',\n",
       " 'product arrived labeled jumbo salted peanuts peanuts actually small sized unsalted sure error vendor intended represent product jumbo']"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cleaned_text[:2]  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:40.809550Z",
     "iopub.status.busy": "2021-07-18T10:06:40.808964Z",
     "iopub.status.idle": "2021-07-18T10:06:55.586273Z",
     "shell.execute_reply": "2021-07-18T10:06:55.585067Z",
     "shell.execute_reply.started": "2021-07-18T10:06:40.809458Z"
    },
    "id": "GsRXocxoFxd-"
   },
   "outputs": [],
   "source": [
    "# Cleaning the \"Summary\" Column\n",
    "\n",
    "cleaned_summary = []\n",
    "for t in data['Summary']:\n",
    "    cleaned_summary.append(text_cleaner(t,1))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "oZeD0gs6Lnb-"
   },
   "source": [
    "Let us look at the first 2 preprocessed summaries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:55.588381Z",
     "iopub.status.busy": "2021-07-18T10:06:55.587913Z",
     "iopub.status.idle": "2021-07-18T10:06:55.597768Z",
     "shell.execute_reply": "2021-07-18T10:06:55.596041Z",
     "shell.execute_reply.started": "2021-07-18T10:06:55.588336Z"
    },
    "id": "jQJdZcAzFxee",
    "outputId": "a1fbe683-c03f-4afb-addf-e075021c121b"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['good quality dog food', 'not as advertised']"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cleaned_summary[:2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:55.600770Z",
     "iopub.status.busy": "2021-07-18T10:06:55.600226Z",
     "iopub.status.idle": "2021-07-18T10:06:55.626962Z",
     "shell.execute_reply": "2021-07-18T10:06:55.625731Z",
     "shell.execute_reply.started": "2021-07-18T10:06:55.600718Z"
    },
    "id": "L1zLpnqsFxey"
   },
   "outputs": [],
   "source": [
    "data['cleaned_text']=cleaned_text\n",
    "data['cleaned_summary']=cleaned_summary"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "KT_D2cLiLy77"
   },
   "source": [
    "## Drop empty rows"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:55.629482Z",
     "iopub.status.busy": "2021-07-18T10:06:55.628947Z",
     "iopub.status.idle": "2021-07-18T10:06:55.750104Z",
     "shell.execute_reply": "2021-07-18T10:06:55.748823Z",
     "shell.execute_reply.started": "2021-07-18T10:06:55.629408Z"
    },
    "id": "sYK390unFxfA"
   },
   "outputs": [],
   "source": [
    "data.replace('', np.nan, inplace=True)\n",
    "data.dropna(axis=0,inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Vm8Fk2TCL7Sp"
   },
   "source": [
    "## Understanding the distribution of the sequences\n",
    "\n",
    "Here, we will analyze the length of the reviews and the summary to get an overall idea about the distribution of length of the text. This will help us fix the maximum length of the sequence:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:55.752450Z",
     "iopub.status.busy": "2021-07-18T10:06:55.751924Z",
     "iopub.status.idle": "2021-07-18T10:06:56.401982Z",
     "shell.execute_reply": "2021-07-18T10:06:56.400405Z",
     "shell.execute_reply.started": "2021-07-18T10:06:55.752383Z"
    },
    "id": "MdF76AHHFxgw",
    "outputId": "e3bbe165-4235-482f-bfd4-36a3f1d95290"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "text_word_count = []\n",
    "summary_word_count = []\n",
    "\n",
    "# populate the lists with sentence lengths\n",
    "for i in data['cleaned_text']:\n",
    "      text_word_count.append(len(i.split()))\n",
    "\n",
    "for i in data['cleaned_summary']:\n",
    "      summary_word_count.append(len(i.split()))\n",
    "\n",
    "length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})\n",
    "\n",
    "length_df.hist(bins = 30)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "QwdSGIhGMEbz"
   },
   "source": [
    "**NOTE: We can fix the maximum length of the summary to 8 since that seems to be the majority summary length.**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let us understand the proportion of the length of summaries below 8"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:56.404800Z",
     "iopub.status.busy": "2021-07-18T10:06:56.404287Z",
     "iopub.status.idle": "2021-07-18T10:06:56.456934Z",
     "shell.execute_reply": "2021-07-18T10:06:56.455679Z",
     "shell.execute_reply.started": "2021-07-18T10:06:56.404742Z"
    },
    "id": "7JRjwdIOFxg3",
    "outputId": "f968be82-c539-471d-ce23-16f18b059ea0"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9439323119536052\n"
     ]
    }
   ],
   "source": [
    "cnt=0\n",
    "for i in data['cleaned_summary']:\n",
    "    if(len(i.split()) <= 8):\n",
    "        cnt=cnt+1\n",
    "print(cnt/len(data['cleaned_summary']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "yYB4Ga9KMjEu"
   },
   "source": [
    "**We observe that 94% of the summaries have length below 8. So, we can fix maximum length of summary to 8.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:56.459474Z",
     "iopub.status.busy": "2021-07-18T10:06:56.458715Z",
     "iopub.status.idle": "2021-07-18T10:06:56.601550Z",
     "shell.execute_reply": "2021-07-18T10:06:56.600008Z",
     "shell.execute_reply.started": "2021-07-18T10:06:56.459394Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.5587076949710033\n"
     ]
    }
   ],
   "source": [
    "cnt=0\n",
    "for i in data['cleaned_text']:\n",
    "    if(len(i.split()) <= 30):\n",
    "        cnt=cnt+1\n",
    "print(cnt/len(data['cleaned_text']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let us fix the maximum length of review to 30"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:56.604119Z",
     "iopub.status.busy": "2021-07-18T10:06:56.603684Z",
     "iopub.status.idle": "2021-07-18T10:06:56.609789Z",
     "shell.execute_reply": "2021-07-18T10:06:56.608132Z",
     "shell.execute_reply.started": "2021-07-18T10:06:56.604075Z"
    },
    "id": "ZKD5VOWqFxhC"
   },
   "outputs": [],
   "source": [
    "max_text_len=30\n",
    "max_summary_len=8"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "E6d48E-8M4VO"
   },
   "source": [
    "Let us select the reviews and summaries whose length falls below or equal to **max_text_len** and **max_summary_len**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:56.618017Z",
     "iopub.status.busy": "2021-07-18T10:06:56.617609Z",
     "iopub.status.idle": "2021-07-18T10:06:56.810766Z",
     "shell.execute_reply": "2021-07-18T10:06:56.809616Z",
     "shell.execute_reply.started": "2021-07-18T10:06:56.617984Z"
    },
    "id": "yY0tEJP0FxhI"
   },
   "outputs": [],
   "source": [
    "cleaned_text =np.array(data['cleaned_text'])\n",
    "cleaned_summary=np.array(data['cleaned_summary'])\n",
    "\n",
    "short_text=[]\n",
    "short_summary=[]\n",
    "\n",
    "for i in range(len(cleaned_text)):\n",
    "    if(len(cleaned_summary[i].split())<=max_summary_len and len(cleaned_text[i].split())<=max_text_len):\n",
    "        short_text.append(cleaned_text[i])\n",
    "        short_summary.append(cleaned_summary[i])\n",
    "        \n",
    "df=pd.DataFrame({'text':short_text,'summary':short_summary}) # new dataframe to use"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:56.815121Z",
     "iopub.status.busy": "2021-07-18T10:06:56.814614Z",
     "iopub.status.idle": "2021-07-18T10:06:56.820413Z",
     "shell.execute_reply": "2021-07-18T10:06:56.818859Z",
     "shell.execute_reply.started": "2021-07-18T10:06:56.815072Z"
    }
   },
   "outputs": [],
   "source": [
    "# add the START and END special tokens at the beginning and end of the summary. Here, We have chosen sostok and eostok as START and END tokens"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:56.823071Z",
     "iopub.status.busy": "2021-07-18T10:06:56.822521Z",
     "iopub.status.idle": "2021-07-18T10:06:56.863828Z",
     "shell.execute_reply": "2021-07-18T10:06:56.862545Z",
     "shell.execute_reply.started": "2021-07-18T10:06:56.823021Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>summary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>bought several vitality canned dog food products found good quality product looks like stew processed meat smells better labrador finicky appreciates product better</td>\n",
       "      <td>sostok good quality dog food eostok</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>product arrived labeled jumbo salted peanuts peanuts actually small sized unsalted sure error vendor intended represent product jumbo</td>\n",
       "      <td>sostok not as advertised eostok</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>looking secret ingredient robitussin believe found got addition root beer extract ordered made cherry soda flavor medicinal</td>\n",
       "      <td>sostok cough medicine eostok</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>great taffy great price wide assortment yummy taffy delivery quick taffy lover deal</td>\n",
       "      <td>sostok great taffy eostok</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>saltwater taffy great flavors soft chewy candy individually wrapped well none candies stuck together happen expensive version fralinger would highly recommend candy served beach themed party every...</td>\n",
       "      <td>sostok great just as good as the expensive brands eostok</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                                                                                                                                                                      text  \\\n",
       "0                                     bought several vitality canned dog food products found good quality product looks like stew processed meat smells better labrador finicky appreciates product better   \n",
       "1                                                                    product arrived labeled jumbo salted peanuts peanuts actually small sized unsalted sure error vendor intended represent product jumbo   \n",
       "2                                                                              looking secret ingredient robitussin believe found got addition root beer extract ordered made cherry soda flavor medicinal   \n",
       "3                                                                                                                      great taffy great price wide assortment yummy taffy delivery quick taffy lover deal   \n",
       "4  saltwater taffy great flavors soft chewy candy individually wrapped well none candies stuck together happen expensive version fralinger would highly recommend candy served beach themed party every...   \n",
       "\n",
       "                                                    summary  \n",
       "0                       sostok good quality dog food eostok  \n",
       "1                           sostok not as advertised eostok  \n",
       "2                              sostok cough medicine eostok  \n",
       "3                                 sostok great taffy eostok  \n",
       "4  sostok great just as good as the expensive brands eostok  "
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['summary'] = df['summary'].apply(lambda x : 'sostok '+ x + ' eostok')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Train-Test Split and Prepare the Tokenizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:56.865985Z",
     "iopub.status.busy": "2021-07-18T10:06:56.865559Z",
     "iopub.status.idle": "2021-07-18T10:06:56.878675Z",
     "shell.execute_reply": "2021-07-18T10:06:56.876986Z",
     "shell.execute_reply.started": "2021-07-18T10:06:56.865940Z"
    },
    "id": "RakakKHcFxhl"
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "x_tr,x_val,y_tr,y_val=train_test_split(np.array(df['text']), np.array(df['summary']),\n",
    "                                       test_size=0.1, random_state=0, shuffle=True) \n",
    "\n",
    "# A tokenizer builds the vocabulary and converts a word sequence to an integer sequence.\n",
    "# We will now build tokenizers for text and summary.\n",
    "\n",
    "x_tokenizer = Tokenizer() \n",
    "x_tokenizer.fit_on_texts(list(x_tr))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "RzvLwYL_PDcx"
   },
   "source": [
    "## Rarewords and its Coverage on Reviews column\n",
    "\n",
    "The threshold is taken as 4 which means word whose count is below 4 is considered as a **rare word**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:57.491921Z",
     "iopub.status.busy": "2021-07-18T10:06:57.491464Z",
     "iopub.status.idle": "2021-07-18T10:06:57.514348Z",
     "shell.execute_reply": "2021-07-18T10:06:57.512850Z",
     "shell.execute_reply.started": "2021-07-18T10:06:57.491873Z"
    },
    "id": "y8KronV2Fxhx",
    "outputId": "d2eb2f27-fbbc-4e61-9556-3c3ff5e4327b",
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "% of rare words in vocabulary: 66.12760790374213\n",
      "Total Coverage of rare words: 4.072762119889796\n"
     ]
    }
   ],
   "source": [
    "thresh=4\n",
    "\n",
    "cnt=0\n",
    "tot_cnt=0\n",
    "freq=0\n",
    "tot_freq=0\n",
    "\n",
    "for key,value in x_tokenizer.word_counts.items():\n",
    "    tot_cnt=tot_cnt+1\n",
    "    tot_freq=tot_freq+value\n",
    "    if(value<thresh):\n",
    "        cnt=cnt+1\n",
    "        freq=freq+value\n",
    "    \n",
    "print(\"% of rare words in vocabulary:\", (cnt/tot_cnt)*100)\n",
    "print(\"Total Coverage of rare words:\", (freq/tot_freq)*100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:57.517335Z",
     "iopub.status.busy": "2021-07-18T10:06:57.516818Z",
     "iopub.status.idle": "2021-07-18T10:06:57.531064Z",
     "shell.execute_reply": "2021-07-18T10:06:57.529673Z",
     "shell.execute_reply.started": "2021-07-18T10:06:57.517289Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "11981\n",
      "18118\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(None, None)"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(cnt),print(tot_cnt)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:57.533644Z",
     "iopub.status.busy": "2021-07-18T10:06:57.533153Z",
     "iopub.status.idle": "2021-07-18T10:06:57.546408Z",
     "shell.execute_reply": "2021-07-18T10:06:57.544904Z",
     "shell.execute_reply.started": "2021-07-18T10:06:57.533597Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "16660\n",
      "409059\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(None, None)"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(freq),print(tot_freq)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "So-J-5kzQIeO"
   },
   "source": [
    "NOTE:\n",
    "\n",
    "* **tot_cnt** gives the size of vocabulary (which means every unique words in the text)\n",
    " \n",
    "*   **cnt** gives me the no. of rare words whose count falls below threshold\n",
    "\n",
    "*  **tot_cnt - cnt** gives me the top most common words "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let us define the tokenizer with **top most common words** for reviews."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Reviews Tokenizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:57.549081Z",
     "iopub.status.busy": "2021-07-18T10:06:57.548600Z",
     "iopub.status.idle": "2021-07-18T10:06:58.873716Z",
     "shell.execute_reply": "2021-07-18T10:06:58.872576Z",
     "shell.execute_reply.started": "2021-07-18T10:06:57.549033Z"
    },
    "id": "J2giEsF3Fxh3"
   },
   "outputs": [],
   "source": [
    "# prepare a tokenizer for reviews on training data\n",
    "\n",
    "x_tokenizer = Tokenizer(num_words=tot_cnt-cnt) # num_words: the maximum number of words to keep, based on word frequency.\n",
    "x_tokenizer.fit_on_texts(list(x_tr))\n",
    "\n",
    "#convert text sequences into integer sequences\n",
    "x_tr_seq    =   x_tokenizer.texts_to_sequences(x_tr) \n",
    "x_val_seq   =   x_tokenizer.texts_to_sequences(x_val)\n",
    "\n",
    "#padding zero upto maximum length\n",
    "x_tr    =   pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')\n",
    "x_val   =   pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')\n",
    "\n",
    "#size of vocabulary ( +1 for padding token)\n",
    "x_voc   =  x_tokenizer.num_words + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:58.875876Z",
     "iopub.status.busy": "2021-07-18T10:06:58.875395Z",
     "iopub.status.idle": "2021-07-18T10:06:58.884195Z",
     "shell.execute_reply": "2021-07-18T10:06:58.882676Z",
     "shell.execute_reply.started": "2021-07-18T10:06:58.875828Z"
    },
    "id": "DCbGMsm4FxiA",
    "outputId": "2d9165f0-e542-4114-91f3-e070d483fce9"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6138"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_voc"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "uQfKP3sqRxi9"
   },
   "source": [
    "## Summary Tokenizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:58.886833Z",
     "iopub.status.busy": "2021-07-18T10:06:58.886317Z",
     "iopub.status.idle": "2021-07-18T10:06:59.186345Z",
     "shell.execute_reply": "2021-07-18T10:06:59.185182Z",
     "shell.execute_reply.started": "2021-07-18T10:06:58.886781Z"
    },
    "id": "eRHqyBkBFxiJ"
   },
   "outputs": [],
   "source": [
    "#prepare a tokenizer for reviews on training data\n",
    "\n",
    "y_tokenizer = Tokenizer()   \n",
    "y_tokenizer.fit_on_texts(list(y_tr))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "KInA6O6ZSkJz"
   },
   "source": [
    "## Rarewords and its Coverage on the summary column\n",
    "\n",
    "The threshold is taken as 6 which means word whose count is below 6 is considered as a **rare word**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:59.188565Z",
     "iopub.status.busy": "2021-07-18T10:06:59.188065Z",
     "iopub.status.idle": "2021-07-18T10:06:59.202000Z",
     "shell.execute_reply": "2021-07-18T10:06:59.200517Z",
     "shell.execute_reply.started": "2021-07-18T10:06:59.188509Z"
    },
    "id": "yzE5OiRLFxiM",
    "outputId": "7f7a4f89-b088-4847-8172-09e5a2383d0e"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "% of rare words in vocabulary: 79.91605782683041\n",
      "Total Coverage of rare words: 7.357814181716485\n"
     ]
    }
   ],
   "source": [
    "thresh=6\n",
    "\n",
    "cnt=0\n",
    "tot_cnt=0\n",
    "freq=0\n",
    "tot_freq=0\n",
    "\n",
    "for key,value in y_tokenizer.word_counts.items():\n",
    "    tot_cnt=tot_cnt+1\n",
    "    tot_freq=tot_freq+value\n",
    "    if(value<thresh):\n",
    "        cnt=cnt+1\n",
    "        freq=freq+value\n",
    "    \n",
    "print(\"% of rare words in vocabulary:\",(cnt/tot_cnt)*100)\n",
    "print(\"Total Coverage of rare words:\",(freq/tot_freq)*100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:59.204715Z",
     "iopub.status.busy": "2021-07-18T10:06:59.203774Z",
     "iopub.status.idle": "2021-07-18T10:06:59.218072Z",
     "shell.execute_reply": "2021-07-18T10:06:59.216242Z",
     "shell.execute_reply.started": "2021-07-18T10:06:59.204667Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5141\n",
      "6433\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(None, None)"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(cnt),print(tot_cnt)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:59.220894Z",
     "iopub.status.busy": "2021-07-18T10:06:59.220270Z",
     "iopub.status.idle": "2021-07-18T10:06:59.230525Z",
     "shell.execute_reply": "2021-07-18T10:06:59.228867Z",
     "shell.execute_reply.started": "2021-07-18T10:06:59.220846Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "8682\n",
      "117997\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(None, None)"
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(freq),print(tot_freq)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0PBhzKuRSw_9"
   },
   "source": [
    "Let us define the tokenizer with **top most common words for summary**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:06:59.233596Z",
     "iopub.status.busy": "2021-07-18T10:06:59.232917Z",
     "iopub.status.idle": "2021-07-18T10:07:00.003463Z",
     "shell.execute_reply": "2021-07-18T10:07:00.002152Z",
     "shell.execute_reply.started": "2021-07-18T10:06:59.233542Z"
    },
    "id": "-fswLvIgFxiR"
   },
   "outputs": [],
   "source": [
    "#prepare a tokenizer for reviews on training data\n",
    "y_tokenizer = Tokenizer(num_words=tot_cnt-cnt) \n",
    "y_tokenizer.fit_on_texts(list(y_tr))\n",
    "\n",
    "#convert text sequences into integer sequences\n",
    "y_tr_seq    =   y_tokenizer.texts_to_sequences(y_tr) \n",
    "y_val_seq   =   y_tokenizer.texts_to_sequences(y_val) \n",
    "\n",
    "#padding zero upto maximum length\n",
    "y_tr    =   pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')\n",
    "y_val   =   pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')\n",
    "\n",
    "#size of vocabulary\n",
    "y_voc  =   y_tokenizer.num_words +1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:07:00.005758Z",
     "iopub.status.busy": "2021-07-18T10:07:00.005248Z",
     "iopub.status.idle": "2021-07-18T10:07:00.010894Z",
     "shell.execute_reply": "2021-07-18T10:07:00.009558Z",
     "shell.execute_reply.started": "2021-07-18T10:07:00.005708Z"
    }
   },
   "outputs": [],
   "source": [
    "#deleting the rows that contain only START and END tokens"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:07:00.013615Z",
     "iopub.status.busy": "2021-07-18T10:07:00.012739Z",
     "iopub.status.idle": "2021-07-18T10:07:00.879576Z",
     "shell.execute_reply": "2021-07-18T10:07:00.878385Z",
     "shell.execute_reply.started": "2021-07-18T10:07:00.013566Z"
    }
   },
   "outputs": [],
   "source": [
    "ind=[]\n",
    "for i in range(len(y_tr)):\n",
    "    cnt=0\n",
    "    for j in y_tr[i]:\n",
    "        if j!=0:\n",
    "            cnt=cnt+1\n",
    "    if(cnt==2):\n",
    "        ind.append(i)\n",
    "\n",
    "y_tr=np.delete(y_tr,ind, axis=0)\n",
    "x_tr=np.delete(x_tr,ind, axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:07:00.882443Z",
     "iopub.status.busy": "2021-07-18T10:07:00.881592Z",
     "iopub.status.idle": "2021-07-18T10:07:00.989297Z",
     "shell.execute_reply": "2021-07-18T10:07:00.988292Z",
     "shell.execute_reply.started": "2021-07-18T10:07:00.882393Z"
    }
   },
   "outputs": [],
   "source": [
    "ind=[]\n",
    "for i in range(len(y_val)):\n",
    "    cnt=0\n",
    "    for j in y_val[i]:\n",
    "        if j!=0:\n",
    "            cnt=cnt+1\n",
    "    if(cnt==2):\n",
    "        ind.append(i)\n",
    "\n",
    "y_val=np.delete(y_val,ind, axis=0)\n",
    "x_val=np.delete(x_val,ind, axis=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "wOtlDcthFxip"
   },
   "source": [
    "# Abstractive Text Summarization - Model building\n",
    "\n",
    "We are finally at the model building part. But before we do that, we need to familiarize ourselves with a few terms which are required prior to building the model.\n",
    "\n",
    "**Return Sequences = True**: When the return sequences parameter is set to True, LSTM produces the hidden state and cell state for every timestep\n",
    "\n",
    "**Return State = True**: When return state = True, LSTM produces the hidden state and cell state of the last timestep only\n",
    "\n",
    "**Initial State**: This is used to initialize the internal states of the LSTM for the first timestep\n",
    "\n",
    "**Stacked LSTM**: Stacked LSTM has multiple layers of LSTM stacked on top of each other. \n",
    "This leads to a better representation of the sequence.\n",
    "\n",
    "Here, we are building a 3 stacked LSTM for the encoder:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:07:00.991555Z",
     "iopub.status.busy": "2021-07-18T10:07:00.991047Z",
     "iopub.status.idle": "2021-07-18T10:07:01.877305Z",
     "shell.execute_reply": "2021-07-18T10:07:01.876235Z",
     "shell.execute_reply.started": "2021-07-18T10:07:00.991508Z"
    },
    "id": "zXef38nBFxir",
    "outputId": "7ae99521-46f8-4c6f-9cba-4979deffeee8"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"model\"\n",
      "__________________________________________________________________________________________________\n",
      "Layer (type)                    Output Shape         Param #     Connected to                     \n",
      "==================================================================================================\n",
      "input_1 (InputLayer)            [(None, 30)]         0                                            \n",
      "__________________________________________________________________________________________________\n",
      "embedding (Embedding)           (None, 30, 100)      613800      input_1[0][0]                    \n",
      "__________________________________________________________________________________________________\n",
      "lstm (LSTM)                     [(None, 30, 300), (N 481200      embedding[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "input_2 (InputLayer)            [(None, None)]       0                                            \n",
      "__________________________________________________________________________________________________\n",
      "lstm_1 (LSTM)                   [(None, 30, 300), (N 721200      lstm[0][0]                       \n",
      "__________________________________________________________________________________________________\n",
      "embedding_1 (Embedding)         (None, None, 100)    129300      input_2[0][0]                    \n",
      "__________________________________________________________________________________________________\n",
      "lstm_2 (LSTM)                   [(None, 30, 300), (N 721200      lstm_1[0][0]                     \n",
      "__________________________________________________________________________________________________\n",
      "lstm_3 (LSTM)                   [(None, None, 300),  481200      embedding_1[0][0]                \n",
      "                                                                 lstm_2[0][1]                     \n",
      "                                                                 lstm_2[0][2]                     \n",
      "__________________________________________________________________________________________________\n",
      "attention_layer (AttentionLayer ((None, None, 300),  180300      lstm_2[0][0]                     \n",
      "                                                                 lstm_3[0][0]                     \n",
      "__________________________________________________________________________________________________\n",
      "concat_layer (Concatenate)      (None, None, 600)    0           lstm_3[0][0]                     \n",
      "                                                                 attention_layer[0][0]            \n",
      "__________________________________________________________________________________________________\n",
      "time_distributed (TimeDistribut (None, None, 1293)   777093      concat_layer[0][0]               \n",
      "==================================================================================================\n",
      "Total params: 4,105,293\n",
      "Trainable params: 4,105,293\n",
      "Non-trainable params: 0\n",
      "__________________________________________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "latent_dim = 300\n",
    "embedding_dim=100\n",
    "\n",
    "# Encoder\n",
    "encoder_inputs = Input(shape=(max_text_len,))\n",
    "\n",
    "#embedding layer\n",
    "enc_emb =  Embedding(x_voc, embedding_dim,trainable=True)(encoder_inputs)\n",
    "\n",
    "#encoder lstm 1\n",
    "encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)\n",
    "encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)\n",
    "\n",
    "#encoder lstm 2\n",
    "encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)\n",
    "encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)\n",
    "\n",
    "#encoder lstm 3\n",
    "encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)\n",
    "encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)\n",
    "\n",
    "# Set up the decoder, using `encoder_states` as initial state.\n",
    "decoder_inputs = Input(shape=(None,))\n",
    "\n",
    "#embedding layer\n",
    "dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True)\n",
    "dec_emb = dec_emb_layer(decoder_inputs)\n",
    "\n",
    "decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)\n",
    "decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])\n",
    "\n",
    "# Attention layer\n",
    "attn_layer = AttentionLayer(name='attention_layer')\n",
    "attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])\n",
    "\n",
    "# Concat attention input and decoder LSTM output\n",
    "decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])\n",
    "\n",
    "#dense layer\n",
    "decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))\n",
    "decoder_outputs = decoder_dense(decoder_concat_input)\n",
    "\n",
    "# Define the model \n",
    "model = Model([encoder_inputs, decoder_inputs], decoder_outputs)\n",
    "\n",
    "model.summary() "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0ZVlfRuMUcoP"
   },
   "source": [
    "Sparse categorical cross-entropy as the loss function since it converts the integer sequence to a one-hot vector on the fly. This overcomes any memory issues."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:07:01.879631Z",
     "iopub.status.busy": "2021-07-18T10:07:01.879104Z",
     "iopub.status.idle": "2021-07-18T10:07:01.896268Z",
     "shell.execute_reply": "2021-07-18T10:07:01.894848Z",
     "shell.execute_reply.started": "2021-07-18T10:07:01.879573Z"
    },
    "id": "Lwfi1Fm8Fxiz"
   },
   "outputs": [],
   "source": [
    "model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "p0ykDbxfUhyw"
   },
   "source": [
    "EarlyStopping monitors the validation loss (val_loss). Our model will stop training once the validation loss increases."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:07:01.899239Z",
     "iopub.status.busy": "2021-07-18T10:07:01.898305Z",
     "iopub.status.idle": "2021-07-18T10:07:01.908928Z",
     "shell.execute_reply": "2021-07-18T10:07:01.907619Z",
     "shell.execute_reply.started": "2021-07-18T10:07:01.899193Z"
    },
    "id": "s-A3J92MUljB"
   },
   "outputs": [],
   "source": [
    "es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Mw6CVECaUq5b"
   },
   "source": [
    "Fit the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:07:01.911287Z",
     "iopub.status.busy": "2021-07-18T10:07:01.910715Z",
     "iopub.status.idle": "2021-07-18T10:37:14.840340Z",
     "shell.execute_reply": "2021-07-18T10:37:14.839385Z",
     "shell.execute_reply.started": "2021-07-18T10:07:01.911229Z"
    },
    "id": "ETnPzA4OFxi3",
    "outputId": "477e374f-7cf2-4d60-f86e-2c49c9cebedb"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "171/171 [==============================] - 123s 639ms/step - loss: 3.2539 - val_loss: 2.6145\n",
      "Epoch 2/50\n",
      "171/171 [==============================] - 108s 629ms/step - loss: 2.5039 - val_loss: 2.4287\n",
      "Epoch 3/50\n",
      "171/171 [==============================] - 106s 621ms/step - loss: 2.3863 - val_loss: 2.3186\n",
      "Epoch 4/50\n",
      "171/171 [==============================] - 106s 618ms/step - loss: 2.2780 - val_loss: 2.2466\n",
      "Epoch 5/50\n",
      "171/171 [==============================] - 107s 625ms/step - loss: 2.1880 - val_loss: 2.1892\n",
      "Epoch 6/50\n",
      "171/171 [==============================] - 106s 619ms/step - loss: 2.1115 - val_loss: 2.1545\n",
      "Epoch 7/50\n",
      "171/171 [==============================] - 104s 609ms/step - loss: 2.0521 - val_loss: 2.1249\n",
      "Epoch 8/50\n",
      "171/171 [==============================] - 104s 611ms/step - loss: 1.9751 - val_loss: 2.0881\n",
      "Epoch 9/50\n",
      "171/171 [==============================] - 106s 620ms/step - loss: 1.9317 - val_loss: 2.0640\n",
      "Epoch 10/50\n",
      "171/171 [==============================] - 105s 611ms/step - loss: 1.8853 - val_loss: 2.0530\n",
      "Epoch 11/50\n",
      "171/171 [==============================] - 105s 617ms/step - loss: 1.8520 - val_loss: 2.0422\n",
      "Epoch 12/50\n",
      "171/171 [==============================] - 105s 612ms/step - loss: 1.8036 - val_loss: 2.0303\n",
      "Epoch 13/50\n",
      "171/171 [==============================] - 104s 609ms/step - loss: 1.7811 - val_loss: 2.0211\n",
      "Epoch 14/50\n",
      "171/171 [==============================] - 106s 620ms/step - loss: 1.7414 - val_loss: 2.0317\n",
      "Epoch 15/50\n",
      "171/171 [==============================] - 106s 621ms/step - loss: 1.6862 - val_loss: 2.0165\n",
      "Epoch 16/50\n",
      "171/171 [==============================] - 107s 625ms/step - loss: 1.6803 - val_loss: 2.0199\n",
      "Epoch 17/50\n",
      "171/171 [==============================] - 105s 614ms/step - loss: 1.6303 - val_loss: 2.0194\n",
      "Epoch 00017: early stopping\n"
     ]
    }
   ],
   "source": [
    "history=model.fit([x_tr, y_tr[:,:-1]], \n",
    "                  y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:,1:],\n",
    "                  epochs=50,\n",
    "                  callbacks=[es],\n",
    "                  batch_size=128, \n",
    "                  validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:])\n",
    "                 )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0ezKYOp2UxG5"
   },
   "source": [
    "## Understanding the Diagnostic plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:37:14.842346Z",
     "iopub.status.busy": "2021-07-18T10:37:14.842043Z",
     "iopub.status.idle": "2021-07-18T10:37:15.049505Z",
     "shell.execute_reply": "2021-07-18T10:37:15.048533Z",
     "shell.execute_reply.started": "2021-07-18T10:37:14.842300Z"
    },
    "id": "tDTNLAURFxjE",
    "outputId": "e2ea6e44-3931-4014-97a1-03fa2a441228"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from matplotlib import pyplot\n",
    "pyplot.plot(history.history['loss'], label='train')\n",
    "pyplot.plot(history.history['val_loss'], label='test')\n",
    "pyplot.legend()\n",
    "pyplot.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "HSyx-HvpUz2o"
   },
   "source": [
    "Next, let’s build the dictionary to convert the index to word for target and source vocabulary:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:37:15.051913Z",
     "iopub.status.busy": "2021-07-18T10:37:15.051622Z",
     "iopub.status.idle": "2021-07-18T10:37:15.059555Z",
     "shell.execute_reply": "2021-07-18T10:37:15.058510Z",
     "shell.execute_reply.started": "2021-07-18T10:37:15.051885Z"
    },
    "id": "sBX0zZnOFxjW"
   },
   "outputs": [],
   "source": [
    "reverse_target_word_index=y_tokenizer.index_word\n",
    "reverse_source_word_index=x_tokenizer.index_word\n",
    "target_word_index=y_tokenizer.word_index"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "eM_nU_VvFxjq"
   },
   "source": [
    "## Inference\n",
    "\n",
    "Set up the inference for the encoder and decoder:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:37:15.062156Z",
     "iopub.status.busy": "2021-07-18T10:37:15.061848Z",
     "iopub.status.idle": "2021-07-18T10:37:15.454635Z",
     "shell.execute_reply": "2021-07-18T10:37:15.450954Z",
     "shell.execute_reply.started": "2021-07-18T10:37:15.062129Z"
    },
    "id": "9QkrNV-4Fxjt"
   },
   "outputs": [],
   "source": [
    "# Encode the input sequence to get the feature vector\n",
    "encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])\n",
    "\n",
    "# Decoder setup\n",
    "# Below tensors will hold the states of the previous time step\n",
    "decoder_state_input_h = Input(shape=(latent_dim,))\n",
    "decoder_state_input_c = Input(shape=(latent_dim,))\n",
    "decoder_hidden_state_input = Input(shape=(max_text_len,latent_dim))\n",
    "\n",
    "# Get the embeddings of the decoder sequence\n",
    "dec_emb2= dec_emb_layer(decoder_inputs) \n",
    "# To predict the next word in the sequence, set the initial states to the states from the previous time step\n",
    "decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])\n",
    "\n",
    "#attention inference\n",
    "attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])\n",
    "decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])\n",
    "\n",
    "# A dense softmax layer to generate prob dist. over the target vocabulary\n",
    "decoder_outputs2 = decoder_dense(decoder_inf_concat) \n",
    "\n",
    "# Final decoder model\n",
    "decoder_model = Model(\n",
    "    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],\n",
    "    [decoder_outputs2] + [state_h2, state_c2])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "zOiyk4ToWe74"
   },
   "source": [
    "We are defining a function below which is the implementation of the inference process"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:37:15.457380Z",
     "iopub.status.busy": "2021-07-18T10:37:15.456927Z",
     "iopub.status.idle": "2021-07-18T10:37:15.482155Z",
     "shell.execute_reply": "2021-07-18T10:37:15.480609Z",
     "shell.execute_reply.started": "2021-07-18T10:37:15.457335Z"
    },
    "id": "6f6TTFnBFxj6"
   },
   "outputs": [],
   "source": [
    "def decode_sequence(input_seq):\n",
    "    # Encode the input as state vectors.\n",
    "    e_out, e_h, e_c = encoder_model.predict(input_seq)\n",
    "    \n",
    "    # Generate empty target sequence of length 1.\n",
    "    target_seq = np.zeros((1,1))\n",
    "    \n",
    "    # Populate the first word of target sequence with the start word.\n",
    "    target_seq[0, 0] = target_word_index['sostok']\n",
    "\n",
    "    stop_condition = False\n",
    "    decoded_sentence = ''\n",
    "    while not stop_condition:\n",
    "      \n",
    "        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])\n",
    "\n",
    "        # Sample a token\n",
    "        sampled_token_index = np.argmax(output_tokens[0, -1, :])\n",
    "        sampled_token = reverse_target_word_index[sampled_token_index]\n",
    "        \n",
    "        if(sampled_token!='eostok'):\n",
    "            decoded_sentence += ' '+sampled_token\n",
    "\n",
    "        # Exit condition: either hit max length or find stop word.\n",
    "        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):\n",
    "            stop_condition = True\n",
    "\n",
    "        # Update the target sequence (of length 1).\n",
    "        target_seq = np.zeros((1,1))\n",
    "        target_seq[0, 0] = sampled_token_index\n",
    "\n",
    "        # Update internal states\n",
    "        e_h, e_c = h, c\n",
    "\n",
    "    return decoded_sentence"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "6GuDf4TPWt6_"
   },
   "source": [
    "Let us define the functions to convert an integer sequence to a word sequence for summary as well as the reviews:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:37:15.485941Z",
     "iopub.status.busy": "2021-07-18T10:37:15.484777Z",
     "iopub.status.idle": "2021-07-18T10:37:15.502204Z",
     "shell.execute_reply": "2021-07-18T10:37:15.500635Z",
     "shell.execute_reply.started": "2021-07-18T10:37:15.485891Z"
    },
    "id": "aAUntznIFxj9"
   },
   "outputs": [],
   "source": [
    "def seq2summary(input_seq):\n",
    "    newString=''\n",
    "    for i in input_seq:\n",
    "        if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):\n",
    "            newString=newString+reverse_target_word_index[i]+' '\n",
    "    return newString\n",
    "\n",
    "def seq2text(input_seq):\n",
    "    newString=''\n",
    "    for i in input_seq:\n",
    "        if(i!=0):\n",
    "            newString=newString+reverse_source_word_index[i]+' '\n",
    "    return newString"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "9gM4ALyfWwA9"
   },
   "source": [
    "Here are a few summaries generated by the model:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:37:15.506010Z",
     "iopub.status.busy": "2021-07-18T10:37:15.505094Z",
     "iopub.status.idle": "2021-07-18T10:37:19.969770Z",
     "shell.execute_reply": "2021-07-18T10:37:19.968449Z",
     "shell.execute_reply.started": "2021-07-18T10:37:15.505964Z"
    },
    "id": "BUtQmQTmFxkI",
    "outputId": "f407d9fc-e0cd-4082-98f5-bd1f562dc26f"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Review: gluten free want crackers one also delicious second order \n",
      "Original summary: cracker of cracker \n",
      "Predicted summary:  great gluten free baking\n",
      "\n",
      "\n",
      "Review: coffee tastes like regular coffee good weak used little less water regular strength coffee taste great smell better whether works stomach needed sure \n",
      "Original summary: the taste is good \n",
      "Predicted summary:  good coffee\n",
      "\n",
      "\n",
      "Review: first time italy wife actually give gifts well good sucks seasonal definitely makes look forward holidays \n",
      "Original summary: better than crack \n",
      "Predicted summary:  the best\n",
      "\n",
      "\n",
      "Review: tea tasty love night little soy milk splenda would recommend people trying healthy green tea \n",
      "Original summary: yummy \n",
      "Predicted summary:  great tea\n",
      "\n",
      "\n",
      "Review: bought cookies gifts open last long good make great gifts would definitely buy \n",
      "Original summary: mouth watery cookies \n",
      "Predicted summary:  best cookies ever\n",
      "\n",
      "\n",
      "Review: brewed pot found strong stronger starbucks even even extra milk still strong taste acidic aftertaste think many enjoy taste \n",
      "Original summary: too strong \n",
      "Predicted summary:  senseo paris french vanilla coffee pods\n",
      "\n",
      "\n",
      "Review: drinking years simple morning lunch tea fancy reliable always satisfying hard beat price \n",
      "Original summary: very nice simple tea \n",
      "Predicted summary:  great tea\n",
      "\n",
      "\n",
      "Review: find balsamic vinegar great cooking salad dressings drizzled fruits cheeses veggies good flavor compliments many dishes \n",
      "Original summary: leaf vinegar for good price \n",
      "Predicted summary:  great product\n",
      "\n",
      "\n",
      "Review: made crab used sauce dipping sauce great love fact msg right amount sweetness little spicy right amount \n",
      "Original summary: really good sauce \n",
      "Predicted summary:  the best\n",
      "\n",
      "\n",
      "Review: coffee sample selection ordered keurig brewer far favorite ordered coffee several times amazon dark roast strong \n",
      "Original summary: wolfgang puck coffee sumatra for keurig \n",
      "Predicted summary:  great coffee\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "for i in range(0, 10):\n",
    "    print(\"Review:\",seq2text(x_tr[i]))\n",
    "    print(\"Original summary:\",seq2summary(y_tr[i]))\n",
    "    print(\"Predicted summary:\",decode_sequence(x_tr[i].reshape(1,max_text_len)))\n",
    "    print(\"\\n\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "OTkaYNjHW4lC"
   },
   "source": [
    "Finally, Our model is able to generate a meaningful summary based on the context present in the text."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Text Summarization using Transformers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T10:37:28.199624Z",
     "iopub.status.busy": "2021-07-18T10:37:28.199290Z",
     "iopub.status.idle": "2021-07-18T10:37:28.205986Z",
     "shell.execute_reply": "2021-07-18T10:37:28.204830Z",
     "shell.execute_reply.started": "2021-07-18T10:37:28.199592Z"
    }
   },
   "outputs": [],
   "source": [
    "from transformers import pipeline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Summarize news articles and other documents.\n",
    "This summarizing pipeline can currently be loaded from pipeline() using the following task identifier: \"summarization\".\n",
    "The models that this pipeline can use are models that have been fine-tuned on a summarization task, which is currently, ‘bart-large-cnn’, ‘t5-small’, ‘t5-base’, ‘t5-large’, ‘t5-3b’, ‘t5-11b’.\n",
    "\n",
    "https://huggingface.co/t5-base\n",
    "\n",
    "https://huggingface.co/models?filter=summarization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T16:15:30.034938Z",
     "iopub.status.busy": "2021-07-18T16:15:30.034498Z",
     "iopub.status.idle": "2021-07-18T16:15:30.268895Z",
     "shell.execute_reply": "2021-07-18T16:15:30.267944Z",
     "shell.execute_reply.started": "2021-07-18T16:15:30.034851Z"
    }
   },
   "outputs": [],
   "source": [
    "import bs4 as bs\n",
    "import urllib.request\n",
    "import re"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T16:15:37.634640Z",
     "iopub.status.busy": "2021-07-18T16:15:37.634248Z",
     "iopub.status.idle": "2021-07-18T16:15:38.807490Z",
     "shell.execute_reply": "2021-07-18T16:15:38.806628Z",
     "shell.execute_reply.started": "2021-07-18T16:15:37.634580Z"
    }
   },
   "outputs": [],
   "source": [
    "scraped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Artificial_intelligence')\n",
    "article = scraped_data.read()\n",
    "parsed_article = bs.BeautifulSoup(article,'lxml')\n",
    "paragraphs = parsed_article.find_all('p')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T16:15:43.123473Z",
     "iopub.status.busy": "2021-07-18T16:15:43.123111Z",
     "iopub.status.idle": "2021-07-18T16:15:43.130507Z",
     "shell.execute_reply": "2021-07-18T16:15:43.129565Z",
     "shell.execute_reply.started": "2021-07-18T16:15:43.123438Z"
    }
   },
   "outputs": [],
   "source": [
    "article_text = \"\"\n",
    "for p in paragraphs:\n",
    "    article_text += p.text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T16:16:29.081781Z",
     "iopub.status.busy": "2021-07-18T16:16:29.081424Z",
     "iopub.status.idle": "2021-07-18T16:16:29.089070Z",
     "shell.execute_reply": "2021-07-18T16:16:29.088151Z",
     "shell.execute_reply.started": "2021-07-18T16:16:29.081749Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'\\n\\nArtificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans or animals. \\nLeading AI textbooks define the field as the study of \"intelligent agents\": any system that perceives its environment and takes actions that maximize its chance of achieving its goals.[a] \\nSome popular accounts use the term \"artificial intelligence\" to describe machines that mimic \"cognitive\" functions that humans associate with the human mind, such as '"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "article_text[0:500]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T16:16:33.342864Z",
     "iopub.status.busy": "2021-07-18T16:16:33.342496Z",
     "iopub.status.idle": "2021-07-18T16:16:33.351787Z",
     "shell.execute_reply": "2021-07-18T16:16:33.350510Z",
     "shell.execute_reply.started": "2021-07-18T16:16:33.342822Z"
    }
   },
   "outputs": [],
   "source": [
    "# Text preprocessing\n",
    "\n",
    "# Removing Square Brackets and Extra Spaces\n",
    "article_text = re.sub(r'\\[[0-9]*\\]', ' ', article_text)\n",
    "article_text = re.sub(r'\\s+', ' ', article_text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T16:16:39.993030Z",
     "iopub.status.busy": "2021-07-18T16:16:39.992717Z",
     "iopub.status.idle": "2021-07-18T16:16:40.004144Z",
     "shell.execute_reply": "2021-07-18T16:16:40.003354Z",
     "shell.execute_reply.started": "2021-07-18T16:16:39.993002Z"
    }
   },
   "outputs": [],
   "source": [
    "# Removing special characters and digits\n",
    "formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )\n",
    "formatted_article_text = re.sub(r'\\s+', ' ', formatted_article_text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T16:16:47.019757Z",
     "iopub.status.busy": "2021-07-18T16:16:47.019310Z",
     "iopub.status.idle": "2021-07-18T16:16:47.025327Z",
     "shell.execute_reply": "2021-07-18T16:16:47.024339Z",
     "shell.execute_reply.started": "2021-07-18T16:16:47.019711Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "44288"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(formatted_article_text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T16:16:54.298439Z",
     "iopub.status.busy": "2021-07-18T16:16:54.298128Z",
     "iopub.status.idle": "2021-07-18T16:16:54.302336Z",
     "shell.execute_reply": "2021-07-18T16:16:54.301337Z",
     "shell.execute_reply.started": "2021-07-18T16:16:54.298411Z"
    }
   },
   "outputs": [],
   "source": [
    "formatted_article_text_1=formatted_article_text[0:500]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": true,
    "execution": {
     "iopub.execute_input": "2021-07-18T16:19:29.552693Z",
     "iopub.status.busy": "2021-07-18T16:19:29.552320Z",
     "iopub.status.idle": "2021-07-18T16:20:06.155179Z",
     "shell.execute_reply": "2021-07-18T16:20:06.154236Z",
     "shell.execute_reply.started": "2021-07-18T16:19:29.552653Z"
    }
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "09895feb826245db8a3bf0558c4cbf7e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/1.20k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b47ff35c2fe44b8ca8bedf86f2693869",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/892M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "All model checkpoint layers were used when initializing TFT5ForConditionalGeneration.\n",
      "\n",
      "All the layers of TFT5ForConditionalGeneration were initialized from the model checkpoint at t5-base.\n",
      "If your task is similar to the task the model of the checkpoint was trained on, you can already use TFT5ForConditionalGeneration for predictions without further training.\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6f8b8ae81f4a44749c945bf5ad9a2018",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/792k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c0aecce9512842a68990988178750ed2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/1.39M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#use t5 in tf\n",
    "summarizer1 = pipeline(\"summarization\", model=\"t5-base\", tokenizer=\"t5-base\", framework=\"tf\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T16:21:02.002098Z",
     "iopub.status.busy": "2021-07-18T16:21:02.001497Z",
     "iopub.status.idle": "2021-07-18T16:21:02.008225Z",
     "shell.execute_reply": "2021-07-18T16:21:02.007446Z",
     "shell.execute_reply.started": "2021-07-18T16:21:02.002057Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "' Artificial intelligence AI is intelligence demonstrated by machines as opposed to the natural intelligence displayed by humans or animals Leading AI textbooks define the field as the study of intelligent agents any system that perceives its environment and takes actions that maximize its chance of achieving its goals a Some popular accounts use the term artificial intelligence to describe machines that mimic cognitive functions that humans associate with the human mind such as learning and prob'"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "formatted_article_text_1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T16:20:25.371110Z",
     "iopub.status.busy": "2021-07-18T16:20:25.370740Z",
     "iopub.status.idle": "2021-07-18T16:20:47.799100Z",
     "shell.execute_reply": "2021-07-18T16:20:47.798307Z",
     "shell.execute_reply.started": "2021-07-18T16:20:25.371079Z"
    },
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Your max_length is set to 500, but you input_length is only 83. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=50)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[{'summary_text': 'artificial intelligence (AI) is intelligence demonstrated by machines as opposed to the natural intelligence displayed by humans or animals . leading AI textbooks define the field as the study of intelligent agents .'}]"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "summarizer1(formatted_article_text_1, min_length=5, max_length=500,do_sample=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T16:20:51.567139Z",
     "iopub.status.busy": "2021-07-18T16:20:51.566812Z",
     "iopub.status.idle": "2021-07-18T16:21:01.999878Z",
     "shell.execute_reply": "2021-07-18T16:21:01.998786Z",
     "shell.execute_reply.started": "2021-07-18T16:20:51.567110Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[{'summary_text': 'AI is intelligence demonstrated by machines as opposed to the natural intelligence displayed by humans or animals .'}]"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "summarizer1(formatted_article_text_1, min_length=5, max_length=20,do_sample=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T16:24:20.965777Z",
     "iopub.status.busy": "2021-07-18T16:24:20.965408Z",
     "iopub.status.idle": "2021-07-18T16:24:20.974953Z",
     "shell.execute_reply": "2021-07-18T16:24:20.973846Z",
     "shell.execute_reply.started": "2021-07-18T16:24:20.965742Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most.'"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "text_string=data.Text[0]\n",
    "text_string"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-07-18T16:25:12.902082Z",
     "iopub.status.busy": "2021-07-18T16:25:12.901756Z",
     "iopub.status.idle": "2021-07-18T16:25:23.456627Z",
     "shell.execute_reply": "2021-07-18T16:25:23.455823Z",
     "shell.execute_reply.started": "2021-07-18T16:25:12.902055Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[{'summary_text': 'my Labrador is finicky and she appreciates this product better than most .'}]"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "summarizer1(text_string, min_length=5, max_length=20,do_sample=False)"
   ]
  }
 ],
 "metadata": {
  "hide_input": false,
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
