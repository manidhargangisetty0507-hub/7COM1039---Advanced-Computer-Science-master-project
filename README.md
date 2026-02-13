# 7COM1039---Advanced-Computer-Science-master-project
The project aims to classify financial documents into predefined categories using Natural Language Processing (NLP) techniques, specifically comparing the performance of a Convolutional Neural Network (CNN) and a BERT-based model.
## Data Generation: A synthetic dataset of 10,000 samples is generated. The dataset consists of short text snippets and their corresponding financial document categories. There are 8 predefined categories: "Balance Sheet", "Income Statement (Profit and Loss Statement)", "Cash Flow Statement", "Statement of Shareholders' Equity", "Expense Reports", "Invoices and Bills", "Contracts", and "Other Financial Documents". Each category has a specific keyword phrase associated with it, which is used to generate the synthetic text samples. This ensures a perfectly balanced and clean dataset for training, which explains the high accuracy seen later.

## Preprocessing Techniques 

Label Encoding: sklearn.preprocessing.LabelEncoder is used to convert categorical string labels (e.g., 'Balance Sheet') into numerical integer labels (label_encoded). This is a necessary step for machine learning models that require numerical inputs for class labels.

Text Cleaning: A custom preprocess function is defined to clean the text data. This function performs several standard NLP preprocessing steps:

Regular Expression: re.sub('[^a-zA-Z]', ' ', text) removes all characters that are not alphabetic, replacing them with spaces. This effectively cleans punctuation and numbers.

Lowercasing and Splitting: The text is converted to lowercase (.lower()) and then split into individual words (.split()).

Stop Word Removal: Common English stop words (e.g., 'the', 'is', 'a') are removed using nltk.corpus.stopwords. This helps reduce noise and focus on more meaningful words.

Stemming: PorterStemmer (from nltk.stem.porter) is applied to reduce words to their root form (e.g., 'running' to 'run'). This helps in reducing the vocabulary size and treating morphologically similar words as the same.

Tokenizer (for CNN) (Cell 70Qh2rSvMbFz): tf.keras.preprocessing.text.Tokenizer is initialized with num_words=5000 to convert the preprocessed text into sequences of integers, limiting the vocabulary to the 5000 most frequent words. fit_on_texts learns the vocabulary from the training data (X_train), and texts_to_sequences converts the texts to sequences.

Padding (for CNN) (Cell 70Qh2rSvMbFz): tf.keras.preprocessing.sequence.pad_sequences is used to ensure all input sequences have a uniform length of maxlen=50. Shorter sequences are padded, and longer sequences are truncated. This is a crucial step for CNNs which expect fixed-size inputs.

BERT Tokenization (Cell 77WlhbpCAIs1): For the BERT model, BertTokenizer.from_pretrained("bert-base-uncased") is used. A tokenize_bert function is defined to perform tokenization specific to BERT's requirements:
add_special_tokens=True: Adds special tokens like [CLS] and [SEP] that BERT expects.
max_length=100: Sets the maximum sequence length.
truncation=True: Truncates sequences longer than max_length.
padding='max_length': Pads shorter sequences to max_length.
return_tensors='tf': Returns TensorFlow tensors.
return_token_type_ids=False and return_attention_mask=True: Specifies the return of attention masks, which BERT uses to distinguish real tokens from padding tokens.
