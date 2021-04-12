# COMP5046 Assignment 1 [Individual Assessment] (20 marks)
<h2>Submission Due: May 2nd, 2021 (11:59PM)</h2>


<b>[XXX] = Lecture/Lab Reference</b><br/>
<b>(Justify your decision) =</b> Please justify your decision/selection in the documentation.  You must show your final decision in your report with empirical evidence.<br/>
<b>(Explain the performance) =</b> Please explain the trend of performance, and the reason (or your opinion) why the trends show like that

<br>




<h1>Sentiment Analysis using Recurrent Neural Networks</h1>
<p><b>In this assignment1, we will focus on developing sentiment analysis model using Recurrent Neural Networks (RNN). </b><br/>
Sentiment analysis <b>[Lecture5]</b> is contextual mining of text which identifies and extracts subjective information in source material, and helps a business to understand the social sentiment of their brand, product or service while monitoring online conversations.<br/><br/>
<i>For your information, the detailed information for each implementation step was specified in the following sections. Note that lab exercises would be a good starting point for the assignment. The useful lab exercises are specified in each section.</i></p>

<br/>
<hr>
<br/>


<h2>1. Data Preprocessing (2 marks)</h2>
<p>In this assignment, you are to use the <b>NLTK's Twitter_Sample</b> dataset. Twitter is well-known microblog service that allows public data to be collected via APIs. NLTK's twitter corpus currently contains a sample of Tweets retrieved from the Twitter Streaming API. If you want to know the more detailed info for the nltk.corpus, please check the <a href="https://www.nltk.org/howto/corpus.html">nltk corpus website</a>.<br/>
The dataset contains twitter posts (tweets) along with their associated binary sentiment polarity labels. Both the training and testing sets are provided in the form of pickle files (testing_data.pkl, training_data.pkl) and can be downloaded from the Google Drive using the provided code in the <b><a href="https://colab.research.google.com/drive/1A6azpUOCUU923JF5B4v7t2pNSzLAQ20t?usp=sharing">Assignment 1 Template ipynb</a></b>.</p>
<p>
In this Data Preprocessing section, you are required to complete the following section in the format:</br>
<ul>
  <li><b>Preprocess data</b>: You are asked to pre-process the training set by integrating several text pre-processing techniques <b><i>[Lab5]</i></b> (e.g. tokenisation, removing numbers, converting to lowercase, removing stop words, stemming, etc.). You should justify the reason why you apply the specific preprocessing techniques <b>(Justify your decision)</b>
  </li>
 </ul>
</p>


<br/>
<hr>
<br/>


<h2>2. Model Implementation <b>(7 marks)</b></h2>
<p>In this section, you are to implement three components, including Word Embedding module, Lexicon Embedding module, and Bi-directional RNN Sequence Model. For training, you are free to choose hyperparameters <b><i>[Lab2,Lab4,Lab5]</i></b> (e.g. dimension of embeddings, learning rate, epochs, etc.).</p>

<img src="https://github.com/usydnlp/COMP5046/blob/main/img/COMP5046_A1_sentiment.png" width="500px"/>

<p>The model architecture can be found in the <b><i>[Lecture5]</i></b></p>

<h3>1)Word Embedding <b>(2 marks)</b></h3>
First, you are asked to build a word embedding model (for representing word vectors, such as word2vec-CBOW, word2vec-Skip gram, fastText, and Glove) for the input embedding of your sequence model <b><i>[Lab2]</i></b>. Note that we used one-hot vectors as inputs for the sequence model <i>in the Lab3 and Lab4</i>. You are required to complete the following sections in the format:
<ul>
  <li><b>Preprocess data for word embeddings</b>: You are to use and preprocess NLTK Twitter dataset (the one provided in the <i>Section 1</i>) and/or any Dataset (e.g. TED talk, Google News) for word embeddings  <b><i>[Lab2]</i></b>. This can be different from the preprocessing technique that you used in Section 1. You can use both training and testing dataset in order to train the word embedding. <b>(Justify your decision)</b> </li>
  
  <li><b>Build training model for word embeddings</b>: You are to build a training model for word embeddings. You are required to articulate the hyperparameters <b><i>[Lab2]</i></b> you chose (dimension of embeddings, window size, learning rate, etc.). Note that any word embeddings model <b><i>[Lab2]</i></b> (e.g. word2vec-CBOW, word2vec-Skip gram, fasttext, glove) can be applied. <b>(Justify your decision)</b> </li>
  
  <li><b>Train model</b>: You are to train the model.</li>
</ul>
  

<h3>2)Lexicon Embedding <b>(2 marks)</b></h3>
<p>Then, you are to check whether each word is in the positive or negative lexicon. In this assignment, we will use the <a href="http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar">Opinion Lexicon</a> <b>(If you cannot downalod this, please right click and open in a new page or You can directly download from the data folder in this github)</b>, which includes a list of english positive and negative opinion words or sentiment words. <b>(2006 positive and 4783 negative words)</b><br/>
Each word needs to be converted into one-dimensional categorical embedding with three categories, such as not_exist(0), negative(1), and positive(2).
This 0,1,2 categories will be used for the input for the Section 2.3 Bi-directional RNN Sequence model. <br/>
NOTE: If you want to use more than one-dimensional or not using categorical embedding, please <b>(Justify your decision)</b> </p>


<h3>3)Bi-directional RNN Sequence Model <b>(3 marks)</b></h3>
Finally, you are asked to build the Many-to-One (N to 1) Sequence model in order to detect the sentiment/emotion. Note that your model should be the best model selected from the evaluation (will be discussed in the Section 3. Evaluation). You are required to implement the following functions:
<ul>
  <li><b>Apply/Import Word and Lexicon Embedding as an input</b>: You are to concatenate the trained word embedding and lexicon embedding, and apply to the sequence model</li>
  
  <li><b>Build training sequence model</b>: You are to build the Bi-directional RNN-based (Bi-RNN or Bi-LSTM or Bi-GRU) Many-to-One (N to One) sequence model (N: word, One: Sentiment - Positive or Negative). You are required to describe how hyperparameters <b><i>[Lab4,Lab5]</i></b> (the Number of Epochs, learning rate, etc.) were decided. <b>(Justify your decision)</b> </li>
  
  <li><b>Train model</b>: While the model is being trained, you are required to display the Training Loss and the Number of Epochs. <b><i>[Lab4,Lab5]</i></b> </li>
</ul>

<h5>Note that it will not be marked if you do not display the Training Loss and the Number of Epochs in the Assignment 1 ipynb.</h5>



<br/>
<hr>
<br/>


<h2>3. Evaluation (7 marks)</h2>
<p>After completing all model training (in Section 1 and 2), you should evaluate two points: 1)Word Embedding Evaluation and 2)Sentiment Analysis Performance Prediction (Apply the trained model to the test set)</p>
<ol>
  <li><b>Word Embedding Evaluation (3 marks)</b>: Intrinsic Evaluation <b><i>[Lecture3]</i></b> - You are required to apply Semantic-Syntactic word relationship tests for understanding of a wide variety of relationships. The example code is provided <a href="https://colab.research.google.com/drive/1VdNkQpeI6iLPHeTsGe6sdHQFcGyV1Kmi?usp=sharing">here - Word Embedding Intrinsic Evaluation</a> (This is discussed and explained in the <b><i>[Lecture5 Recording]</i></b> ). You also are to visualise the result (the example can be found in the Table 2 and Figure 2 from the <a href="https://nlp.stanford.edu/pubs/glove.pdf">Original GloVe Paper</a>) <b>(Explain the performance)</b> 
 </li>
  
  <li><b>Performance Evaluation (2 marks)</b>: You are to represent the precision, recall, and f1 <b><i>[Lab4]</i></b> of your model in the table <b>(Explain the performance)</b></li>
  
  <li><b>Hyperparameter Testing (2 marks)</b>: You are to provide the line graph, which shows the hyperparameter testing (with the test dataset) and explain the optimal number of epochs based on the learning rate you choose. You can have multiple graphs with different learning rates. In the graph, the x-axis would be # of epoch and the y-axis would be the f1.  <b>(Explain the performance)</b></li>
  
</p>

<h5>Note that it will not be marked if you do not display it in the ipynb file.</h5>


<br/>
<hr>
<br/>


<h2>4. Documentation (4 marks)</h2>
<p>In the section 1,2, and 3, you are required to describe and justify any decisions you made for the final implementation. You can find the tag <b>(Justify your decision)</b> or <b>(Explain the performance)</b> for the point that you should justify the purpose of applying the specific technique/model and explain the performance.<br/>
For example, for section 1 (preprocess data), you need to describe which pre-processing techniques (removing numbers, converting to lowercase, removing stop words, stemming, etc.) were conducted and justify your decision (the purpose of choosing a specific pre-processing techniques, and benefit of using that technique or the integration of techniques for your AI) in your ipynb file</p>
  
<br/>
<hr>
<br/>


<h2>Submission Instruction</h2>
<p>Submit an ipynb file - (file name: your_unikey_COMP5046_Ass1.ipynb) that contains all above sections(Section 1,2,3, and 4).<br/>
 The ipynb template can be found in the <a href="https://colab.research.google.com/drive/1A6azpUOCUU923JF5B4v7t2pNSzLAQ20t?usp=sharing">Assignment 1 template</a></p>

<br/>
<hr>
<br/>


<h2>FAQ</h2>
<p>
  <b>Question:</b> What do I need to write in the justification? How much do I need to articulate?<br/>
  <b>Answer:</b> As you can see the 'Read me' section in the ipynb Assingment 1 template, visualizing the comparison of different testing results is a good to justify your decision. You can find another way (other than comparing different models) as well - like showing any theoretical comparison or using different hyper parameters </p>

<p>
  <b>Question:</b> Is there any marking scheme/marking criteria available for assignment 1?<br/>
  <b>Answer:</b> The assignment specification is extremely detailed. The marking will be conducted based on the specification.
</p>

<p>
  <b>Question:</b> My Word Embedding/ Sentiment Analysis performs really bad (Low accuracy). What did i do wrong?<br/>
  <b>Answer:</b> Please don't bother about the low accuracy as our training dataset is very small and your model is very basic deep learning model. 
</p>

<p>
  <b>Question:</b> Do I need to use only NLTKTwitter dataset for training the word embedding?<br/>
  <b>Answer:</b> No, as mentioned in the lecture 5 (assignment 1 specification description), you can use any dataset (including TED, Google News) or NLTKtwitter dataset for training your word embedding. Word embedding is just for training the word meaning space so you can use any data. 
  Note: Training word embedding is different from training the Bi-RNN prediction model for sentiment analysis. For the Bi-RNN sentiment analysis model training, you should use only training dataset (from  the NLTK twitter dataset that we provided in the assignment 1 template)
</p>
  

<h5>If you have any question, please come to LiveQA and post it in the Edstem anytime!</h5>
