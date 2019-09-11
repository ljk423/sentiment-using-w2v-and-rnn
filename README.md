# sentiment-using-w2v-and-rnn
It's kind of sentiment analysis using word2vec weights and GRU model

This code is a kind of Sentiment Analysis which classify Pos/Neg (binary classification problem)
I basically used amazon website's review data (using crawling code)

From Text data, the code preprocess and cleaning the text data
And then it runs word2vec to get weights.

After that, I embedded all the texts and use word2vec weights as init weights of rnn model (specifically GRU model)

<img src="https://github.com/ljk423/sentiment-using-w2v-and-rnn/blob/master/model.png" width="90%"></img>

I used Bidirectional CuDNNGRU for this model

After training, there is a code which makes a word2vec distance plot

<img src="https://github.com/ljk423/sentiment-using-w2v-and-rnn/blob/master/similar_words.png" width="90%"></img>

And also there is 3D version plot code as well
