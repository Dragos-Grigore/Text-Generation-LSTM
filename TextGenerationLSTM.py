#!/usr/bin/env python
# coding: utf-8

# In[1]:


def read_file(filepath):
    with open(filepath) as f:
        str_text = f.read()
        
    return str_text


# In[2]:


read_file('moby_dick_four_chapters.txt')


# In[3]:


import spacy


# In[4]:


nlp=spacy.load('en_core_web_sm',disable=['parser','tagger','ner'])


# In[5]:


nlp.max_length=1198623


# In[6]:


def separate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']


# In[7]:


d=read_file('moby_dick_four_chapters.txt')


# In[8]:


tokens=separate_punc(d)


# In[9]:


tokens


# In[10]:


train_len=25+1

text_sequences=[]

for i in range(train_len,len(tokens)):
    seq=tokens[i-train_len:i]
    
    text_sequences.append(seq)


# In[11]:


from keras.preprocessing.text import Tokenizer


# In[12]:


tokenizer=Tokenizer()
tokenizer.fit_on_texts(text_sequences)


# In[13]:


sequences=tokenizer.texts_to_sequences(text_sequences)


# In[14]:


for i in sequences[0]:
    print(f"{i}: {tokenizer.index_word[i]}")


# In[15]:


tokenizer.word_counts


# In[16]:


vocabulary_size=len(tokenizer.word_counts)


# In[17]:


vocabulary_size


# In[18]:


import numpy as np


# In[19]:


sequences=np.array(sequences)


# In[20]:


sequences


# In[21]:


from keras.utils import to_categorical


# In[24]:


X=sequences[:,:-1]


# In[25]:


y=sequences[:,-1]


# In[26]:


y=to_categorical(y,num_classes=vocabulary_size+1)


# In[30]:


seq_len=X.shape[1]


# In[31]:


X.shape


# In[32]:


from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding


# In[33]:


def create_model(vocabulary_size,seq_len):
    
    model=Sequential()
    model.add(Embedding(vocabulary_size,seq_len,input_length=seq_len))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50,activation='relu'))
    
    model.add(Dense(vocabulary_size,activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    model.summary()
    
    return model


# In[34]:


model= create_model(vocabulary_size+1,seq_len)


# In[35]:


from pickle import dump,load


# In[37]:


model.fit(X,y,batch_size=128,epochs=2,verbose=1)


# In[38]:


model.save('my_mobydick_model.h5')


# In[39]:


dump(tokenizer,open('my_simpletokenizer','wb'))


# In[49]:


from keras.preprocessing.sequence import pad_sequences


# In[52]:


def generate_text(model,tokenizer,seq_len,seed_text,num_gen_words):
    
    output_text=[]
    
    input_text=seed_text
    
    for i in range(num_gen_words):
        encoded_text=tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded=pad_sequences([encoded_text],maxlen=seq_len,truncating='pre')
        pred_word_ind=model.predict_classes(pad_encoded,verbose=0)[0]
        pred_word=tokenizer.index_word[pred_word_ind]
        input_text+=' '+pred_word
        output_text.append(pred_word)
    
    return ' '.join(output_text)


# In[42]:


import random
random.seed(101)
random_pick=random.randint(0,len(text_sequences))


# In[43]:


random_seed_text=text_sequences[random_pick]


# In[44]:


random_seed_text


# In[45]:


seed_text=' '.join(random_seed_text)


# In[53]:


generate_text(model,tokenizer,seq_len,seed_text=seed_text,num_gen_words=25)


# In[54]:


from keras.models import load_model


# In[55]:


model=load_model('epochBIG.h5')


# In[56]:


tokenizer=load(open('epochBig','rb'))


# In[57]:


generate_text(model,tokenizer,seq_len,seed_text=seed_text,num_gen_words=25)


# In[ ]:




