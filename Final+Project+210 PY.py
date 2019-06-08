
# coding: utf-8

# ## Analyzing Children's Stories using Python 
# 
# ## Works Chosen:
# #### [The Blue Fairy Book (1889)](https://en.wikipedia.org/wiki/Andrew_Lang%27s_Fairy_Books#The_Blue_Fairy_Book_(1889) by [Andrew Lang](https://en.wikipedia.org/wiki/Andrew_Lang)
#  - First book in the Fairy Book series (25 books in the Fairy series)
#  - Stories were pulled from many different sources, like the Brothers Grimm, but many were left unattributed.
#  - Recognizable stories include: Little Red Riding Hood, The Sleeping Beauty in the Wood, Cinderella, Aladdin, Rumpelstiltskin, and etc.
#  
# #### [Russian Fairy Tales: A Choice Collection of Muscovite Folk-lore (1887)](https://archive.org/details/russianfairytale00rals) by [William Ralston Shedden-Ralston](https://en.wikipedia.org/wiki/William_Ralston_Shedden-Ralston)
#  - Also published under the title "Russian Folk Tales"
#  - As the name suggests, this book pulls from Russian folklore. Details are scarce, but WRS mostly translated this himself.
#  - Interestingly enough, the book is dedicated to Alexander Afanasyev, a man who published his own *"Народные Русские Сказки" *or,* "Russian Folk Tales"* between 1855 and 1863.
#  - Recognizable stories include Baba Yaga
#  - Other stories include: The Dead Mother, The Headless Princess, and The Two Corpses.
# 

# ## Analyzing Children's Stories Using Python
# #### Allison Lourens
# 
#   I chose to do my Python and NLTK analysis project because what stories and what lessons we teach our children are important to their growth and how they develop. What we teach our children is what they learn. What we consider important when it comes to values differs by people, families, cities, and even countries. This project will be focusing on a bigger scale rather than a smaller one by comparing the [Blue Fairy Book](http://www.gutenberg.org/ebooks/503) and [Russian Fairy Tales](https://www.gutenberg.org/ebooks/22373), both available on Project Gutenberg.  
#   First, we'll discuss the books themselves. *The Blue Fairy Book* was written in 1889 by Andrew Lang, who was a Scottish poet, novelist, and literary critic who was most famous for collecting folk and fairy tales. The *Blue Fairy Book* was the first book in the *Fairy Book* series and many stories from the book were pulled from lots of different sources, but left unattributed. Recognizable stories include "Little Red Riding Hood", "Cinderella", and "Aladdin". The *Russian Fairy Tales*  book was written in 1887 by William Ralston Shedden-Ralston. Shedden-Ralston was a noted translator of Russian and a British scholar who was gifted with storytelling. As the name suggests, The *Russian Fairy Tales* book was a collection of Russian folk tales and stories. Details are scarce, but Shedden-Ralston likely translated the stories himself. Recognizable stories include "Baba Yaga".   

# In[1]:


import nltk
import os
import requests 
from nltk.corpus import stopwords


# In[2]:


from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize  import word_tokenize

import string
import gensim
from gensim import corpora


# ### Setting Up: Requesting and Tokenizing and Etc.

# In[3]:


requestsObj = requests.get("https://www.gutenberg.org/files/503/503-0.txt")

requestsObj.status_code == requests.codes.ok 

#gfile = open("data/bluebook.txt","wb") commented out- licensing has been edited, and i don't want to rerun and have to deal with it again!

#for chunk in requestsObj.iter_content(1000000): 
    #gfile.write(chunk)
#gfile.close()


# In[4]:


requestsObj = requests.get("https://www.gutenberg.org/files/22373/22373-0.txt")

requestsObj.status_code == requests.codes.ok

#gfile = open("data/russian.txt","wb") same here- I promise I've run this before and it worked

#for chunk in requestsObj.iter_content(1000000):
    #gfile.write(chunk)
#gfile.close()


# In[5]:


BlueBookString = open("data/bluebook.txt", "r", encoding = "utf-8").read()
RussianString = open("data/russian.txt", "r", encoding = "utf-8").read()


# In[6]:


len(BlueBookString)


# In[7]:


len(RussianString)


# In[8]:


BlueBookTokenized = nltk.word_tokenize(BlueBookString)
RussianTokenized = nltk.word_tokenize(RussianString)


# In[9]:


len(BlueBookTokenized)


# In[10]:


len(RussianTokenized)


# In[11]:


BlueBookTokenized = [word for word in BlueBookTokenized if word[0].isalpha()] 
BlueBookTokenizedlower = [word.lower() for word in BlueBookTokenized]


# In[12]:


RussianTokenized = [word for word in RussianTokenized if word[0].isalpha()] 
RussianTokenizedlower = [word.lower() for word in RussianTokenized]


# ## Analysis Part 1: Lexical Diversity and Comparison
# 
# First, we'll analyze the lexical diversity and compare the two books. This is important because the lexical diversity of the books is determined by the author's or authors' writing style.

# In[13]:


doc_blue = [BlueBookTokenizedlower]
doc_rus = [RussianTokenizedlower]

doc_all = [doc_blue, doc_rus]


# In[14]:


#stop words list
stop = set(stopwords.words('english'))


# In[15]:


#punct list
exclude = set(string.punctuation)


# In[16]:


print("Unique Words: Basic Analysis Results")
print("Blue Fairy Book:", len(BlueBookTokenizedlower), "words.")
print("Russian Folk Tales:", len(RussianTokenizedlower), "words.")
print("Blue Fairy Book:", len(set(BlueBookTokenizedlower)), "unique words.")
print("Russian Folk Tales:", len(set(RussianTokenizedlower)), "unique words.")


# ## Results and Inferring
# 
# While the Blue Fairy Book has more words, the Russian Folk Tales has a greater lexical diversity. My theory on why this is the case is that the stories from the Blue Fairy Book are quite popular, these stories are most likely lifted word-for-word from other books. In addition to this, the stories in the Blue Fairy Book had been translated for a while before that, and have been copied down before. The copying and re-copying of these stories waters them down to common English, reducing the lexical diversity. However, the Russian Folk Tales were probably translated for maybe one of the first times in this collection. In line with that logic, Shedden-Ralston mostly likely tried to find different words to properly express words that had been originally written in Russian. 

# ## Analysis Part 2: Characters and Lessons
# 
# With the background and lexical makeup of these books discussed, we can discuss a lot more about the two most important things about fairy tales, folk stories, and other lessons discussed by these books: Who is being taught these lessons, and what are the lessons being taught? The characters learning these lessons are important because ultimately, the main character of most fairy tales is who we're meant to emulate, learn from, or identify with. Who we're meant to identify with depends on a lot, but the origin of many of these tales often concerns gender, as we lived (and still live) in a gendered society. Who we're concerned with teaching and what we teach them is what we'll take a look at.

# ## 2A: Characters:

# In[17]:


wnl = WordNetLemmatizer()


# In[18]:


blue_lemmas = [wnl.lemmatize(word) for word in BlueBookTokenizedlower]
rus_lemmas = [wnl.lemmatize(word) for word in RussianTokenizedlower]
print("The Blue Fairy Book:")
print("Count of 'child' in tokens: ", BlueBookTokenizedlower.count("child"))
print("Count of 'adult' in tokens: ", BlueBookTokenizedlower.count("adult"))
print("Count of 'girl' in tokens: ", BlueBookTokenizedlower.count("girl"))
print("Count of 'boy' in tokens: ", BlueBookTokenizedlower.count("boy"))
print("Count of 'child' in lemmas: ", blue_lemmas.count("child"))
print("Count of 'adult' in lemmas: ", blue_lemmas.count("adult"))
print("Count of 'girl' in lemmas: ", blue_lemmas.count("girl"))
print("Count of 'boy' in lemmas: ", blue_lemmas.count("boy"))

print("Russian Folk Tales:")
print("Count of 'child' in tokens: ", RussianTokenizedlower.count("child"))
print("Count of 'adult' in tokens: ", RussianTokenizedlower.count("adult"))
print("Count of 'girl' in tokens: ", RussianTokenizedlower.count("girl"))
print("Count of 'boy' in tokens: ", RussianTokenizedlower.count("boy"))
print("Count of 'child' in lemmas: ", rus_lemmas.count("child"))
print("Count of 'adult' in lemmas: ", rus_lemmas.count("adult"))
print("Count of 'girl' in lemmas: ", rus_lemmas.count("girl"))
print("Count of 'boy' in lemmas: ", rus_lemmas.count("boy"))


# In[19]:


get_ipython().magic('matplotlib inline')
nltk.Text(blue_lemmas).dispersion_plot(["girl", "boy"])
nltk.Text(rus_lemmas).dispersion_plot(["girl", "boy"])


# ### Raw Data
# #### Gender of Protagonist:
# 
# ##### Blue Fairy Book: 
# *Count of 'girl' in tokens:  69*  
# *Count of 'boy' in tokens:  71*
# 
# *Count of 'girl' in lemmas:  91*  
# *Count of 'boy' in lemmas:  80*
# 
# ##### Russian Folk Tales
# *Count of 'girl' in tokens:  81*  
# *Count of 'boy' in tokens:  36*
# 
# *Count of 'girl' in lemmas:  **123*** 
# *Count of 'boy' in lemmas:  **39***
# 
# ## Analysis of Results 2A:
# 
# As a preliminary check, I first looked at the count of "adult" to see if there were any adults mentioned. There were 0 mentions of "adult", confirming that yes, these are stories for children. Next, I took a look at the gender of the protagonists in the stories. There were 91 counts of "girl" and 80 counts of "boy" within the *Blue Fairy Book*, indicating a nice balance. However, there were 39 counts of "boy" and **123** counts of "girl" within the *Russian Folk Tales* book. But these are strange results, and will be verified. We'll do that next by searching for names.

# ## 2A: Names

# In[20]:


from nltk.corpus import names 
from nltk.corpus import words
from string import punctuation


# In[21]:


fnames = [fn.lower() for fn in names.words("female.txt")]
mnames = [mn.lower() for mn in names.words("male.txt")]


# In[22]:


blue_words = BlueBookString.split()
rus_words = RussianString.split()


# In[23]:


female_counts = {}
for w in blue_words: 
    if w in fnames: 
        female_counts[w] = female_counts.get(w,0) + 1
    
blue_female_names = list(female_counts.keys())
blue_female_names.sort()

print("Amount of girl names in The Blue Fairy Book:", len(blue_female_names))


# In[24]:


male_counts = {}
for w in blue_words: 
    if w in mnames: 
        male_counts[w] = male_counts.get(w,0) + 1

blue_male_names = list(male_counts.keys())
blue_male_names.sort()

print("Amount of boy names in The Blue Fairy Book:", len(blue_male_names))


# In[25]:


female_counts = {}
for w in rus_words: 
    if w in fnames: 
        female_counts[w] = female_counts.get(w,0) + 1
    
rus_female_names = list(female_counts.keys())
rus_female_names.sort()

print("Amount of girl names in Russian Fairy Tales:", len(rus_female_names))


# In[26]:


male_counts = {}
for w in rus_words: 
    if w in mnames: 
        male_counts[w] = male_counts.get(w,0) + 1

rus_male_names = list(male_counts.keys())
rus_male_names.sort()

print("Amount of boy names in Russian Fairy Tales:", len(rus_male_names))


# ## Verification and Interpretation:
# 
# Amount of girl names in The Blue Fairy Book: 71  
# Amount of boy names in The Blue Fairy Book: 96
# 
# Amount of girl names in Russian Fairy Tales: 61  
# Amount of boy names in Russian Fairy Tales: 83
# 
# In *The Blue Fairy Book*, the amount of girl names was 71, and the amount of boy names was 96, which holds, but with a slight leaning towards boys. In contrast to what we saw before, the amount of girl names in *Russian Fairy Tales* is 61 compared to 83 boy names. To validify this, we'll take a look at the number of personal pronouns used.

# ## But Wait, There's More!

# In[27]:


from nltk.tokenize import PunktSentenceTokenizer


# In[28]:


BlueBookPOSTag = (nltk.pos_tag(BlueBookTokenizedlower))
RussianPOSTag = (nltk.pos_tag(RussianTokenizedlower))


# In[29]:


btag_fd = nltk.FreqDist(tag for (word,tag) in BlueBookPOSTag)
btag_fd.most_common(10)


# In[30]:


rtag_fd = nltk.FreqDist(tag for (word,tag) in RussianPOSTag)
rtag_fd.most_common(10)


# In[34]:


BlueBookPronouns = [token for token, pos in BlueBookPOSTag if "PRP" == pos]
BlueBookPronounFreq = nltk.FreqDist(BlueBookPronouns)
BlueBookPronounFreq.plot(20)


# In[32]:


RussianPronouns = [token for token, pos in RussianPOSTag if "PRP" == pos]
RussianPronounFreq = nltk.FreqDist(RussianPronouns)
RussianPronounFreq.plot(20)


# ## Final results and Analysis 
# 
# In both books, there was an overwhelming lean towards "he" versus "she". At first glance, it might seem that many of the stories are about girls, but as the books were analyzed further, it was revealed that these stories are focused on male characters! In fact, in both cases, "you" is more popular than "she", indicating a story about the indeterminate reader is preferable to a story about a girl. Unsurprisingly, this connects to the common knowledge that we live in a masculine-focused world. What this tells us about the societies these were read in is that they were concerned with what they taught boys, and less concerned for the lessons girls learn.

# ## 2B: What?

# ## Issues
# 
# I ran into some issues, mainly concerning my hardware. I was able to find some sort of workaround, however. My issue occured when attempting sentiment analysis. The issue occurs in the line "featuresets = [(find_features(rev),category) for (rev,category) in documents]". When I ran this line specifically, Jupyter kept giving me MemoryError. I suspect this is an issue with Jupyter running in-browser, because I'm sure that my computer has enough RAM to complete this, but I found a  work-around that was successful to some degree using the same tactic, but with encouraging or discouraging words. I've edited this out of the .py file.

# In[87]:


blue_lemmas = [wnl.lemmatize(word) for word in BlueBookTokenizedlower]
rus_lemmas = [wnl.lemmatize(word) for word in RussianTokenizedlower]


# In[1]:


print("Count of 'always' in Blue Fairy Book: ", blue_lemmas.count("always"))
print("Count of 'never' in Blue Fairy Book: ", blue_lemmas.count("never"))
print("Count of 'always' in Russian Folk Tales: ", rus_lemmas.count("always"))
print("Count of 'never' in Russian Folk Tales: ", rus_lemmas.count("never"))


# In[93]:


get_ipython().magic('matplotlib inline')
nltk.Text(blue_lemmas).dispersion_plot(["always", "never"])
nltk.Text(rus_lemmas).dispersion_plot(["always", "never"])


# ## Final Analysis and Reflection:
# 
# As you can see, the use of always and never is more frequent within the *Blue Fairy Book* than the *Russian Fairy Tales*. What this says is that the *Blue Fairy Book* is more direct than the *Russian Fairy Tales* book, where it is willing to distinctly tell the reader what the lesson is. 
# 
# Overall, while the *Blue Fairy Book* and *Russian Fairy Tales* book both seemed like they focused on stories for girls, more analysis disproved this, and it was shown that the stories are more focused on boys, which makes sense, given the masculine-glorifying society we live in. In addition, the *Blue Fairy Book* is more forthcoming with the lessons it wants you to learn. Meanwhile, the *Russian Fairy Tales* book is vague and not as clear with its messages. 
# 
# The stories we told and still tell today are the values we inherit and choose to pass on: What we tell our children is what they carry with them, and what they will too eventually pass on. It doesn't come as a surprise to see the focus on boys in these books when we look at the society we live in that follows these values. This is why these fairy tales may not seem important, but are important for the effects they have: they provide role models and lessons. While change may come eventually, it is ultimately up to each individual person what lessons and values they wish to pass on. 
