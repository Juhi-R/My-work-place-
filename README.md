from bs4 import BeautifulSoup
import spacy
import unidecode
from word2number import w2n
import contractions


nlp = spacy.load("en_core_web_sm")


# exclude words from spacy stopwords list
deselect_stop_words = ['no', 'not']
for w in deselect_stop_words:
    nlp.vocab[w].is_stop = False




def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text




def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())




def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text




def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    text = contractions.fix(text)
    return text




def text_preprocessing(text, accented_chars=True, contractions=True, 
                       convert_num=True, extra_whitespace=True, 
                       lemmatization=True, lowercase=True, punctuations=True,
                       remove_html=True, remove_num=True, special_chars=True, 
                       stop_words=True):
    """preprocess text with default option set to true for all steps"""
    if remove_html == True: #remove html tags
        text = strip_html_tags(text)
    if extra_whitespace == True: #remove extra whitespaces
        text = remove_whitespace(text)
    if accented_chars == True: #remove accented characters
        text = remove_accented_chars(text)
    if contractions == True: #expand contractions
        text = expand_contractions(text)
    if lowercase == True: #convert all characters to lowercase
        text = text.lower()


    doc = nlp(text) #tokenise text


    clean_text = []
    
    for token in doc:
        flag = True
        edit = token.text
        # remove stop words
        if stop_words == True and token.is_stop and token.pos_ != 'NUM': 
            flag = False
        # remove punctuations
        if punctuations == True and token.pos_ == 'PUNCT' and flag == True: 
            flag = False
        # remove special characters
        if special_chars == True and token.pos_ == 'SYM' and flag == True: 
            flag = False
        # remove numbers
        if remove_num == True and (token.pos_ == 'NUM' or token.text.isnumeric()) \
        and flag == True:
            flag = False
        # convert number words to numeric numbers
        if convert_num == True and token.pos_ == 'NUM' and flag == True:
            edit = w2n.word_to_num(token.text)
        # convert tokens to base form
        elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_
        # append tokens edited and not removed to list 
        if edit != "" and flag == True:
            clean_text.append(edit)        
    return clean_text


real_text_frame['clean']=real_text_frame.apply(lambda x: text_preprocessing(x['text']), axis=1)
real_text_frame['clean']=real_text_frame.apply(lambda x: " ".join(x['clean']), axis=1)


from collections import Counter


real_total_text = [text for text in real_text_frame['clean']]
real_total_text = ' '.join(real_total_text).split()


real_counts = Counter(real_total_text)


real_common_words = [word[0] for word in real_counts.most_common(20)]
real_common_counts = [word[1] for word in real_counts.most_common(20)]


fig = plt.figure(figsize=(18,6))
sns.barplot(x=real_common_words, y=real_common_counts)
plt.title('Most Common Words used in Real Job Ads')
plt.show()



fake_text_frame['clean']=fake_text_frame.apply(lambda x: text_preprocessing(x['text']), axis=1)
fake_text_frame['clean']=fake_text_frame.apply(lambda x: " ".join(x['clean']), axis=1)


from collections import Counter


fake_total_text = [text for text in fake_text_frame['clean']]
fake_total_text = ' '.join(fake_total_text).split()


fake_counts = Counter(fake_total_text)


fake_common_words = [word[0] for word in fake_counts.most_common(20)]
fake_common_counts = [word[1] for word in fake_counts.most_common(20)]


fig = plt.figure(figsize=(18,6))
sns.barplot(x=fake_common_words, y=fake_common_counts)
plt.title('Most Common Words used in Fake Job Ads')
plt.show()


