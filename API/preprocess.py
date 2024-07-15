import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import glob
from langdetect import detect
import time
from afinn import Afinn

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')

# Initialize the stemmer and stop words list
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
words_set = set(nltk.corpus.words.words())
afinn = Afinn()


# function
# remove emoji
def remove_emoji(text):
    if isinstance(text, str):
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"  # other miscellaneous symbols
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    else:
        return text


# preprocess(include emoji remove) tokenization，stemming，stop word，special character
def preprocess_text(text):
    if isinstance(text, str):
        # Remove emojis
        text = remove_emoji(text)
        # Remove special characters and punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Tokenize the text
        tokens = word_tokenize(text)
        # Remove stop words and apply stemming
        processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
        # Join tokens back into a single string
        return ' '.join(processed_tokens)
    return text


# check comment is english
def is_english(text):
    # 是否是string
    if not isinstance(text, str):
        return False

    # 避免影响后续操作
    text = remove_emoji(text)

    # 使用langdetect库检测text是否是en
    try:
        lang = detect(text)
        if lang != 'en':
            return False
    # 太短或无法识别
    except:
        return False

    # 将comment 分词，判断text跟词库的匹配程度（大于50%则视为英文text）
    words_in_text = set(w.lower() for w in nltk.word_tokenize(text) if w.isalpha())

    if not words_in_text:
        return False

    return len(words_in_text.intersection(words_set)) / len(words_in_text) > 0.5


def select_english_comments(dataframe, num_comment=5000, random_state=1):
    rest_dataframe = dataframe
    sample = rest_dataframe.sample(n=num_comment, random_state=random_state)
    sample = sample[sample['CommentTextDisplay'].apply(is_english)].copy()
    sample = sample['CommentTextDisplay']

    A = pd.DataFrame(rest_dataframe)
    B = pd.DataFrame(sample)
    mask = ~(A.set_index(['CommentTextDisplay']).index.isin(B.set_index(['CommentTextDisplay']).index))
    rest_dataframe = rest_dataframe[mask]

    while len(sample) < num_comment:
        new_sample = rest_dataframe.sample(n=min(num_comment - len(sample), len(rest_dataframe)),
                                           random_state=random_state)
        new_sample = new_sample[new_sample['CommentTextDisplay'].apply(is_english)].copy()

        if len(new_sample) > 0:
            sample = pd.concat([sample, new_sample])

        A = pd.DataFrame(rest_dataframe)
        B = pd.DataFrame(new_sample)
        mask = ~(A.set_index(['CommentTextDisplay']).index.isin(B.set_index(['CommentTextDisplay']).index))
        rest_dataframe = rest_dataframe[mask]

        random_state += 1

    return sample


def sentiment_score(text):
    score = afinn.score(text)
    if score > 0:
        return 1
    elif score < 0:
        return -1
    else:
        return 0


# read all file

file_path = '~/Desktop/Sentiment-Classification-of-YouTube-comments-using-Machine-Learning-Techniques-main/API/comment/*.txt'

all_files = glob.glob(file_path)
df_list = [pd.read_csv(file, delimiter='\t', low_memory=False) for file in all_files]
df = pd.concat(df_list, ignore_index=True)

print(df.count())

# preprocess
# select english comments
start_time = time.time()
sample_Comment = select_english_comments(df)
end_time = time.time()
print(f"耗时: {end_time - start_time:.4f} 秒")
print('1')

#
# unicomment = sample_Comment['VideoID'].unique()
# distinct_video_ids_list = unicomment.tolist()
# print(len(distinct_video_ids_list))


# preprocess
start_time = time.time()
sample_Comment['cleanComment'] = sample_Comment['CommentTextDisplay'].apply(preprocess_text)
end_time = time.time()
print(f"耗时: {end_time - start_time:.4f} 秒")
print('2')

# label machinely
start_time = time.time()
sample_Comment['Sentiment'] = sample_Comment['CommentTextDisplay'].apply(sentiment_score)
end_time = time.time()
print(f"耗时: {end_time - start_time:.4f} 秒")
print('3')

print(len(sample_Comment[sample_Comment['Sentiment'] == 1]))
print(len(sample_Comment[sample_Comment['Sentiment'] == 0]))
print(len(sample_Comment[sample_Comment['Sentiment'] == -1]))

new_file_path = 'filtered_comments_cleaned.txt'
new_file_path_excel = 'filtered_comments_cleaned.xlsx'
new_file_path_csv = 'filtered_comments_cleaned.csv'

# Save the filtered DataFrame to a new text file
sample_Comment[['CommentTextDisplay', 'cleanComment', 'Sentiment']].to_csv(new_file_path, sep='\t', index=False)
sample_Comment[['CommentTextDisplay', 'cleanComment', 'Sentiment']].to_excel(new_file_path_excel, index=False)
sample_Comment[['CommentTextDisplay', 'cleanComment', 'Sentiment']].to_csv(new_file_path_csv, sep='\t', index=False)
