'''
before next meeting:

abs,Intr,Lit,Method,Result,Dissu,Concl

Email: Expadation of the methodology
Literature review
Methodology
Initial results of the data process

(top10 video comments of dude perfect, then random choose max 5k)
'''


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


file_path = '/API/comment/ym9E1YG3_QQ.txt'

# df
df = pd.read_csv(file_path, delimiter='\t')
print(df.count())
'''
# 选comment
comments = df[['CommentTextDisplay', 'VideoID']].reset_index()

comments.columns = ['SequenceNumber', 'CommentTextDisplay', 'VideoID']

comments_array = comments.values.tolist()

# output
for comment in comments_array[:1]:
    print(comment)

# 删除错误comment
unicomment = df[~df['VideoID'].str.contains('http://www.youtube.com', na=False)]

unicomment = unicomment['VideoID'].unique()

distinct_video_ids_list = unicomment.tolist()

# Display the list of distinct VideoID values
print(distinct_video_ids_list)
'''

'''
filtered_comments = df[df['VideoID'] == 'VJwoSfTOhyM']

# Count the number of such rows
num_filtered_comments = filtered_comments.shape[0]

print(num_filtered_comments)

new_file_path = 'VJwoSfTOhyM.txt'

# Save the filtered DataFrame to a new text file
filtered_comments.to_csv(new_file_path, sep='\t', index=False)

'''