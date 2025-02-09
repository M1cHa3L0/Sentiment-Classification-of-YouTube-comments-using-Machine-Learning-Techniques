# run preprocess and models code
import pandas as pd
import glob
import preprocess
import time

def read_file():
    file_path = '/Users/apple/Desktop/Master/final project/FinalProjectGit/comment/*.txt'
    all_files = glob.glob(file_path)
    df_list = [pd.read_csv(file, delimiter='\t', low_memory=False) for file in all_files]
    df = pd.concat(df_list, ignore_index=True)
    return df

def save_file(df):
     # save file
    new_file_path = 'file/filtered_comments_cleaned.txt'
    new_file_path_excel = 'file/filtered_comments_cleaned.xlsx'
    new_file_path_csv = 'file/filtered_comments_cleaned.csv'

    # Save the filtered DataFrame to a new text file
    df[['CommentTextDisplay', 'cleanComment', 'Sentiment']].to_csv(new_file_path, sep='\t', index=False)
    df[['CommentTextDisplay', 'cleanComment', 'Sentiment']].to_excel(new_file_path_excel, index=False)
    df[['CommentTextDisplay', 'cleanComment', 'Sentiment']].to_csv(new_file_path_csv, sep='\t', index=False)

    print('saved')



if __name__ == '__main__':
    ######## preprocess data ########
    df = read_file()
    print(df.count())
    
    # select english comments
    sample_Comment = preprocess.select_english_comments(df)
    print('selected')

    # preprocess
    sample_Comment['cleanComment'] = sample_Comment['CommentTextDisplay'].apply(preprocess.preprocess_text)
    print('preprocessed')

    # label sentiment
    sample_Comment['Sentiment'] = sample_Comment['CommentTextDisplay'].apply(preprocess.sentiment_score)
    print('labeled')

    # save file
    save_file(sample_Comment)

    '''
    start_time = time.time()
    end_time = time.time()
    print(f"time: {end_time - start_time:.4f} s")
    '''

    ######## models ########


