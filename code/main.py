# run preprocess and models code
import pandas as pd
import glob
import preprocess
import models as model
import time

def read_file(file_path):
    all_files = glob.glob(file_path)
    if len(all_files) > 1:
        df_list = [pd.read_csv(file, delimiter='\t', low_memory=False) for file in all_files]
        df = pd.concat(df_list, ignore_index=True)
        return df
    else:
        return pd.read_csv(file_path, delimiter='\t', low_memory=False)


def save_file(df):
     # save file
    new_file_path = 'file/filtered_comments_cleaned.txt'
    new_file_path_excel = 'file/filtered_comments_cleaned.xlsx'
    new_file_path_csv = 'file/filtered_comments_cleaned.csv'

    # Save the filtered DataFrame to a new text file
    df[['CommentTextDisplay', 'cleanComment', 'Sentiment']].to_csv(new_file_path, sep='\t', index=False)
    df[['CommentTextDisplay', 'cleanComment', 'Sentiment']].to_excel(new_file_path_excel, index=False)
    df[['CommentTextDisplay', 'cleanComment', 'Sentiment']].to_csv(new_file_path_csv, sep='\t', index=False)

    print('######## file saved')



if __name__ == '__main__':
    ######## preprocess data ########
    file_path = '/Users/apple/Desktop/Master/final project/FinalProjectGit/comment/*.txt'
    df = read_file(file_path)
    print(df.count())
    
    # select english comments
    sample_Comment = preprocess.select_english_comments(df)
    print('######## selected')

    # preprocess
    sample_Comment['cleanComment'] = sample_Comment['CommentTextDisplay'].apply(preprocess.preprocess_text)
    print('######## preprocessed')

    # label sentiment
    sample_Comment['Sentiment'] = sample_Comment['CommentTextDisplay'].apply(preprocess.sentiment_score)
    print('######## labeled')

    # save file
    save_file(sample_Comment)

    '''
    start_time = time.time()
    end_time = time.time()
    print(f"time: {end_time - start_time:.4f} s")
    '''

    ######## models ########

    # read file
    path = '/Users/apple/Desktop/Master/final project/FinalProjectGit/file/filtered_comments_cleaned.txt'
    df = read_file(path)

    # tf-idf
    data = preprocess.tf_idf(df)

    # model training & hyperparameter tuning
    print('######## model training...')
    performance_data = model.train_model(data)

    columns = ['Model'] + [f'Fold_{i+1}' for i in range(10)] + ['Mean Accuracy', 'Best Params']
    performance_df = pd.DataFrame(performance_data, columns=columns)
    print(performance_df)

    # csv file
    performance_df.to_csv('file/model_performance_with_params.csv', index=False)
    print('######## file saved')


