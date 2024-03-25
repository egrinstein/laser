# This file is used to create a .csv file for the audiocaps dataset, which is later used to create the json file.
# The first step consists of converting the AudioCaps captions into the AudioSet captions.
# This is done using the AudioSet and AudioCaps .csv files.
# The AudioCaps .csv files contain the following columns:
#
# audiocap_id,youtube_id,start_time,caption
# 91139,Zn4ViKWcw,130,A woman talks nearby as water pours
#
# The AudioSet .csv files contain the following columns:
#
# YTID, start_seconds, end_seconds, positive_labels
# --Zn4ViKWcw, 30.000, 40.000, "/m/09x0r"
# 
# Note the first two rows of AudioSet are ignored, as they contain some useless metadata. 
#
# In turn, the positive labels are converted into textual descriptions using the class_labels_indices.csv file.
# The class_labels_indices.csv file contains the following columns:
# index,mid,display_name
# 0,/m/09x0r,"Speech"

# The second step consists of creating the template json file.
# The template json file is created by iterating over the AudioCaps .csv file and creating a dictionary for each row.
# The dictionary contains the path to the audio file and the textual description.
# The dictionary is then appended to a list.
# Finally, the list is written to a json file.

import pandas as pd


def main():
    # 1.1. Load the AudioCaps csvs
    print("Loading the AudioCaps and AudioSet .csv files")
    audiocaps_train = pd.read_csv('config/datafiles/train.csv')
    audiocaps_val = pd.read_csv('config/datafiles/val.csv')
    audiocaps_test = pd.read_csv('config/datafiles/test.csv')
    
    # 1.2. Load audioset csv (unbalanced_train_segments.csv), ignore first two rows
    audioset_train = pd.read_csv(
        'config/datafiles/unbalanced_train_segments.csv',
        sep=', ', 
        skiprows=2,
        engine='python'
    )
    # 1.2.1. Remove the quotes from the 'positive_labels' column
    audioset_train['positive_labels'] = audioset_train['positive_labels'].str.replace('"', '')
    # 1.3. Load the class_labels_indices.csv, which maps the codes to captions
    audioset_captions = pd.read_csv('config/datafiles/class_labels_indices.csv')
    audioset_captions_dict = dict(zip(audioset_captions['mid'], audioset_captions['display_name']))

    # 2.1. Convert the AudioCaps captions into the AudioSet captions
    # To do this, we will create a new column in the AudioCaps dataframes
    # called audioset_caption.
    # To populate this colunm, we need to find the find the corresponding
    # row in the AudioSet dataframe and get its 'positive_labels' column.
    print("Converting the AudioCaps captions into the AudioSet captions")

    merged_train = audiocaps_train.merge(audioset_train, left_on='youtube_id', right_on='# YTID')
    merged_val = audiocaps_val.merge(audioset_train, left_on='youtube_id', right_on='# YTID')
    merged_test = audiocaps_test.merge(audioset_train, left_on='youtube_id', right_on='# YTID')

    # 2.2. Convert the positive labels into textual descriptions
    # Note that each row in the 'positive_labels' column is a string
    # which possibly contains multiple labels, separated by commas.
    # We will split the string by commas and then find the corresponding
    # caption in the audioset_captions dataframe.
    # As the captions may contain commas (for example, "Male speech, man speaking"),
    # we will store the captions in a list and then join them using the '|' character.

    def add_audioset_captions_to_audiocaps(df):
        df['audioset_caption'] = df['positive_labels'].apply(
            lambda x: '| '.join([audioset_captions_dict[label.strip()] for label in x.split(',')]))
        return df
    print("Converting the positive labels into textual descriptions")
    merged_train = add_audioset_captions_to_audiocaps(merged_train)
    merged_val = add_audioset_captions_to_audiocaps(merged_val)
    merged_test = add_audioset_captions_to_audiocaps(merged_test)

    # 3. Save the new csv files
    print("Saving the new csv files")
    merged_train.to_csv('config/datafiles/train_audioset.csv')
    merged_val.to_csv('config/datafiles/val_audioset.csv')
    merged_test.to_csv('config/datafiles/test_audioset.csv')

    
if __name__ == '__main__':
    main()