
from function.constants import *
import nlpaug.augmenter.word as naw
#.augmenter.word as naw

import nltk
nltk.download('averaged_perceptron_tagger_eng')


def ProcessAugDataframe(df, selected_column, method, **kwargs):
    new_df = df.copy()
    aug = None
    if method == SYNONYM_AUG:
        aug = naw.SynonymAug(**kwargs)
    elif method == ANTONYM_AUG:
        aug = naw.AntonymAug(**kwargs)
    elif method == SPELLING_AUG:
        aug = naw.SpellingAug(**kwargs)

    if aug is None:
        return new_df

    # For each row in new_df apply the augmenter to the selected column
    for i, row in new_df.iterrows():
        new_row = row[selected_column]
        # Apply to the selected column
        new_row = aug.augment(new_row)
        new_df.at[i, selected_column] = new_row
    return new_df

def DataFrameToCSV(dataframe):
    return dataframe.to_csv(index=False)

