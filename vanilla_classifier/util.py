import pandas as pd
from sklearn import metrics
from config import *


def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return roc_t['threshold'].item()

def get_thresholds(result):
    thresholds = []
    for col in char_class_labels:
        threshold = find_optimal_cutoff(result[col], result[col+'_score'])
        thresholds.append(threshold)
    return torch.tensor(thresholds)

def get_diagnosis_predictions(result, num_melanoma_chars=1):
    preds = []
    for idx, row in result.iterrows():
        if (row[mel_class_labels_pred] == 1).sum() >= num_melanoma_chars:
            preds.append(1)
        else:
            preds.append(0)
    return preds

def display_metrics(result):
    print('balanced acc: ', metrics.balanced_accuracy_score(result['benign_malignant'], result['prediction']).round(5))
    print('sensitivity: ', metrics.recall_score(result['benign_malignant'], result['prediction']).round(5))
    print('specificity: ', metrics.recall_score(result['benign_malignant'], result['prediction'], pos_label=0).round(5))



def get_predictions(trainer, model, split='test', dx_threshold=0.5):
    """
    Stores predictions, scores, and true values in a DataFrame.
    """
    
    if split == 'test':
        predictions = trainer.predict(model, model.test_dataloader())
        metadata_df = model.test_set.metadata[["image_id", "lesion_id", "benign_malignant"]]
    elif split == 'val':
        predictions = trainer.predict(model, model.val_dataloader())
        metadata_df = model.val_set.metadata[["image_id", "lesion_id", "benign_malignant"]]
    elif split == 'external':
        predictions = trainer.predict(model, model.external_dataloader())
        metadata_df = model.external_set.metadata[["image_id", "lesion_id", "benign_malignant"]]

    dfs = []

    for preds in predictions:
        
        diagnosis_predictions, y_dx, image_id, x = preds
        
        df_img = pd.DataFrame(image_id, columns=['image_id'])
        
        df_true = pd.DataFrame(y_dx, columns=['true'])
        
        df_score = pd.DataFrame(diagnosis_predictions, columns=['dx_score'])
        
        df = pd.concat([df_img, df_true, df_score], axis=1)
        
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
    result['prediction'] = result['dx_score'].apply(lambda x: 1 if x >= dx_threshold else 0)
    result = pd.merge(metadata_df, result, on='image_id')

    return result


def get_results(model, split, char_threshold, dx_threshold):
        result = get_char_predictions(trainer, model, split=split, threshold=char_threshold)
        result['prediction'] = result['dx_pred'].apply(lambda x: 1 if x >= dx_threshold else 0)
        return result
    


def get_dx_predictions(trainer, model):
    """
    Stores predictions, scores, and true values in a DataFrame.
    """

    predictions = trainer.predict(model)

    dfs = []

    for preds in predictions:

        y_pred, y_true = preds

        df = pd.DataFrame(y_pred, columns=['score'])
        df['pred'] = df['score'].round()
        df['true'] = pd.DataFrame(y_true, columns=['true'])

        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True).dropna()

    return result


def display_scores(result):

    for true, pred, score in zip(char_class_labels, char_class_labels_pred, char_class_labels_score):
        print('\n=====')
        print(true)
        try:
            print('AUC:', metrics.roc_auc_score(result[true], result[score]))
            print('Balanced Acc:', metrics.balanced_accuracy_score(result[true], result[pred]))
            print('Sensitivity:', metrics.recall_score(result[true], result[pred]))
            print('Specificity:', metrics.recall_score(result[true], result[pred], pos_label=0))
        except Exception as e:
            print(e)
        print('=====\n')


    
"""
def setup():
    
    metadata = pd.read_csv(metadata_file)
    #metadata = metadata[metadata.dataset.isin(['ham', 'scp'])]

    test_set = metadata[metadata['split'] == 'test']
    train = metadata[metadata['split'] == 'train']#.drop_duplicates(subset='lesion_id', keep='last')
    external_set = metadata[metadata['split'] == 'external']

    # Drop lesion Ids from train set that are also in test set
    train = train[~train['lesion_id'].isin(test_set['lesion_id'])]

    train_lesions, val_lesions = train_test_split(
        train.drop_duplicates('lesion_id')['lesion_id'], test_size=0.18, stratify=train.drop_duplicates('lesion_id')[dx_class_label],
        random_state=seed
    )

    train_set = train[train['lesion_id'].isin(train_lesions)].drop_duplicates(subset='lesion_id', keep='last')
    val_set = train[train['lesion_id'].isin(val_lesions)].drop_duplicates(subset='lesion_id', keep='last')
    
"""    