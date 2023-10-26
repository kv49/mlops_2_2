import os
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

from clearml import Task, Logger
task = Task.init(project_name="TPS", task_name='blending_NN')

parameters = {
    'TEST_SIZE' : 0.2,
    'SEED' : 42,
    'EPOCH': 6
}

parameters = task.connect(parameters)

path = Path('/home/user02/Projects/mlops_2_2')

submission = pd.read_csv(path / 'sample_submission.csv', index_col='id')
labels = pd.read_csv(path / 'train_labels.csv', index_col='id')

# the ids of the submission rows (useful later)
sub_ids = submission.index

# the ids of the labeled rows (useful later)
gt_ids = labels.index 

# list of files in the submission folder
#subs = sorted(os.listdir(path / 'submission_files'))

#s0 = pd.read_csv(path / 'submission_files' / subs[4998], index_col='id')
#score = log_loss(labels, s0.loc[gt_ids])
#print(subs[0], f'{score:.10f}')

#s1 = pd.read_csv(path / 'submission_files' / subs[4999], index_col='id')
#score = log_loss(labels, s1.loc[gt_ids])
#print(subs[1], f'{score:.10f}')

#blend = (s0 + s1) / 2

#blend = pd.read_csv(path / 'submission_files' / subs[0], index_col='id')
#for i in np.arange(3):
#    tmp = pd.read_csv(path / 'submission_files' / subs[i], index_col='id')
#    blend = blend + tmp
#blend = blend / (3)

#blend = pd.read_csv(path / 'submission_files' / subs[0], index_col='id')
#blend['pred'] = 0.5

X_all_df = pd.read_csv(path / 'all_submissions.csv', delimiter = ',', low_memory=False)
labels1 = labels.reset_index()


if parameters['TEST_SIZE'] > 0:
    train_ids, test_ids, train_labels, test_labels = \
        train_test_split(labels1['id'], labels1['label'],
                         test_size=parameters['TEST_SIZE'],
                         random_state=parameters['SEED'], stratify=labels1['label'])
    df_train = X_all_df[X_all_df['id'].isin(train_ids)]
    df_test = X_all_df[X_all_df['id'].isin(test_ids)]
else:
    df_train = X_all_df[0:20000]
    df_test = X_all_df[0:0]


x_train = df_train.loc[:, '0':].to_numpy()
y_train = labels1[labels1['id'].isin(train_ids)]['label'].to_numpy()

x_test = df_test.loc[:, '0':].to_numpy()
y_test = labels1[labels1['id'].isin(test_ids)]['label'].to_numpy()

df_submit = X_all_df[X_all_df['id'].isin(sub_ids)]
x_submit = df_submit.loc[:, '0':].to_numpy()

clf = MLPClassifier(hidden_layer_sizes=(5000,),
                    max_iter=parameters['EPOCH'],
                    random_state=parameters['SEED'],
                    verbose=True)
clf.fit(x_train, y_train)

task.get_logger().report_scalar(title='clf.loss_', series='clf.loss_', value=clf.loss_, iteration=parameters['EPOCH'])
if parameters['TEST_SIZE'] > 0:
    test_pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)
else:
    test_pred = clf.predict(x_train)
    y_prob = clf.predict_proba(x_train)

accuracy = accuracy_score(y_test, test_pred)
score = log_loss(y_test, y_prob.transpose()[1])
task.get_logger().report_scalar(title='accuracy_score', series='accuracy_score', value=accuracy, iteration=parameters['EPOCH'])
task.get_logger().report_scalar(title='log_loss', series='log_loss', value=score, iteration=parameters['EPOCH'])

#task.apload_artifact(name='Score', artifact_object={'score', score})
#task.get_logger().report_scalar(title='score', series='score', value=score, iteration=8)

y_submit_pred = clf.predict_proba(x_submit).transpose()[1]
submit_pred = np.hstack((np.array(sub_ids),y_submit_pred))
submit_pred.shape = (2,20000)
df_submit_pred = pd.DataFrame(submit_pred.transpose(), columns = ['id','pred'])
df_submit_pred['id'] = df_submit_pred['id'].astype('int64')
df_submit_pred.to_csv('submission.csv', sep = ',', index=False)
    
task.get_logger().report_table("Result csv", series='submission.csv', table_plot=df_submit_pred)
#print(f'blend score: {score:.10f}')

#blend.loc[sub_ids].to_csv('blend.csv')