import os
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import log_loss

from clearml import Task
task = Task.init(project_name="TPS", task_name='blending8')

path = Path('.')

submission = pd.read_csv(path / 'sample_submission.csv', index_col='id')
labels = pd.read_csv(path / 'train_labels.csv', index_col='id')

# the ids of the submission rows (useful later)
sub_ids = submission.index

# the ids of the labeled rows (useful later)
gt_ids = labels.index 

# list of files in the submission folder
subs = sorted(os.listdir(path / 'submission_files'))

#s0 = pd.read_csv(path / 'submission_files' / subs[4998], index_col='id')
#score = log_loss(labels, s0.loc[gt_ids])
#print(subs[0], f'{score:.10f}')

#s1 = pd.read_csv(path / 'submission_files' / subs[4999], index_col='id')
#score = log_loss(labels, s1.loc[gt_ids])
#print(subs[1], f'{score:.10f}')

#blend = (s0 + s1) / 2

blend = pd.read_csv(path / 'submission_files' / subs[0], index_col='id')
for i in np.arange(3):
    tmp = pd.read_csv(path / 'submission_files' / subs[i], index_col='id')
    blend = blend + tmp
blend = blend / (3)

#blend = pd.read_csv(path / 'submission_files' / subs[0], index_col='id')
#blend['pred'] = 0.5

score = log_loss(labels, blend.loc[gt_ids])
#task.apload_artifact(name='Score', artifact_object={'score', score})
task.get_logger().report_scalar(title='score', series='score', value=score, iteration=8)
task.get_logger().report_table("Result csv", series='blend.csv', table_plot=blend.loc[sub_ids])
#print(f'blend score: {score:.10f}')

blend.loc[sub_ids].to_csv('blend.csv')