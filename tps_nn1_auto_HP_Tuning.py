from clearml import Task, Logger
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, 
    RandomSearch, UniformIntegerParameterRange)


task = Task.init(project_name='TPS',
                 task_name='Auto HP Optimization',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)
args = {
    'template_task_id': "dc1b41f7916f4ff19492d0cf48625edb",
    'run_as_service': False,
}

an_optimizer = HyperParameterOptimizer(
    base_task_id=args['template_task_id'],
    hyper_parameters=[
    UniformIntegerParameterRange('General/EPOCH', min_value=5, max_value=15, step_size=1)
   ],
   objective_metric_title='log_loss',
   objective_metric_series='log_loss',
   objective_metric_sign='min',
)

an_optimizer.start()
top_exp = an_optimizer.get_top_experiments(top_k=3)
print([t.id for t in top_exp])
task.upload_artifact('top_exp', top_exp)
an_optimizer.wait()
an_optimizer.stop()