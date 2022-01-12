# -*- coding: utf-8 -*-#
import json
import os

def set_dist_env(dist_params):
    ps_hosts = (dist_params['ps_hosts'] or "").split(',')
    worker_hosts = dist_params['worker_hosts'].split(',')
    chief_hosts = worker_hosts[0:1]  # get first worker as chief
    worker_hosts = worker_hosts[1:]  # the rest as worker
    task_index = dist_params['task_index']
    job_name = dist_params['job_name']

    if job_name == "worker" and task_index == 0:
        job_name = "chief"
    # the others as worker
    if job_name == "worker" and task_index > 0:
        task_index -= 1
    tf_config = {
        'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
        'task': {'type': job_name, 'index': task_index}
    }
    print(json.dumps(tf_config))
    del os.environ['TF_CONFIG']

    slice_count = len(worker_hosts) + 1
    if job_name == "ps":
        return 1, 0
    else:
        slice_id = int(dist_params['task_index'])
        print("slice_id={}/{}".format(slice_id, slice_count))
        return slice_count, slice_id
