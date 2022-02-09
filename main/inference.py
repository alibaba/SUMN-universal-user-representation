# -*- coding: utf-8 -*-#
import tensorflow as tf
from tensorflow.python.platform import gfile
import argparse
import model
from data_loader import loader
from data_dumper import dumper
from util import env
from util import helper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--outputs", type=str, help="Destination of the table ")
    parser.add_argument("--buckets", type=str, help="Worker task index")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--separator", type=str, default=" ")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--hop", type=int, default=0)

    # Arguments for distributed inference
    parser.add_argument("--task_index", type=int, help="Task index")
    parser.add_argument("--ps_hosts", type=str, help="")
    parser.add_argument("--worker_hosts", type=str, help="")
    parser.add_argument("--job_name", type=str)

    return parser.parse_known_args()[0]


def _do_prediction(result_iter, writer, args, model_args):
    import time
    print("Start inference......")
    t_start = t_batch_start = time.time()
    report_gap = 10000

    indices = [0, 1, 2]

    for i, prediction in enumerate(result_iter, 1):
        record = [
            prediction["id"],
            prediction["feature"],
            int(prediction["length"])
        ]

        writer.write(record, indices)
        if i % report_gap == 0:
            t_now = time.time()
            print("[{}]Processed {} samples, {} records/s, cost {} s totally, {} records/s averagely".format(
                args.task_index,
                i,
                report_gap / (t_now - t_batch_start),
                (t_now - t_start),
                i / (t_now - t_start)
            ))
            t_batch_start = t_now

    writer.close()


def main():
    # Parse arguments and print them
    args = parse_args()
    print("\nMain arguments:")
    for k, v in args.__dict__.items():
        print("{}={}".format(k, v))

    # Setup distributed inference
    dist_params = {
        "task_index": args.task_index,
        "ps_hosts": args.ps_hosts,
        "worker_hosts": args.worker_hosts,
        "job_name": args.job_name
    }
    slice_count, slice_id = env.set_dist_env(dist_params)
    # Load model arguments
    model_save_dir = args.buckets + args.checkpoint_dir
    model_args = helper.load_args(model_save_dir)

    transformer_model = model.TextTransformerNet(
        model_configs=model.TextTransformerNet.ModelConfigs(
            dropout_rate=model_args.dropout_rate,
            num_vocabulary = model_args.num_vocabulary+2,
            model_dim = model_args.model_dim,
            word_emb_dim=model_args.word_emb_dim,
            pooling=model_args.pooling,
            train_batch_size=0,
            mlp_layers=model_args.mlp_layers,
            max_query_per_week=model_args.max_query_per_week,
            max_words_per_query=model_args.max_words_per_query
        ),
        train_configs=model.TrainConfigs( # train configs won't make any effects when inference
            learning_rate=model_args.learning_rate,
            batch_size=model_args.batch_size,
            init_checkpoint=None,
            max_grad_norm=None,
            weighted_target=False
        ),
        predict_configs=model.PredictConfigs(
            separator=args.separator,
            hop=args.hop
        ),
        run_configs=model.RunConfigs(
            log_every=200
        )
    )

    estimator = tf.estimator.Estimator(
        model_fn=transformer_model.model_fn,
        model_dir=model_save_dir,
        config=tf.estimator.RunConfig(
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)),
            save_checkpoints_steps=model_args.max_steps,
            keep_checkpoint_max=1
        )
    )

    checkpoint_path = None
    if args.step > 0:
        checkpoint_path = model_save_dir + "model.ckpt-{}".format(args.step)
        if not gfile.Exists(checkpoint_path + ".index"):
            checkpoint_path = model_save_dir + "model.ckpt-{}".format(args.step+1)

    result_iter = estimator.predict(
        loader.OdpsDataLoader(
            table_name=args.tables,
            mode=tf.estimator.ModeKeys.PREDICT,
            max_query_per_week=model_args.max_query_per_week,
            max_words_per_query=model_args.max_words_per_query,
            batch_size=32,
            target_length=0,
            slice_id=slice_id,
            slice_count=slice_count,
            shuffle=0,
            repeat=1
        ).input_fn,
        checkpoint_path=checkpoint_path
    )

    odps_writer = dumper.get_odps_writer(
        args.outputs,
        slice_id=slice_id
    )
    _do_prediction(result_iter, odps_writer, args, model_args)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

