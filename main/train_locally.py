# -*- coding: utf-8 -*-#
import tensorflow as tf
import argparse
import model
from data_loader import loader
tf.enable_eager_execution()

flags = tf.app.flags
FLAGS = flags.FLAGS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", help="Input table GUID", default = './local_data/universal_transformer_sample.txt')
    #parser.add_argument("--task_index", help="Worker task index")
    # parser.add_argument("--worker_hosts", help="Worker host list")
    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping",default=20)
    parser.add_argument("--snapshot", type=int, help="Snapshot gap")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path", default = "./local_checkpoint/")
    parser.add_argument("--max_length", type=int, default=2400)
    parser.add_argument("--target_length", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--step", type=int, default=100001)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_vocabulary", type=int, default=178422)
    parser.add_argument("--feed_forward_in_dim", type=int, default=2048)
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--date_span", type=int, default=60)
    parser.add_argument("--enable_date_time_emb", type=int, default=1)
    parser.add_argument("--word_emb_dim", type=int, default=512)
    parser.add_argument("--max_query_count", type=int, default=300)
    return parser.parse_known_args()[0]

def main():
    # Parse arguments and print them
    args = parse_args()
    print("\nMain arguments:")
    for k, v in args.__dict__.items():
        print("{}={}".format(k, v))
    # tf.enable_eager_execution()
    transformer_model = model.TextTransformerNet(
        model_configs=model.TextTransformerNet.ModelConfigs(
            dropout_rate=args.dropout_rate,
            num_vocabulary = args.num_vocabulary,
            feed_forward_in_dim = args.feed_forward_in_dim,
            model_dim = args.model_dim,
            num_blocks = args.num_blocks,
            num_heads = args.num_heads,
            enable_date_time_emb = args.enable_date_time_emb,
            word_emb_dim=args.word_emb_dim,
            date_span=args.date_span
        ),
        train_configs=model.TrainConfigs(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            dropout_rate = args.dropout_rate
        ),
        predict_configs=None,
        run_configs=model.RunConfigs(
            log_every=10
        )
    )

    estimator = tf.estimator.Estimator(
        model_fn=transformer_model.model_fn,
        model_dir=args.checkpoint_dir,
        config=tf.estimator.RunConfig(
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)),
            save_checkpoints_steps=args.snapshot,
            keep_checkpoint_max=20
        )
    )

    print("Start training......")
    estimator.train(
        loader.LocalFileDataLoader(
            file_path=args.file_path,
            mode=tf.estimator.ModeKeys.TRAIN,
            hist_length=args.max_length,
            target_length=args.target_length
        ).input_fn,
        steps=args.max_steps
    )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

