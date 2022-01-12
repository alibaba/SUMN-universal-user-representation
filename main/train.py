# -*- coding: utf-8 -*-#
import tensorflow as tf
import argparse
import model
from data_loader import loader
# from util import env
from util import helper
# from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str)
    parser.add_argument("--buckets", type=str)
    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping")
    parser.add_argument("--snapshot", type=int, help="Number of iterations to dump model")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--target_length", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_vocabulary", type=int, default=178422)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--word_emb_dim", type=int, default=256)
    parser.add_argument("--mlp_layers", type=str, default="512")
    parser.add_argument("--pooling", type=str, default="max")
    parser.add_argument("--distributed", type=int, default=0)
    parser.add_argument("--weighted_target", type=int, default=0)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=0)

    parser.add_argument("--max_query_per_week", type=int, default=900)
    parser.add_argument("--max_words_per_query", type=int, default=24)
    return parser.parse_known_args()[0]


def main():
    # Parse arguments and print them
    args = parse_args()
    print("\nMain arguments:")
    for k, v in args.__dict__.items():
        print("{}={}".format(k, v))

    # Check if the model has already exisited
    model_save_dir = args.buckets + args.checkpoint_dir
    if tf.gfile.Exists(model_save_dir + "/checkpoint"):
        raise ValueError("Model %s has already existed, please delete them and retry" % model_save_dir)

    helper.dump_args(model_save_dir, args)

    init_checkpoint = args.init_checkpoint
    if init_checkpoint:
        pass # add your own codes

    transformer_model = model.TextTransformerNet(
        model_configs=model.TextTransformerNet.ModelConfigs(
            dropout_rate=args.dropout_rate,
            num_vocabulary = args.num_vocabulary+2,
            mlp_layers=args.mlp_layers,
            model_dim=args.model_dim,
            max_query_per_week=args.max_query_per_week,
            max_words_per_query=args.max_words_per_query,
            word_emb_dim=args.word_emb_dim,
            pooling=args.pooling,
            train_batch_size=args.batch_size
        ),
        train_configs=model.TrainConfigs(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            init_checkpoint=init_checkpoint,
            max_grad_norm=args.max_grad_norm,
            weighted_target=args.weighted_target
        ),
        predict_configs=None,
        run_configs=model.RunConfigs(
            log_every=50
        )
    )
    # checkpoint_path = None
    # if args.step > 0:
    #     checkpoint_path = model_save_dir + "/model.ckpt-{}".format(args.step)
    # warm_start_settings = tf.estimator.WarmStartSettings(checkpoint_path,
    #                                                      vars_to_warm_start='(.*Embedding|Conv-[1-4]|MlpLayer-1)')

    if not args.distributed:
        distribution = None
    else:
        raise ValueError("Not implemented! Add your own codes here.")

    estimator = tf.estimator.Estimator(
        model_fn=transformer_model.model_fn,
        model_dir=model_save_dir,
        config=tf.estimator.RunConfig(
            session_config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=False),
                allow_soft_placement=True
            ),
            save_checkpoints_steps=args.snapshot,
            keep_checkpoint_max=20,
            train_distribute=distribution
        )
    )
    print("Start training......")
    estimator.train(
        loader.OdpsDataLoader(
            table_name=args.tables,
            mode=tf.estimator.ModeKeys.TRAIN,
            max_query_per_week=args.max_query_per_week,
            max_words_per_query=args.max_words_per_query,
            target_length=args.target_length,
            batch_size=args.batch_size,
            weighted_target=args.weighted_target
        ).input_fn,
        steps=args.max_steps
    )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

