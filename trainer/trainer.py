# -*- coding: UTF-8 -*-
import tensorflow as tf


class Trainer:
    def __init__(self, model, loader, dumper, run_conf, mode):
        self._model = model
        self._loader = loader
        self._dumper = dumper
        self._run_conf = run_conf
        self._mode = mode

        self._estimator = self._init_estimator()

    def  _init_checkpoint_path(self):
        start_step = self._run_conf.step
        return None if start_step <= 0 else self._run_conf.model_path.checkpoint_path(start_step)

    def _extra_log_hooks(self):
        return [
            diagnose.GraphPrinterHook(), diagnose.ArgumentPrinterHook()
        ]

    def _init_estimator(self):
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True

        run_config = tf.estimator.RunConfig(
            session_config=session_config,
            save_checkpoints_steps=self._run_conf.save_every,
            keep_checkpoint_max=999,
            save_summary_steps=None,
            log_step_count_steps=self._run_conf.log_every
        )

        return tf.estimator.Estimator(
            model_fn=self._model.model_fn,
            model_dir=str(self._run_conf.model_path),
            config=run_config
        )

    def _train(self):
        self._estimator.train(
            self._loader.input_fn,
            steps=self._run_conf.max_steps,
            hooks=self._extra_log_hooks()
        )

    def _predict(self):
        self._dumper.output_fn(
            self._estimator.predict(
                self._loader.input_fn,
                checkpoint_path=self._init_checkpoint_path(),
                hooks=self._extra_log_hooks()
            )
        )

    def _evaluate(self):
        self._estimator.evaluate(
            self._loader.input_fn,
            checkpoint_path=self._init_checkpoint_path(),
            hooks=self._extra_log_hooks()
        )

    def run(self):
        {
            tf.estimator.ModeKeys.TRAIN: self._train,
            tf.estimator.ModeKeys.EVAL: self._evaluate,
            tf.estimator.ModeKeys.PREDICT: self._predict
        }[self._mode]()

    # Code below can be merged into a base class
    @classmethod
    def make_for_training(cls, model, loader, run_conf, **kwargs):
        return cls(model=model, loader=loader, dumper=None, run_conf=run_conf,
                   mode=env.TaskKeys.TRAIN, **kwargs)

    @classmethod
    def train(cls, model, loader, run_conf, **kwargs):
        cls.make_for_training(model, loader, run_conf, **kwargs).run()

    @classmethod
    def make_for_prediction(cls, model, loader, dumper, run_conf, **kwargs):
        return cls(model=model, loader=loader, dumper=dumper, run_conf=run_conf,
                   mode=env.TaskKeys.PREDICT, **kwargs)

    @classmethod
    def predict(cls, model, loader, dumper, run_conf, **kwargs):
        return cls.make_for_prediction(model, loader, dumper, run_conf, **kwargs).run()

    @classmethod
    def make_for_evaluation(cls, model, loader, run_conf, **kwargs):
        return cls(model=model, loader=loader, dumper=None, run_conf=run_conf,
                   mode=env.TaskKeys.EVAL, **kwargs)

    @classmethod
    def evaluate(cls, model, loader, run_conf, **kwargs):
        return cls.make_for_prediction(model, loader, run_conf, **kwargs).run()
