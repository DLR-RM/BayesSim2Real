import logging
import os
from collections import OrderedDict
import weakref
from detectron2.engine import DefaultTrainer, hooks, create_ddp_model
from detectron2.engine.hooks import EvalHook
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer, TrainerBase
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import EventStorage
from fvcore.nn.precise_bn import get_bn_modules
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)

class EvalHook_after_train(EvalHook):
    def after_train(self):
        # This condition is to prevent the eval from running after a failed training
        # if self.trainer.iter + 1 >= self.trainer.max_iter:
        #     self._do_eval()
        
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        # del self._func
        additional_state = {"iteration":  self.trainer.iter} 
        self.trainer.checkpointer.save(
            "{}_{:07d}".format("model", self.trainer.iter), **additional_state
        )

class AL_Trainer(DefaultTrainer):
    def __init__(self, cfg, test_set="", val_set=""):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg)
        self.test_set = test_set
        self.val_set = val_set
        self.best_AP = 0.0
        self.finish_trn_flag = 0

    def train(self):
        """inheritate from super().train(), add early stoping function
        """
        start_iter = self.start_iter 
        max_iter = self.max_iter
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                    if (self.iter % (self.cfg.TEST.EVAL_PERIOD+1)) == 0:
                        if self.finish_trn_flag:
                            print("Inside train(), finish_trn_flag is ", self.finish_trn_flag)
                            # self.after_train()
                            return
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
        
        # if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
        #     assert hasattr(
        #         self, "_last_eval_results"
        #     ), "No evaluation results obtained during training!"
        #     verify_results(self.cfg, self._last_eval_results)
        #     return self._last_eval_results

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        logger = logging.getLogger("detectron2")
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            if self.val_set == "" or self.val_set is None:
                self._last_eval_results = self.test(self.cfg, self.model, test_set=self.test_set)
            else:
                self._last_eval_results = self.test(self.cfg, self.model, test_set=self.val_set)
                # print(self._last_eval_results)
                cur_AP = self._last_eval_results['bbox']['AP']
                logger.info("cur_AP is {:.3f}, best_AP is {:.3f}.".format(cur_AP, self.best_AP))
                
                if cur_AP < self.best_AP:
                    logger.info("current iter is {}".format(self.iter))
                    logger.info("cur_AP {:.3f} is lower than best_AP {:.3f}, finishing training.".format(cur_AP, self.best_AP))
                    logger.info("finish_trn_flag is set to 1")
                    self.finish_trn_flag = 1
                else:
                    self.best_AP = cur_AP
                    logger.info("best_AP is set to {}".format(cur_AP))
                    # raise Exception("cur_AP {:.3f} is lower than best_AP {:.3f}, then save the current weights and evaluate on test set.".format(cur_AP, self.best_AP))
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(EvalHook_after_train(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
        
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Builds evaluators for post-training mAP report.
        Args:
            cfg(CfgNode): a detectron2 CfgNode
            dataset_name(str): registered dataset name

        Returns:
            detectron2 DatasetEvaluators object
        """
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)

    @classmethod
    # def test(cls, cfg, model, evaluators=None, test_set="", val_set="",val_flag=False):
    def test(cls, cfg, model, evaluators=None, test_set=""):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
            val_flag: flag to use val set or test set.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        # if isinstance(evaluators, DatasetEvaluator):
        #     evaluators = [evaluators]
        # if evaluators is not None:
        #     assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
        #         len(cfg.DATASETS.TEST), len(evaluators)
        #     )

        results = OrderedDict()
        # for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
        dataset_name = test_set
        data_loader = cls.build_test_loader(cfg, dataset_name)
        # When evaluators are passed in as arguments,
        # implicitly assume that evaluators can be created before data_loader.
        # if evaluators is not None:
        #     evaluator = evaluators[idx]
        # else:
        try:
            evaluator = cls.build_evaluator(cfg, dataset_name)
        except NotImplementedError:
            logger.warn(
                "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                "or implement its `build_evaluator` method."
            )
            results[dataset_name] = {}
            # continue
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            assert isinstance(
                results_i, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results_i
            )
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results