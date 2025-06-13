# my_custom_hooks.py
import numpy as np
import os
import os.path as osp
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.registry import HOOKS # 假设这是用于目标检测的，如果是其他库，HOOKS的来源可能不同

@HOOKS.register_module()
class SaveScoresHook(Hook):
    """
    自定义Hook，用于在每个训练、验证和测试周期结束后保存模型累积的分数。
    分数列表在每次保存后会从模型中清除。
    """
    def __init__(self,
                 out_dir=None,
                 save_bg_similarity=True,
                 save_original_scores=True):
        """
        Args:
            out_dir (str, optional): 保存 .npy 文件的目录。
                如果为 None，则使用 runner.work_dir。默认为 None。
            save_bg_similarity (bool, optional): 是否保存 background_similarity。默认为 True。
            save_original_scores (bool, optional): 是否保存 original_scores。默认为 True。
        """
        super().__init__()
        self.out_dir = out_dir
        self.save_bg_similarity = save_bg_similarity
        self.save_original_scores = save_original_scores

    def _get_model_and_clear_scores(self, runner: Runner, score_attribute_name: str) -> list:
        """
        从模型获取指定的分数列表，并随后清空该列表。
        """
        model_instance = runner.model
        if hasattr(model_instance, 'module'):
            actual_model = model_instance.module  # 处理 DDP/DP
        else:
            actual_model = model_instance

        scores_list = []
        if hasattr(actual_model, score_attribute_name):
            scores_list = getattr(actual_model, score_attribute_name, [])
            if not isinstance(scores_list, list): # 确保获取到的是列表
                runner.logger.warning(
                    f"在 SaveScoresHook 中：模型属性 '{score_attribute_name}' 不是一个列表，"
                    f"类型为 {type(scores_list)}。将尝试使用空列表。"
                )
                scores_list = []
            # 清空模型中的列表
            setattr(actual_model, score_attribute_name, [])
            runner.logger.info(
                f"在 SaveScoresHook 中：已获取并清空模型中的 '{score_attribute_name}' 列表。"
            )
        else:
            runner.logger.warning(
                f"在 SaveScoresHook 中：模型没有 '{score_attribute_name}' 属性。"
            )
        return scores_list

    def _save_scores_to_file(self, runner: Runner, scores_list: list, score_name: str, file_prefix: str):
        """
        辅助函数，用于拼接和保存单个分数列表到文件。
        """
        if not scores_list:
            runner.logger.info(f"在 SaveScoresHook 中：列表 '{score_name}' (前缀: {file_prefix}) 为空，跳过保存。")
            return

        try:
            all_scores_np = np.concatenate(scores_list, axis=0)
        except ValueError as e:
            runner.logger.error(
                f"在 SaveScoresHook 中：拼接 '{score_name}' 列表 (前缀: {file_prefix}) 时出错: {e}。"
                " 请检查列表中数组的形状。"
            )
            for i, arr in enumerate(scores_list):
                if isinstance(arr, np.ndarray):
                    runner.logger.info(f"  '{score_name}' 列表元素 {i} 的形状: {arr.shape}")
                else:
                    runner.logger.info(f"  '{score_name}' 列表元素 {i} 不是一个 NumPy 数组: {type(arr)}")
            return

        # 确定输出目录
        current_out_dir = self.out_dir if self.out_dir is not None else runner.work_dir
        
        # 确保目录存在
        if not osp.exists(current_out_dir):
            try:
                os.makedirs(current_out_dir, exist_ok=True)
                runner.logger.info(f"在 SaveScoresHook 中：已创建目录: {current_out_dir}")
            except Exception as e:
                runner.logger.error(f"在 SaveScoresHook 中：创建目录 {current_out_dir} 失败: {e}")
                return

        filename_score_part = score_name.lower().replace(' ', '_')
        save_path = osp.join(current_out_dir, f"{file_prefix}_{filename_score_part}.npy")
        
        try:
            np.save(save_path, all_scores_np)
            runner.logger.info(
                f"在 SaveScoresHook 中：已保存 '{score_name}' (前缀: {file_prefix})，"
                f"形状为: {all_scores_np.shape}，路径为: {save_path}"
            )
        except Exception as e:
            runner.logger.error(f"在 SaveScoresHook 中：保存 '{score_name}' 到 {save_path} (前缀: {file_prefix}) 失败: {e}")

    def _process_and_save_scores(self, runner: Runner, stage_prefix: str):
        """
        统一处理获取、保存和清空分数的逻辑。
        """
        # runner.epoch 在训练时是当前epoch (0-indexed)，在验证和测试时可能需要特别注意其含义
        # MMDetection 的 ValLoop 和 TestLoop 也会传递 runner，其中 epoch 通常是训练时的 epoch
        epoch_num = runner.epoch + 1 # 转为 1-indexed

        file_prefix_with_epoch = f"{stage_prefix}_epoch_{epoch_num}"
        runner.logger.info(
            f"SaveScoresHook: {stage_prefix} 钩子被调用 (Epoch: {epoch_num})。"
            f" 文件名前缀将为: {file_prefix_with_epoch}"
        )

        if self.save_bg_similarity:
            bg_scores = self._get_model_and_clear_scores(runner, 'accumulated_background_similarity')
            self._save_scores_to_file(runner, bg_scores, "background_similarity", file_prefix_with_epoch)

        if self.save_original_scores:
            orig_scores = self._get_model_and_clear_scores(runner, 'accumulated_original_scores')
            self._save_scores_to_file(runner, orig_scores, "original_scores", file_prefix_with_epoch)

    def after_train_epoch(self, runner: Runner):
        """在每个训练周期结束后被调用。"""
        self._process_and_save_scores(runner, "train")

    def after_val_epoch(self, runner: Runner):
        """在每个验证周期结束后被调用。"""
        # 验证前，模型中的累积列表应该由验证数据填充
        # （假设 MMEngine 的 ValLoop 会调用模型进行前向传播）
        self._process_and_save_scores(runner, "val")

    def after_test(self, runner: Runner):
        """
        在测试结束后被调用。
        注意: MMDetection 的标准 TestLoop 通常只运行一个 "周期"。
        如果测试有多个周期（不常见），此钩子会每个周期执行。
        """
        # 测试前，模型中的累积列表应该由测试数据填充
        self._process_and_save_scores(runner, "test")

