import wandb
import os
import pandas as pd
from pointcept.engines.hooks import CheckpointLoader, CheckpointSaver, is_main_process, HOOKS

@HOOKS.register_module()
class CheckpointLoaderAllowMismatch(CheckpointLoader):
    def prep_loaded_state_dict(self, checkpoint):
        loaded_state_dict = super().prep_loaded_state_dict(checkpoint)
        current_model_dict = self.trainer.model.state_dict()
        # allows size mismatch
        removed_keys = {
            k
            for k, v in loaded_state_dict.items()
            if k in current_model_dict and v.size() != current_model_dict[k].size()
        }

        for key in removed_keys:
            loaded_state_dict.pop(key)

        self.trainer.logger.warning(
            f"Removed keys due to size mismatch: {removed_keys}"
        )

        return loaded_state_dict

@HOOKS.register_module()
class CheckpointSaverWandb(CheckpointSaver):
    def __init__(self, save_freq=None):
        self.save_freq = save_freq  # None or int, None indicate only save model last

        # best model stat-time, dummy value to start with
        self.last_model_best_stat = os.stat(__file__)

    def delete_oldest_artifact_version(self, name_part):
        logged_artifacts = wandb.Api().run(wandb.run._get_path()).logged_artifacts()

        interest_artifacts = [a for a in logged_artifacts if name_part in a.name]
        dates = pd.to_datetime([a.created_at for a in interest_artifacts])

        # Don't delete my only artifact!
        if len(dates) > 1:
            idx = dates.argmin()
            interest_artifacts[idx].delete()

    def after_epoch(self):
        super().after_epoch()
        if is_main_process():
            # our super() saved it here
            filename = os.path.join(
                self.trainer.cfg.save_path, "model", "model_last.pth"
            )

            # Log to wandb and remove old versions for storage
            wandb.log_model(filename, name=f"{wandb.run.id}-model_last")
            self.delete_oldest_artifact_version(name_part="model_last")

            # Check if a new best model exists
            model_best = os.path.join(
                self.trainer.cfg.save_path, "model", "model_best.pth"
            )

            # use os.stat to check modified time because i don't wanna actually check the file contents
            if os.stat(model_best).st_atime != self.last_model_best_stat.st_atime:
                wandb.log_model(model_best, name=f"{wandb.run.id}-model_best")
                self.delete_oldest_artifact_version(name_part="model_best")

            self.last_model_best_stat = os.stat(model_best)

