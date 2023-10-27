import wandb


class WandBLogger:
    def __init__(self, project, config, name, group, active=False):
        self.project = project
        self.config = config
        self.name = name
        self.group = group
        self.active = active
        if self.active:
            self._build()

    def _build(self):
        wandb.init(
            project=self.project,
            config=self.config,
            name=self.name,
            group=self.group
        )

    def log(self, log_dict, step=None, commit=None):
        if self.active:
            wandb.log(log_dict, step=step, commit=commit)
        else:
            pass


class SweepLogger:
    def __init__(self, project, name, group):
        self.project = project
        self.name = name
        self.group = group
        self._build()

    def _build(self):
        wandb.init(
            project=self.project,
            name=self.name,
            group=self.group
        )

    @staticmethod
    def log(log_dict, step=None, commit=None):
        wandb.log(log_dict, step=step, commit=commit)
