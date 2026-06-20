import os
import glob
import torch
import traceback
from lib.train.admin import multigpu
from torch.utils.data.distributed import DistributedSampler


class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading checkpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)
        self.settings = settings

    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            '''2021.1.4 New function: specify checkpoint dir'''
            if self.settings.save_dir is None:
                self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            else:
                self._checkpoint_dir = os.path.join(self.settings.save_dir, 'checkpoints')
            print("checkpoints will be saved to %s" % self._checkpoint_dir)

            if self.settings.local_rank in [-1, 0]:
                if not os.path.exists(self._checkpoint_dir):
                    print("Training with multiple GPUs. checkpoints directory doesn't exist. "
                          "Create checkpoints directory")
                    os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

    def train(self, max_epochs, load_latest=False, fail_safe=True, load_previous_ckpt=False, distill=False):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.
        """

        # ── best-model tracking (YOLO-style) ──────────────────────────
        self._best_metric_val = float("inf")  # lower is better
        self._best_metric_name = getattr(self.settings, "save_best_metric", "Loss/total")
        self._save_best_enabled = getattr(self.settings, "save_best", True)
        # Find the validation loader name once
        self._val_loader_name = None
        for ld in self.loaders:
            if not ld.training:
                self._val_loader_name = ld.name
                break
        # ───────────────────────────────────────────────────────────────

        epoch = -1
        num_tries = 1
        for i in range(num_tries):
            try:
                if load_latest:
                    self.load_checkpoint()
                if load_previous_ckpt:
                    directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path_prv)
                    self.load_state_dict(directory)
                if distill:
                    directory_teacher = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path_teacher)
                    self.load_state_dict(directory_teacher, distill=True)
                for epoch in range(self.epoch+1, max_epochs+1):
                    self.epoch = epoch

                    self.train_epoch()

                    if self.lr_scheduler is not None:
                        if self.settings.scheduler_type != 'cosine':
                            self.lr_scheduler.step()
                        else:
                            self.lr_scheduler.step(epoch - 1)

                    # only save the last 10 checkpoints
                    save_every_epoch = getattr(self.settings, "save_every_epoch", False)
                    save_epochs = getattr(self.settings, "save_epochs", [79, 159, 239])
                    if epoch > (max_epochs - 1) or save_every_epoch or epoch % 40 == 0 or epoch in save_epochs or epoch > (max_epochs - 5):
                    # if epoch > (max_epochs - 10) or save_every_epoch or epoch % 100 == 0:
                        if self._checkpoint_dir:
                            if self.settings.local_rank in [-1, 0]:
                                self.save_checkpoint()
            except:
                print('Training crashed at epoch {}'.format(epoch))
                if fail_safe:
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')

    def train_epoch(self):
        raise NotImplementedError

    # ── YOLO-style best-model tracking ────────────────────────────────

    def _try_save_best(self):
        """
        After each epoch, check whether the validation metric improved.
        If so, overwrite ``{net_type}_best.pth.tar`` — exactly like YOLO's best.pt.
        """
        if not self._save_best_enabled or self._checkpoint_dir is None:
            return
        if self.settings.local_rank not in [-1, 0]:
            return
        if self._val_loader_name is None:
            return

        val_stats = self.stats.get(self._val_loader_name)
        if val_stats is None:
            return  # validation didn't run this epoch

        metric_meter = val_stats.get(self._best_metric_name)
        if metric_meter is None:
            return
        if not hasattr(metric_meter, 'avg'):
            return

        current = metric_meter.avg
        if current < self._best_metric_val:
            self._best_metric_val = current
            self.save_checkpoint(tag="best")
            print(f"🏆 New best model saved!  {self._best_metric_name} = {current:.5f}")

    # ───────────────────────────────────────────────────────────────────

    def save_checkpoint(self, tag=None):
        """Saves a checkpoint of the network and other variables.

        Args:
            tag: If given (e.g. "best"), saves as ``{net_type}_{tag}.pth.tar``
                 instead of ``{net_type}_ep{epoch:04d}.pth.tar``.
        """

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'settings': self.settings,
        }
        # Carry best-metric value when saving best checkpoint
        if tag == "best" and hasattr(self, '_best_metric_val'):
            state['best_metric'] = self._best_metric_val
            state['best_metric_name'] = getattr(self, '_best_metric_name', 'Loss/total')

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        print(directory)
        if not os.path.exists(directory):
            print("directory doesn't exist. creating...")
            os.makedirs(directory)

        # Choose filename: tag-based overwrites, epoch-based accumulates
        if tag is not None:
            file_path = '{}/{}_{}.pth.tar'.format(directory, net_type, tag)
        else:
            file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)

        # Atomically save via tmp file. Use os.replace to overwrite existing file on Windows.
        tmp_file_path = file_path + '.tmp'
        torch.save(state, tmp_file_path)
        try:
            os.replace(tmp_file_path, file_path)
        except Exception:
            # Fallback: remove existing file and rename (best-effort)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            os.rename(tmp_file_path, file_path)

    def load_checkpoint(self, checkpoint = None, fields = None, ignore_fields = None, load_constructor = False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._checkpoint_dir,
                                                                             self.settings.project_path, net_type)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_dir, self.settings.project_path,
                                                                 net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

            # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net.load_state_dict(checkpoint_dict[key])
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoint_dict[key])
            else:
                setattr(self, key, checkpoint_dict[key])

        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch
        # 2021.1.10 Update the epoch in data_samplers
            for loader in self.loaders:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
        return True

    def load_state_dict(self, checkpoint=None, distill=False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        if distill:
            net = self.actor.net_teacher.module if multigpu.is_multi_gpu(self.actor.net_teacher) \
                else self.actor.net_teacher
        else:
            net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        net_type = type(net).__name__

        if isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        print("Loading pretrained model from ", checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        missing_k, unexpected_k = net.load_state_dict(checkpoint_dict["net"], strict=False)
        print("previous checkpoint is loaded.")
        print("missing keys: ", missing_k)
        print("unexpected keys:", unexpected_k)

        return True
