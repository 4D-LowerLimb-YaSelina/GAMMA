import time
import shutil
import traceback
import random
from pathlib import Path

import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from src.infra.registry import (
    DATASET_REGISTRY,
    DATALOADER_REGISTRY,
    NETWORK_REGISTRY,
    MODULE_REGISTRY,
    LOSS_REGISTRY,
    METRIC_REGISTRY,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    SCRIPT_REGISTRY,
)
from src.utils.tensor import to_device
from src.utils.misc import format_time


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --- training scripts ---
@SCRIPT_REGISTRY.register()
class TrainScript:
    def __init__(self, opt):
        self.opt = opt

        # device select
        if self.opt.device_select == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.opt.device_select

        print(f"running on {self.device}...")

        # current loss & bench
        self.current_loss = torch.inf
        self.current_bench = torch.inf

    def run(self):
        self.setup_dir()
        self.load_data()
        self.train_prep()
        self.train_loop()

    def dry_run(self, step=10):
        """
        Dry run the train loop for verification
        """
        self.setup_dir()

        try:
            self.load_data()
            self.train_prep()

            # total step calculation
            epochs = self.opt.train.epochs
            start_time = time.time()
            self.global_step = 0
            self.total_step = epochs * len(self.train_dataloader)
            batch_num = len(self.train_dataloader)
            self.best_bench = torch.inf

            if type(self.opt.train.test_interval) is float:
                self.opt.train.test_interval = (
                    int(self.total_step * self.opt.train.test_interval) + 1
                )
                print(f"will test every {self.opt.train.test_interval} steps")

            # dry run the 1st few steps
            epoch = 0

            for batch, data in enumerate(self.train_dataloader):
                self.global_step += 1
                self.print_training_progress(
                    epoch,
                    epochs,
                    batch,
                    batch_num,
                    self.global_step,
                    self.total_step,
                    start_time,
                )

                data = to_device(data, self.device)
                self._train_step(data)

                if self.global_step % self.opt.train.test_interval == 0:
                    self._test_step()

                if self.opt.train.checkpoint_interval and (
                    self.global_step % self.opt.train.checkpoint_interval == 0
                ):
                    self._checkpoint_step()

                for name, scheduler in self.opt.train.scheduler.items():
                    if scheduler.step_per_batch:
                        self.scheduler_dict[name].step()
                        self.writer.add_scalar(
                            f"lr/{name}",
                            self.scheduler_dict[name].get_last_lr()[0],
                            self.global_step,
                        )

                if self.global_step >= step:
                    break

        except Exception as e:
            print("-" * 100)
            traceback.print_exc()
            print("-" * 100)

        # remove the experiment path
        shutil.rmtree(self.opt.path)
        print(f"deleted experiment folder {self.opt.path}")

    def setup_dir(self):
        Path(self.opt.path).mkdir(parents=True)
        print(f"created experiment folder {self.opt.path}")

        shutil.copy(self.opt.config, Path(self.opt.path) / "config.yaml")
        print(f"copied configurations to {Path(self.opt.path) / 'config.yaml'}")

        self.writer = SummaryWriter(Path(self.opt.path) / "log")
        (Path(self.opt.path) / "checkpoint").mkdir(parents=True)

    def load_data(self):
        g_train = torch.Generator()
        g_train.manual_seed(self.opt.random_seed)
        self.train_dataset = DATASET_REGISTRY[self.opt.train.dataset.train.name](
            **self.opt.train.dataset.train.args,
        )
        self.train_dataloader = DATALOADER_REGISTRY[
            self.opt.train.dataset.train.dataloader.name
        ](
            self.train_dataset,
            generator=g_train,
            **self.opt.train.dataset.train.dataloader.args,
        )
        print(f"load training dataset with {len(self.train_dataset)} samples")

        g_test = torch.Generator()
        g_test.manual_seed(self.opt.random_seed)
        self.test_dataset = DATASET_REGISTRY[self.opt.train.dataset.test.name](
            **self.opt.train.dataset.test.args,
        )
        self.test_dataloader = DATALOADER_REGISTRY[
            self.opt.train.dataset.test.dataloader.name
        ](
            self.test_dataset,
            generator=g_test,
            **self.opt.train.dataset.test.dataloader.args,
        )
        print(f"load testing dataset with {len(self.test_dataset)} samples")

    def train_prep(self):
        # init/load model
        self.network = NETWORK_REGISTRY[self.opt.network.name](
            self.opt, **self.opt.network.args
        ).to(self.device)
        self.network_dict = dict(self.network.named_children())
        print(f"initialized model {self.opt.network.name}")

        if "load_from" in self.opt.network:
            pth_path = self.opt.network.load_from
            self.network.load_state_dict(torch.load(pth_path, map_location=self.device))
            print(f"loaded model weights from {pth_path}")

        # init optimizer dict
        self.optimizer_dict = {}

        for name, optimizer in self.opt.train.optimizer.items():
            if name == "all":
                assert len(self.opt.train.optimizer.keys()) == 1, (
                    'optimizer with name "all" should be the only one in the optimizer dict'
                )
                self.optimizer_dict[name] = OPTIMIZER_REGISTRY[optimizer["name"]](
                    **optimizer["args"], params=self.network.parameters()
                )

            else:
                self.optimizer_dict[name] = OPTIMIZER_REGISTRY[optimizer["name"]](
                    **optimizer["args"], params=self.network_dict[name].parameters()
                )

            if "load_from" in optimizer:
                pth_path = optimizer.load_from
                self.optimizer_dict[name].load_state_dict(
                    torch.load(pth_path, map_location=self.device)
                )
                print(f"loaded optimizer weights from {pth_path}")

        # init scheduler dict
        if "scheduler" in self.opt.train:
            self.scheduler_dict = {}

            for name, scheduler in self.opt.train.scheduler.items():
                if name == "all":
                    assert len(self.opt.train.scheduler.keys()) == 1, (
                        'scheduler with name "all" should be the only one in the scheduler dict'
                    )
                    self.scheduler_dict[name] = SCHEDULER_REGISTRY[scheduler["name"]](
                        **scheduler["args"], optimizer=self.optimizer_dict["all"]
                    )

                else:
                    self.scheduler_dict[name] = SCHEDULER_REGISTRY[scheduler["name"]](
                        **scheduler["args"], optimizer=self.optimizer_dict[name]
                    )

                if "load_from" in scheduler:
                    pth_path = scheduler.load_from
                    self.scheduler_dict[name].load_state_dict(
                        torch.load(pth_path, map_location=self.device)
                    )
                    print(f"loaded scheduler weights from {pth_path}")

        # init loss dict
        self.loss_dict = {}
        self.loss_weight = {}

        for name, loss in self.opt.train.loss.items():
            self.loss_dict[name] = LOSS_REGISTRY[loss["name"]](**loss["args"]).to(
                self.device
            )
            self.loss_weight[name] = loss["weight"]

        # init metric dict
        self.metric_dict = {}

        for name, metric in self.opt.train.metric.items():
            self.metric_dict[name] = METRIC_REGISTRY[metric["name"]](
                **metric["args"]
            ).to(self.device)

    def print_training_progress(
        self, epoch, epoch_num, batch, batch_num, step, step_total, start_time
    ):
        progress_rate = step / step_total
        elapsed_time = time.time() - start_time
        step_throughput = (step + 1) / elapsed_time
        eta_time = (
            elapsed_time / progress_rate - elapsed_time if progress_rate > 0 else 0
        )
        print(
            f"training | epoch {epoch + 1}/{epoch_num} batch {batch + 1}/{batch_num} step {self.global_step}/{step_total} ({progress_rate * 100:.2f}%) | loss {self.current_loss} | throughput {step_throughput:.2f} batch/s elapsed {format_time(elapsed_time)} | eta {format_time(eta_time)} @{self.opt.exp_name}"
        )

    def print_testing_progress(self, batch, batch_num, start_time):
        test_epoch_num = self.total_step // self.opt.train.test_interval

        if self.global_step == self.total_step:
            test_epoch = "-"
        else:
            test_epoch = self.global_step // self.opt.train.test_interval

        progress_rate = batch / batch_num
        elapsed_time = time.time() - start_time
        batch_throughput = (batch + 1) / elapsed_time
        eta_time = (
            elapsed_time / progress_rate - elapsed_time if progress_rate > 0 else 0
        )
        print(
            f"testing | epoch {test_epoch}/{test_epoch_num} batch {batch + 1}/{batch_num} ({progress_rate * 100:.2f}%) | bench {self.current_bench} | throughput {batch_throughput:.2f} batch/s elapsed {format_time(elapsed_time)} | eta {format_time(eta_time)} @{self.opt.exp_name}"
        )

    def train_loop(self):
        epochs = self.opt.train.epochs
        start_time = time.time()
        self.global_step = 0
        self.total_step = epochs * len(self.train_dataloader)
        batch_num = len(self.train_dataloader)
        self.best_bench = torch.inf

        if type(self.opt.train.test_interval) is float:
            self.opt.train.test_interval = (
                int(self.total_step * self.opt.train.test_interval) + 1
            )
            print(f"will test every {self.opt.train.test_interval} steps")

        for epoch in range(epochs):
            for batch, data in enumerate(self.train_dataloader):
                self.global_step += 1
                self.print_training_progress(
                    epoch,
                    epochs,
                    batch,
                    batch_num,
                    self.global_step,
                    self.total_step,
                    start_time,
                )

                data = to_device(data, self.device)
                self._train_step(data)

                if self.global_step % self.opt.train.test_interval == 0:
                    self._test_step()

                if self.opt.train.checkpoint_interval and (
                    self.global_step % self.opt.train.checkpoint_interval == 0
                ):
                    self._checkpoint_step()

                # step scheduler per batch if specified
                for name, scheduler in self.opt.train.scheduler.items():
                    if scheduler.step_per_batch:
                        self.scheduler_dict[name].step()
                        self.writer.add_scalar(
                            f"lr/{name}",
                            self.scheduler_dict[name].get_last_lr()[0],
                            self.global_step,
                        )

            # step scheduler per epoch if specified
            for name, scheduler in self.opt.train.scheduler.items():
                if scheduler.step_per_epoch:
                    self.scheduler_dict[name].step()
                    self.writer.add_scalar(
                        f"lr/{name}",
                        self.scheduler_dict[name].get_last_lr()[0],
                        self.global_step,
                    )

        # final test & checkpoint
        print("running final checkpoint & testing...")
        self._checkpoint_step()
        self._test_step()
        self.writer.flush()

    def _train_step(self, data):
        self.network.train()

        # forward pass & loss calculation
        infer = self.network(data)
        loss_total = 0.0

        for name, loss in self.loss_dict.items():
            loss.train()
            loss_val = loss(infer, data)
            loss_total += loss_val * self.loss_weight[name]
            self.writer.add_scalar(
                f"loss/train/{name}",
                loss_val.item() / self.train_dataloader.batch_size,
                self.global_step,
            )

        self.current_loss = loss_total.item() / self.train_dataloader.batch_size
        self.writer.add_scalar("loss/train/total", self.current_loss, self.global_step)

        # loss backward
        for optimizer in self.optimizer_dict.values():
            optimizer.zero_grad()

        loss_total.backward()

        if "gradient_clipping" in self.opt.train.keys():
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.opt.train.gradient_clipping.max_norm,
            )

        for optimizer in self.optimizer_dict.values():
            optimizer.step()

    def _test_step(self):
        start_time = time.time()
        batch_num = len(self.test_dataloader)

        # turn on eval mode and start feed
        self.network.eval()

        for name, loss in self.loss_dict.items():
            loss.eval()
            loss.start_test_feed(self, name, log_tag="test")

        for name, metric in self.metric_dict.items():
            metric.eval()
            metric.start_test_feed(self, name, log_tag="test")

        # turn off autograd & run the testing
        with torch.no_grad():
            for batch, data in enumerate(self.test_dataloader):
                self.print_testing_progress(batch, batch_num, start_time)

                # forward pass
                data = to_device(data, self.device)
                infer = self.network(data)

                # loss calculation
                for name, loss in self.loss_dict.items():
                    val = loss.test_feed(infer, data)

                    if name == self.opt.train.save_best:
                        self.current_bench = val

                # metric calculation
                for name, metric in self.metric_dict.items():
                    val = metric.test_feed(infer, data)

                    if name == self.opt.train.save_best:
                        self.current_bench = val

        # end feed
        for name, loss in self.loss_dict.items():
            loss.end_test_feed()
            print(f"loss {name}: {loss.loss_avg}")

        for name, metric in self.metric_dict.items():
            metric.end_test_feed()
            print(f"metric {name}: {metric.metric_avg}")

        # save current model if it's the best so far
        bench_name = self.opt.train.save_best

        if (
            bench_name in self.loss_dict
            and self.best_bench > self.loss_dict[bench_name].loss_avg
        ):
            self.best_bench = self.loss_dict[bench_name].loss_avg
            self._checkpoint_step(mark="best")
            self.writer.add_scalar(
                f"loss/best-{bench_name}", self.best_bench, self.global_step
            )

        if (
            bench_name in self.metric_dict
            and self.best_bench > self.metric_dict[bench_name].metric_avg
        ):
            self.best_bench = self.metric_dict[bench_name].metric_avg
            self._checkpoint_step(mark="best")
            self.writer.add_scalar(
                f"metric/best-{bench_name}", self.best_bench, self.global_step
            )
            print(f"new best {bench_name} metric: {self.best_bench}")

    def _checkpoint_step(self, mark: str = None):
        mark = self.global_step if mark is None else mark

        # save model
        torch.save(
            self.network.state_dict(),
            Path(self.opt.path) / "checkpoint" / f"model-{mark}.pth",
        )

        # save optimizer state
        for name, optimizer in self.optimizer_dict.items():
            torch.save(
                optimizer.state_dict(),
                Path(self.opt.path) / "checkpoint" / f"optimizer-{name}-{mark}.pth",
            )

        # save scheduler state
        for name, scheduler in self.scheduler_dict.items():
            torch.save(
                scheduler.state_dict(),
                Path(self.opt.path) / "checkpoint" / f"scheduler-{name}-{mark}.pth",
            )

        print(f"saved checkpoint at ({mark}) to {Path(self.opt.path) / 'checkpoint'}")


@SCRIPT_REGISTRY.register()
class DDPTrainScript(TrainScript):
    def __init__(self, opt, rank, world_size):
        self.opt = opt
        self.rank = rank
        self.world_size = world_size

        # device select
        self.device = self.rank
        print(f"[rank {self.rank}] launched")

        # current loss & bench
        self.current_loss = torch.inf
        self.current_bench = torch.inf

    def run(self):
        if self.rank == 0:
            self.setup_dir()

        self.load_data()
        self.train_prep()
        self.train_loop()

    def dry_run(self, step=10):
        """
        Dry run the train loop for verification
        """
        if self.rank == 0:
            self.setup_dir()

        try:
            self.load_data()
            self.train_prep()

            # total step calculation
            epochs = self.opt.train.epochs
            start_time = time.time()
            self.global_step = 0
            self.total_step = epochs * len(self.train_dataloader)
            batch_num = len(self.train_dataloader)
            self.best_bench = torch.inf

            if type(self.opt.train.test_interval) is float:
                self.opt.train.test_interval = (
                    int(self.total_step * self.opt.train.test_interval) + 1
                )
                print(f"will test every {self.opt.train.test_interval} steps")

            # dry run the 1st few steps
            epoch = 0
            self.train_sampler.set_epoch(epoch)

            for batch, data in enumerate(self.train_dataloader):
                self.global_step += 1
                self.print_training_progress(
                    epoch,
                    epochs,
                    batch,
                    batch_num,
                    self.global_step,
                    self.total_step,
                    start_time,
                )
                self._train_step(data)

                if self.global_step % self.opt.train.test_interval == 0:
                    self._test_step()

                if self.opt.train.checkpoint_interval and (
                    self.global_step % self.opt.train.checkpoint_interval == 0
                ):
                    self._checkpoint_step()

                for name, scheduler in self.opt.train.scheduler.items():
                    if scheduler.step_per_batch:
                        self.scheduler_dict[name].step()

                        if self.rank == 0:
                            self.writer.add_scalar(
                                f"lr/{name}",
                                self.scheduler_dict[name].get_last_lr()[0],
                                self.global_step,
                            )

                if self.global_step >= step:
                    break

        except Exception as e:
            print("-" * 100)
            traceback.print_exc()
            print("-" * 100)

        # remove the experiment path
        if self.rank == 0:
            shutil.rmtree(self.opt.path)
            print(f"deleted experiment folder {self.opt.path}")

    def load_data(self):
        # training data
        self.train_dataset = DATASET_REGISTRY[self.opt.train.dataset.train.name](
            **self.opt.train.dataset.train.args
        )

        self.train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.opt.random_seed,
        )

        g_train = torch.Generator()
        g_train.manual_seed(self.opt.random_seed + self.rank)

        self.train_dataloader = DATALOADER_REGISTRY[
            self.opt.train.dataset.train.dataloader.name
        ](
            self.train_dataset,
            sampler=self.train_sampler,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g_train,
            **self.opt.train.dataset.train.dataloader.args,
        )

        print(
            f"[rank {self.rank}] will sample {len(self.train_sampler)} out of {len(self.train_dataset)} samples for training"
        )

        # testing data
        self.test_dataset = DATASET_REGISTRY[self.opt.train.dataset.test.name](
            **self.opt.train.dataset.test.args
        )

        self.test_sampler = DistributedSampler(
            self.test_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.opt.random_seed,
        )

        g_test = torch.Generator()
        g_test.manual_seed(self.opt.random_seed + self.rank)

        self.test_dataloader = DATALOADER_REGISTRY[
            self.opt.train.dataset.test.dataloader.name
        ](
            self.test_dataset,
            sampler=self.test_sampler,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g_test,
            **self.opt.train.dataset.test.dataloader.args,
        )

        print(
            f"[rank {self.rank}] will sample {len(self.test_sampler)} out of {len(self.test_dataset)} samples for testing"
        )

    def train_prep(self):
        # init/load model
        self.network = NETWORK_REGISTRY[self.opt.network.name](
            self.opt, **self.opt.network.args
        ).to(self.device)
        self.network_dict = dict(self.network.named_children())
        print(f"[rank {self.rank}] initialized model {self.opt.network.name}")

        if "load_from" in self.opt.network:
            pth_path = self.opt.network.load_from
            map_location = {"cuda:%d" % 0: "cuda:%d" % self.rank}
            self.network.load_state_dict(
                torch.load(pth_path, map_location=map_location)
            )
            print(f"[rank {self.rank}] loaded model weights from {pth_path}")

        self.ddp_network = DDP(self.network, device_ids=[self.rank])

        # init optimizer dict
        self.optimizer_dict = {}

        for name, optimizer in self.opt.train.optimizer.items():
            if name == "all":
                assert len(self.opt.train.optimizer.keys()) == 1, (
                    'optimizer with name "all" should be the only one in the optimizer dict'
                )
                self.optimizer_dict[name] = OPTIMIZER_REGISTRY[optimizer["name"]](
                    **optimizer["args"], params=self.network.parameters()
                )

            else:
                self.optimizer_dict[name] = OPTIMIZER_REGISTRY[optimizer["name"]](
                    **optimizer["args"], params=self.network_dict[name].parameters()
                )

            if "load_from" in optimizer:
                pth_path = optimizer.load_from
                map_location = {"cuda:%d" % 0: "cuda:%d" % self.rank}
                self.optimizer_dict[name].load_state_dict(
                    torch.load(pth_path, map_location=map_location)
                )
                print(f"[rank {self.rank}] loaded optimizer weights from {pth_path}")

        # init scheduler dict
        if "scheduler" in self.opt.train:
            self.scheduler_dict = {}

            for name, scheduler in self.opt.train.scheduler.items():
                if name == "all":
                    assert len(self.opt.train.scheduler.keys()) == 1, (
                        'scheduler with name "all" should be the only one in the scheduler dict'
                    )
                    self.scheduler_dict[name] = SCHEDULER_REGISTRY[scheduler["name"]](
                        **scheduler["args"], optimizer=self.optimizer_dict["all"]
                    )

                else:
                    self.scheduler_dict[name] = SCHEDULER_REGISTRY[scheduler["name"]](
                        **scheduler["args"], optimizer=self.optimizer_dict[name]
                    )

                if "load_from" in scheduler:
                    pth_path = scheduler.load_from
                    map_location = {"cuda:%d" % 0: "cuda:%d" % self.rank}
                    self.scheduler_dict[name].load_state_dict(
                        torch.load(pth_path, map_location=map_location)
                    )
                    print(
                        f"[rank {self.rank}] loaded scheduler weights from {pth_path}"
                    )

        # init loss dict
        self.loss_dict = {}
        self.loss_weight = {}

        for name, loss in self.opt.train.loss.items():
            self.loss_dict[name] = LOSS_REGISTRY[loss["name"]](**loss["args"]).to(
                self.device
            )
            self.loss_weight[name] = loss["weight"]

        # init metric dict
        self.metric_dict = {}

        for name, metric in self.opt.train.metric.items():
            self.metric_dict[name] = METRIC_REGISTRY[metric["name"]](
                **metric["args"]
            ).to(self.device)

    def print_training_progress(
        self, epoch, epoch_num, batch, batch_num, step, step_total, start_time
    ):
        progress_rate = step / step_total
        elapsed_time = time.time() - start_time
        step_throughput = (step + 1) / elapsed_time
        eta_time = (
            elapsed_time / progress_rate - elapsed_time if progress_rate > 0 else 0
        )
        print(
            f"[rank {self.rank}] training | epoch {epoch + 1}/{epoch_num} batch {batch + 1}/{batch_num} step {self.global_step}/{step_total} ({progress_rate * 100:.2f}%) | loss {self.current_loss} | throughput {step_throughput:.2f} batch/s elapsed {format_time(elapsed_time)} | eta {format_time(eta_time)} @{self.opt.exp_name}"
        )

    def print_testing_progress(self, batch, batch_num, start_time):
        test_epoch_num = self.total_step // self.opt.train.test_interval

        if self.global_step == self.total_step:
            test_epoch = "-"
        else:
            test_epoch = self.global_step // self.opt.train.test_interval

        progress_rate = batch / batch_num
        elapsed_time = time.time() - start_time
        batch_throughput = (batch + 1) / elapsed_time
        eta_time = (
            elapsed_time / progress_rate - elapsed_time if progress_rate > 0 else 0
        )
        print(
            f"[rank {self.rank}] testing | epoch {test_epoch}/{test_epoch_num} batch {batch + 1}/{batch_num} ({progress_rate * 100:.2f}%) | bench {self.current_bench} | throughput {batch_throughput:.2f} batch/s elapsed {format_time(elapsed_time)} | eta {format_time(eta_time)} @{self.opt.exp_name}"
        )

    def train_loop(self):
        epochs = self.opt.train.epochs
        start_time = time.time()
        self.global_step = 0
        self.total_step = epochs * len(self.train_dataloader)
        batch_num = len(self.train_dataloader)
        self.best_bench = torch.inf

        if type(self.opt.train.test_interval) is float:
            self.opt.train.test_interval = (
                int(self.total_step * self.opt.train.test_interval) + 1
            )
            print(f"will test every {self.opt.train.test_interval} steps")

        for epoch in range(epochs):
            self.train_sampler.set_epoch(epoch)

            for batch, data in enumerate(self.train_dataloader):
                self.global_step += 1
                self.print_training_progress(
                    epoch,
                    epochs,
                    batch,
                    batch_num,
                    self.global_step,
                    self.total_step,
                    start_time,
                )

                self._train_step(data)

                if self.global_step % self.opt.train.test_interval == 0:
                    self._test_step()

                if self.opt.train.checkpoint_interval and (
                    self.global_step % self.opt.train.checkpoint_interval == 0
                ):
                    self._checkpoint_step()

                # step scheduler per batch if specified
                for name, scheduler in self.opt.train.scheduler.items():
                    if scheduler.step_per_batch:
                        self.scheduler_dict[name].step()

                        if self.rank == 0:
                            self.writer.add_scalar(
                                f"lr/{name}",
                                self.scheduler_dict[name].get_last_lr()[0],
                                self.global_step,
                            )

            # step scheduler per epoch if specified
            for name, scheduler in self.opt.train.scheduler.items():
                if scheduler.step_per_epoch:
                    self.scheduler_dict[name].step()

                    if self.rank == 0:
                        self.writer.add_scalar(
                            f"lr/{name}",
                            self.scheduler_dict[name].get_last_lr()[0],
                            self.global_step,
                        )

        # final test & checkpoint
        print(f"[rank {self.rank}] running final checkpoint & testing...")
        self._checkpoint_step()
        self._test_step()

        if self.rank == 0:
            self.writer.flush()

    def _train_step(self, data):
        data = to_device(data, self.device)
        self.ddp_network.train()

        # forward pass & loss calculation
        infer = self.ddp_network(data)
        loss_total = 0.0

        for name, loss in self.loss_dict.items():
            loss.train()
            loss_val = loss(infer, data)
            loss_total += loss_val * self.loss_weight[name]

            if self.rank == 0:
                # only log training loss of rank 0
                # p.s. gather losses from all ranks w/ all reduce is possible
                # but it unnecessarily introduces communication overhead
                self.writer.add_scalar(
                    f"loss/train/{name}",
                    loss_val.item() / self.train_dataloader.batch_size,
                    self.global_step,
                )

        self.current_loss = loss_total.item() / self.train_dataloader.batch_size

        if self.rank == 0:
            self.writer.add_scalar(
                "loss/train/total",
                loss_total.item() / self.train_dataloader.batch_size,
                self.global_step,
            )

        # loss backward
        for optimizer in self.optimizer_dict.values():
            optimizer.zero_grad()

        loss_total.backward()

        if "gradient_clipping" in self.opt.train.keys():
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.opt.train.gradient_clipping.max_norm,
            )

        for optimizer in self.optimizer_dict.values():
            optimizer.step()

    def _test_step(self):
        start_time = time.time()
        batch_num = len(self.test_dataloader)

        # turn on eval mode and start feed
        self.ddp_network.eval()

        for name, loss in self.loss_dict.items():
            loss.eval()
            loss.start_test_feed(self, name, self.rank, log_tag="test")

        for name, metric in self.metric_dict.items():
            metric.eval()
            metric.start_test_feed(self, name, self.rank, log_tag="test")

        # turn off autograd & run the testing
        with torch.no_grad():
            for batch, data in enumerate(self.test_dataloader):
                self.print_testing_progress(batch, batch_num, start_time)

                # forward pass
                data = to_device(data, self.device)
                infer = self.ddp_network(data)

                # loss calculation
                for name, loss in self.loss_dict.items():
                    val = loss.test_feed(infer, data)

                    if name == self.opt.train.save_best:
                        self.current_bench = val

                # metric calculation
                for name, metric in self.metric_dict.items():
                    val = metric.test_feed(infer, data)

                    if name == self.opt.train.save_best:
                        self.current_bench = val

        # end feed
        for name, loss in self.loss_dict.items():
            loss.end_test_feed()
            print(f"[rank {self.rank}] loss {name}: {loss.loss_avg}")

        for name, metric in self.metric_dict.items():
            metric.end_test_feed()
            print(f"[rank {self.rank}] metric {name}: {metric.metric_avg}")

        # save current model if it's the best so far
        # p.s. only in rank 0
        if self.rank == 0:
            bench_name = self.opt.train.save_best

            if (
                bench_name in self.loss_dict
                and self.best_bench > self.loss_dict[bench_name].loss_avg
            ):
                self.best_bench = self.loss_dict[bench_name].loss_avg
                self._checkpoint_step(mark="best")
                self.writer.add_scalar(
                    f"loss/best-{bench_name}", self.best_bench, self.global_step
                )

            if (
                bench_name in self.metric_dict
                and self.best_bench > self.metric_dict[bench_name].metric_avg
            ):
                self.best_bench = self.metric_dict[bench_name].metric_avg
                self._checkpoint_step(mark="best")
                self.writer.add_scalar(
                    f"metric/best-{bench_name}", self.best_bench, self.global_step
                )

    def _checkpoint_step(self, mark: str = None):
        if self.rank == 0:
            mark = self.global_step if mark is None else mark

            # save model
            torch.save(
                self.network.state_dict(),
                Path(self.opt.path) / "checkpoint" / f"model-{mark}.pth",
            )

            # save optimizer state
            for name, optimizer in self.optimizer_dict.items():
                torch.save(
                    optimizer.state_dict(),
                    Path(self.opt.path) / "checkpoint" / f"optimizer-{name}-{mark}.pth",
                )

            # save scheduler state
            for name, scheduler in self.scheduler_dict.items():
                torch.save(
                    scheduler.state_dict(),
                    Path(self.opt.path) / "checkpoint" / f"scheduler-{name}-{mark}.pth",
                )

            print(
                f"[rank {self.rank}] saved checkpoint at ({mark}) to {Path(self.opt.path) / 'checkpoint'}"
            )


# --- benchmark scripts ---
@SCRIPT_REGISTRY.register()
class BenchScript:
    def __init__(self, opt, bench_name, bench_setting):
        self.opt = opt
        self.bench_name = bench_name
        self.bench_setting = bench_setting

        # device select
        if self.opt.device_select == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.opt.device_select

        # current loss & bench
        self.current_loss = torch.inf
        self.current_bench = torch.inf

        print(f"[{self.bench_name}] running on {self.device}...")

    def run(self):
        self.writer = SummaryWriter(Path(self.opt.path) / "bench")
        self.load_data()
        self.bench_prep()
        self.bench_loop()

    def load_data(self):
        self.test_dataset = DATASET_REGISTRY[self.bench_setting.dataset.name](
            **self.bench_setting.dataset.args
        )
        self.test_dataloader = DATALOADER_REGISTRY[
            self.bench_setting.dataset.dataloader.name
        ](self.test_dataset, **self.bench_setting.dataset.dataloader.args)
        print(
            f"[{self.bench_name}] load testing dataset with {len(self.test_dataset)} samples"
        )

    def bench_prep(self):
        # init model
        self.network = NETWORK_REGISTRY[self.opt.network.name](
            self.opt, **self.opt.network.args
        ).to(self.device)
        print(f"[{self.bench_name}] initialized model {self.opt.network.name}")

        # load model weights
        if "network_weight" in self.bench_setting:
            pth_path = self.bench_setting.network_weight

        else:
            pth_path = Path(self.opt.path) / "checkpoint" / "model-best.pth"

        state_dict = torch.load(pth_path, map_location=self.device)
        if any(k.startswith("module.") for k in state_dict.keys()):
            # handle DDP 'module.' prefix
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        self.network.load_state_dict(state_dict)
        print(f"[{self.bench_name}] loaded model weights from {pth_path}")

        # init loss dict
        self.loss_dict = {}
        self.loss_weight = {}

        for name, loss in self.bench_setting.loss.items():
            self.loss_dict[name] = LOSS_REGISTRY[loss["name"]](**loss["args"]).to(
                self.device
            )
            self.loss_weight[name] = loss["weight"]

        # init metric dict
        self.metric_dict = {}

        for name, metric in self.bench_setting.metric.items():
            self.metric_dict[name] = METRIC_REGISTRY[metric["name"]](
                **metric["args"]
            ).to(self.device)

    def print_benchmarking_progress(self, batch, batch_num, start_time):
        progress_rate = batch / batch_num
        elapsed_time = time.time() - start_time
        batch_throughput = (batch + 1) / elapsed_time
        eta_time = (
            elapsed_time / progress_rate - elapsed_time if progress_rate > 0 else 0
        )
        print(
            f"[{self.bench_name}] benchmarking | bench {self.current_bench} | batch {batch + 1}/{batch_num} ({progress_rate * 100:.2f}%) | throughput {batch_throughput:.2f} batch/s elapsed {format_time(elapsed_time)} eta {format_time(eta_time)} @{self.opt.exp_name}"
        )

        if batch == batch_num - 1:
            self.writer.add_scalar(
                f"bench/{self.bench_name}/throughput",
                batch_throughput,
                self.global_step,
            )

    def bench_loop(self):
        self.global_step = 0
        batch_num = len(self.test_dataloader)
        start_time = time.time()

        # turn on eval mode and start feed
        self.network.eval()

        for name, loss in self.loss_dict.items():
            loss.eval()
            loss.start_test_feed(self, name, log_tag=f"bench/{self.bench_name}")

        for name, metric in self.metric_dict.items():
            metric.eval()
            metric.start_test_feed(self, name, log_tag=f"bench/{self.bench_name}")

        # initialize per-batch storage
        loss_batch_vals = {name: [] for name in self.loss_dict.keys()}
        metric_batch_vals = {name: [] for name in self.metric_dict.keys()}
        batch_start = getattr(self.bench_setting, "batch_start", None)
        batch_end = getattr(self.bench_setting, "batch_end", None)

        # turn off autograd & run the testing
        with torch.no_grad():
            for batch, data in enumerate(self.test_dataloader):
                if batch_start is not None and batch < batch_start:
                    continue

                if batch_end is not None and batch > batch_end:
                    break

                self.print_benchmarking_progress(batch, batch_num, start_time)

                # forward pass
                data = to_device(data, self.device)
                infer = self.network(data)

                # loss calculation
                for name, loss in self.loss_dict.items():
                    val = loss.test_feed(infer, data)
                    loss_batch_vals[name].append(
                        val.item() if torch.is_tensor(val) else val
                    )

                    if name == self.opt.train.save_best:
                        self.current_bench = val

                # metric calculation
                for name, metric in self.metric_dict.items():
                    val = metric.test_feed(infer, data)
                    metric_batch_vals[name].append(
                        val.item() if torch.is_tensor(val) else val
                    )

                    if name == self.opt.train.save_best:
                        self.current_bench = val

        # end feed
        for name, loss in self.loss_dict.items():
            loss.end_test_feed()
            print(f"[{self.bench_name}] {name} average: {loss.loss_avg}")

        for name, metric in self.metric_dict.items():
            metric.end_test_feed()
            print(f"[{self.bench_name}] {name} average: {metric.metric_avg}")

        # Export per-batch values to CSV
        import csv

        bench_dir = Path(self.opt.path) / "bench"
        (bench_dir / self.bench_name).mkdir(parents=True, exist_ok=True)

        for name, vals in loss_batch_vals.items():
            csv_path = bench_dir / self.bench_name / f"loss_{name}_per_batch.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["batch_idx", "value"])
                for batch_idx, val in enumerate(vals):
                    writer.writerow([batch_idx, val])
            print(f"[{self.bench_name}] exported {len(vals)} loss values to {csv_path}")

        for name, vals in metric_batch_vals.items():
            csv_path = bench_dir / self.bench_name / f"metric_{name}_per_batch.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["batch_idx", "value"])
                for batch_idx, val in enumerate(vals):
                    writer.writerow([batch_idx, val])
            print(
                f"[{self.bench_name}] exported {len(vals)} metric values to {csv_path}"
            )

        self.writer.flush()
