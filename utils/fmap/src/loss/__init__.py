import torch
from torch import nn
import torch.distributed as dist

# customized base class
class BaseLoss(nn.Module):
    def forward(self, infer, data):
        """Default method to invoke a forward pass"""
        raise NotImplementedError("Forward method not implemented!")

    def get_batch_size(self, infer, data):
        """
        Method to get the batch size from the inference results or data
        """
        raise NotImplementedError("Forward method not implemented!")

    def start_test_feed(self, script, name, rank=None, log_tag: str = 'test'):
        """
        Method to start feeding the loss method. Typically called at the beginning of the testing/benchmark epoch
        
        Args:
            script: The training/benchmark script object
            name: The name of the loss method
            rank: The rank of the process (if using distributed training)
            log_tag: The tag for logging the loss. Default is 'test', since it's typically used in the testing epoch. If used in training, it can be left unset. If used in benchmarking, it can be set to 'benchmark/<bench_name>'.
        """
        self.script = script
        self.name = name
        self.rank = rank
        self.log_tag = log_tag

        # init state for feed
        self.loss_total = 0.0
        self.loss_avg = 0.0
        self.sample_total = 0

    def test_feed(self, infer, data):
        """
        Wrapping method around the forward method with total loss gathering functionalities. Typically called during test epoch
        """
        loss_val = self(infer, data)
        self.loss_total += loss_val
        self.sample_total += self.get_batch_size(infer, data)
        return loss_val

    def end_test_feed(self):
        """
        Method to end feeding the loss method. Typically called at the end of the testing/benchmark epoch

        Noted that `self.metric_avg` will be used to determin whether the current model is the best so far and thus be checkpointed
        """
        if self.rank is None:
            # if not using distributed training, simply log the average loss
            self.loss_avg = self.loss_total / self.sample_total
            self.script.writer.add_scalar(f'loss/{self.log_tag}/{self.name}', self.loss_avg, self.script.global_step)

        else:
            # if using distributed training, gather the total loss and sample count across all ranks
            # only log in rank 0
            self.sample_total = torch.tensor(self.sample_total).to(device=self.rank)
            dist.all_reduce(self.loss_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.sample_total, op=dist.ReduceOp.SUM)

            if self.rank == 0:
                self.loss_avg = self.loss_total / self.sample_total
                self.script.writer.add_scalar(f'loss/{self.log_tag}/{self.name}', self.loss_avg, self.script.global_step)
