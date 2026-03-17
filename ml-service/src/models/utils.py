import numpy as np
from torch.utils.data import Sampler

class TaskBalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=16):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Extract task IDs to group indices
        task_ids = np.array(dataset['task_id'])
        self.task_0_idx = np.where(task_ids == 0)[0] # Quality
        self.task_1_idx = np.where(task_ids == 1)[0] # Component
        self.task_2_idx = np.where(task_ids == 2)[0] # Stance
        
        # Define the exact composition of every batch
        self.n_t0 = batch_size // 2  # 8 slots
        self.n_t1 = batch_size // 4  # 4 slots
        self.n_t2 = batch_size - self.n_t0 - self.n_t1 # 4 slots
        
        # Define epoch length by the largest dataset (Quality)
        self.num_batches = len(self.task_0_idx) // self.n_t0

    def __iter__(self):
        # Shuffle indices at the start of each epoch
        np.random.shuffle(self.task_0_idx)
        np.random.shuffle(self.task_1_idx)
        np.random.shuffle(self.task_2_idx)

        for i in range(self.num_batches):
            batch = []
            
            # 1. Add Quality examples (Majority class)
            start_0 = i * self.n_t0
            batch.extend(self.task_0_idx[start_0 : start_0 + self.n_t0])
            
            # 2. Add Component examples (Wraps around if it runs out)
            start_1 = (i * self.n_t1) % len(self.task_1_idx)
            end_1 = start_1 + self.n_t1
            if end_1 > len(self.task_1_idx):
                idx = np.concatenate((self.task_1_idx[start_1:], self.task_1_idx[:end_1 - len(self.task_1_idx)]))
            else:
                idx = self.task_1_idx[start_1:end_1]
            batch.extend(idx)
            
            # 3. Add Stance examples (Wraps around if it runs out)
            start_2 = (i * self.n_t2) % len(self.task_2_idx)
            end_2 = start_2 + self.n_t2
            if end_2 > len(self.task_2_idx):
                idx = np.concatenate((self.task_2_idx[start_2:], self.task_2_idx[:end_2 - len(self.task_2_idx)]))
            else:
                idx = self.task_2_idx[start_2:end_2]
            batch.extend(idx)
            
            # Shuffle within the batch so the model doesn't learn rigid positional patterns
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches