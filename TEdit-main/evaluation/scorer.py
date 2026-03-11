import torch
import numpy as np

import evaluation.basic_metrics as basic_metrics


class Scorer:
    def __init__(self, metric="MAE"):
        self.metric = getattr(basic_metrics, metric)
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.total_scores = 0
    
    def __call__(self, pred, gt):
        """
        Get score for a single pair of (pred, gt).
        """
        batch_size = pred.shape[0]
        gt = gt.to(pred.device)

        scores = self.metric(pred, gt)
        scores = torch.reshape(scores, [batch_size, -1])
        scores = torch.mean(scores, dim=-1).cpu().numpy()
        self.total_scores += np.sum(scores)
        self.total_samples += batch_size
        return scores

    @property
    def mean(self):
        return self.total_scores/self.total_samples


if __name__ == "__main__":
    pred = torch.randn((5,3))
    gt = torch.randn((5,3))

    mse = Scorer("MAE")
    mse(pred, gt)
    print((pred-gt).mean())
    print(mse.mean())