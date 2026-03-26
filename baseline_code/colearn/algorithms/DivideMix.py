"""DivideMix: Learning with Noisy Labels as Semi-Supervised Learning (ICLR 2020).

Reference: Li, Sober, and Han, "DivideMix: Learning with Noisy Labels as
Semi-Supervised Learning," in ICLR, 2020.

Core idea:
  1. Fit a two-component GMM on per-sample cross-entropy losses to split the
     training set into a *clean* (labeled) subset and a *noisy* (unlabeled) subset.
  2. Apply MixMatch-style semi-supervised training: Mixup augmentation on the
     labeled set and sharpened pseudo-labels on the unlabeled set.
  3. Two networks are trained in a co-divide fashion -- each network's GMM
     provides the split for the *other* network's training, reducing
     confirmation bias.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DivideMix:
    """Two-network co-divide training with GMM noise detection and MixMatch."""

    def __init__(self, config=None, input_channel=3, num_classes=10,
                 model1=None, model2=None):
        self.model1 = model1
        self.model2 = model2
        self.num_classes = num_classes
        self.epochs = 80
        self.warmup_epochs = 10
        self.lambda_u = 25.0
        self.T = 0.5
        self.alpha = 4.0
        self.lr = 0.02

        if config is not None:
            self.epochs = config.get("epochs", self.epochs)
            self.warmup_epochs = config.get("warmup_epochs", self.warmup_epochs)
            self.lambda_u = config.get("lambda_u", self.lambda_u)
            self.T = config.get("T", self.T)
            self.alpha = config.get("alpha", self.alpha)
            self.lr = config.get("lr", self.lr)
            self.num_classes = config.get("num_classes", self.num_classes)

        self.optimizer1 = None
        self.optimizer2 = None
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def _init_optimizers(self):
        if self.optimizer1 is None:
            self.optimizer1 = torch.optim.SGD(
                self.model1.parameters(), lr=self.lr,
                momentum=0.9, weight_decay=5e-4)
        if self.optimizer2 is None:
            self.optimizer2 = torch.optim.SGD(
                self.model2.parameters(), lr=self.lr,
                momentum=0.9, weight_decay=5e-4)

    # ------------------------------------------------------------------
    # GMM-based clean/noisy split
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_losses(self, model, dataloader):
        """Return per-sample CE losses for the entire dataset."""
        model.eval()
        all_losses = []
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = self.ce_loss(logits, labels)
            all_losses.append(loss.cpu())
        return torch.cat(all_losses).numpy()

    def gmm_split(self, model, dataloader):
        """Fit a 2-component GMM on losses; return clean probability per sample."""
        losses = self._compute_losses(model, dataloader)
        losses = losses.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=50,
                               tol=1e-3, reg_covar=5e-4)
        gmm.fit(losses)
        probs = gmm.predict_proba(losses)
        clean_comp = np.argmin(gmm.means_.flatten())
        return probs[:, clean_comp]

    # ------------------------------------------------------------------
    # MixMatch-style semi-supervised step
    # ------------------------------------------------------------------
    @staticmethod
    def _sharpen(p, T):
        pt = p ** (1.0 / T)
        return pt / pt.sum(dim=1, keepdim=True)

    @staticmethod
    def _mixup(x1, y1, x2, y2, alpha):
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1.0 - lam)
        xm = lam * x1 + (1.0 - lam) * x2
        ym = lam * y1 + (1.0 - lam) * y2
        return xm, ym

    def _semisup_loss(self, logits_x, targets_x, logits_u, targets_u, epoch):
        """Labeled CE + unlabeled MSE with linear ramp-up."""
        loss_x = -torch.mean(
            torch.sum(F.log_softmax(logits_x, dim=1) * targets_x, dim=1))
        loss_u = torch.mean((F.softmax(logits_u, dim=1) - targets_u) ** 2)
        ramp = min(1.0, float(epoch) / max(self.warmup_epochs, 1))
        return loss_x + self.lambda_u * ramp * loss_u

    def _mixmatch_step(self, model, optimizer, images_l, labels_l,
                       images_u, epoch):
        """One MixMatch training step on a single (labeled, unlabeled) pair."""
        model.train()
        batch = images_l.size(0)

        targets_l = F.one_hot(labels_l, self.num_classes).float()

        with torch.no_grad():
            logits_u = model(images_u)
            targets_u = self._sharpen(F.softmax(logits_u, dim=1), self.T)

        all_x = torch.cat([images_l, images_u], dim=0)
        all_y = torch.cat([targets_l, targets_u], dim=0)

        idx = torch.randperm(all_x.size(0))
        mix_x, mix_y = self._mixup(all_x, all_y, all_x[idx], all_y[idx],
                                    self.alpha)

        logits = model(mix_x)
        logits_l = logits[:batch]
        logits_u = logits[batch:]

        loss = self._semisup_loss(logits_l, mix_y[:batch],
                                  logits_u, mix_y[batch:], epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Warmup: standard CE
    # ------------------------------------------------------------------
    def _warmup_step(self, model, optimizer, images, labels):
        model.train()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Public API (matches colearn framework)
    # ------------------------------------------------------------------
    def train(self, trainloader, epoch):
        """One epoch of DivideMix training."""
        self._init_optimizers()

        if epoch < self.warmup_epochs:
            pbar = tqdm(trainloader, desc=f"[DivideMix warmup {epoch+1}]")
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)
                l1 = self._warmup_step(self.model1, self.optimizer1,
                                       images, labels)
                l2 = self._warmup_step(self.model2, self.optimizer2,
                                       images, labels)
                pbar.set_postfix(loss1=f"{l1:.4f}", loss2=f"{l2:.4f}")
            return

        clean_prob1 = self.gmm_split(self.model1, trainloader)
        clean_prob2 = self.gmm_split(self.model2, trainloader)
        threshold = 0.5

        all_images, all_labels = [], []
        for images, labels in trainloader:
            all_images.append(images)
            all_labels.append(labels)
        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        self._co_train_one(self.model1, self.optimizer1,
                           all_images, all_labels, clean_prob2,
                           threshold, epoch, net_id=1)
        self._co_train_one(self.model2, self.optimizer2,
                           all_images, all_labels, clean_prob1,
                           threshold, epoch, net_id=2)

    def _co_train_one(self, model, optimizer, images, labels, clean_prob,
                      threshold, epoch, net_id=1):
        """Train *model* using the GMM split from the *other* network."""
        clean_mask = clean_prob >= threshold
        noisy_mask = ~clean_mask

        imgs_l = images[clean_mask].to(device)
        lbls_l = labels[clean_mask].to(device)
        imgs_u = images[noisy_mask].to(device)

        if imgs_l.size(0) == 0 or imgs_u.size(0) == 0:
            pbar = tqdm(range(0, images.size(0), 128),
                        desc=f"[DivideMix net{net_id} fallback e{epoch+1}]")
            for i in pbar:
                batch_imgs = images[i:i+128].to(device)
                batch_lbls = labels[i:i+128].to(device)
                loss = self._warmup_step(model, optimizer,
                                         batch_imgs, batch_lbls)
                pbar.set_postfix(loss=f"{loss:.4f}")
            return

        bs = 128
        n_batches = max(imgs_l.size(0), imgs_u.size(0)) // bs + 1
        pbar = tqdm(range(n_batches),
                    desc=f"[DivideMix net{net_id} e{epoch+1}]")
        for b in pbar:
            idx_l = torch.randint(0, imgs_l.size(0), (bs,))
            idx_u = torch.randint(0, imgs_u.size(0), (bs,))
            loss = self._mixmatch_step(
                model, optimizer,
                imgs_l[idx_l], lbls_l[idx_l],
                imgs_u[idx_u], epoch)
            pbar.set_postfix(loss=f"{loss:.4f}")

    def evaluate(self, test_loader):
        """Return (acc1, acc2) matching the colearn interface."""
        acc1 = self._eval_single(self.model1, test_loader)
        acc2 = self._eval_single(self.model2, test_loader)
        return acc1, acc2

    @staticmethod
    @torch.no_grad()
    def _eval_single(model, loader):
        model.eval()
        correct, total = 0, 0
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            _, pred = logits.max(1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum().item()
        return 100.0 * correct / total
