"""
Our improvement: Balanced Test-time Prototype Refinement (TPR).

WaffleCLIP / CLIP classify an image by comparing it to one text embedding per class
("class prototype"). Two things go wrong:

  (a) Text and image embeddings live in different regions of CLIP's space (the
      "modality gap"), so a text prototype is not exactly where the images of that
      class actually are.
  (b) CLIP is biased towards some "popular" classes and predicts them far too often.

Idea (no training, no labels, ~15 lines):
  1. Classify all test images with the normal text prototypes  -> soft pseudo-labels.
     Fix (b): lower the score of classes that receive too many images (class balance).
  2. For every class, average the image embeddings weighted by those pseudo-labels
     -> an "image prototype" that sits where the class's images really are.
  3. Fix (a): new prototype = text prototype + alpha * image prototype. Repeat a few times.
"""
import torch
import torch.nn.functional as F


def class_balance(logits, iters=3):
    """Per-class score offset so that every class receives the same total probability.
    If a class is predicted too often, its offset becomes negative (Sinkhorn balancing)."""
    bias = torch.zeros(logits.shape[1], device=logits.device)
    for _ in range(iters):
        bias = bias - (logits + bias).log_softmax(dim=-1).logsumexp(dim=0)  # = log(total prob. per class), NaN-safe
    return bias


def refine_prototypes(text_protos, image_embs, alpha=1.0, steps=3, temperature=100.0, balance=True):
    """
    text_protos: [num_classes, dim]  one text embedding per class (e.g. mean of Waffle prompts)
    image_embs:  [num_images, dim]   L2-normalised embeddings of the (unlabelled) test images
    returns:     refined prototypes [num_classes, dim] and a per-class score offset [num_classes]
    """
    text_protos = F.normalize(text_protos.float(), dim=-1)
    image_embs = F.normalize(image_embs.float(), dim=-1)

    protos = text_protos
    for _ in range(steps):
        logits = temperature * image_embs @ protos.T                    # [N, C]
        if balance:
            logits = logits + class_balance(logits)
        probs = logits.softmax(dim=-1)                                   # soft pseudo-labels
        image_protos = F.normalize(probs.T @ image_embs, dim=-1)        # [C, D] weighted class centroids
        protos = F.normalize(text_protos + alpha * image_protos, dim=-1)

    bias = torch.zeros(protos.shape[0], device=protos.device)
    if balance:  # offset for the final prediction, in raw cosine-similarity units
        bias = class_balance(temperature * image_embs @ protos.T) / temperature
    return protos, bias
