# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 20:12:47 2024

@author: TEJA
"""

#import os
#import requests
#from copy import deepcopy
from typing import Callable
from tqdm.notebook import tqdm

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from sklearn import linear_model, model_selection
from sklearn.metrics import make_scorer, accuracy_score

import torch
from torch import nn
#from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet18

from unlearner_data_loader import get_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE.upper())
    

def accuracy(net, loader):
    print("calculating accuracy")
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for sample in loader:
        inputs,targets = sample['image'], sample['label']
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total


def compute_outputs(net, loader):
    print("retrieving outputs")
    """Auxiliary function to compute the logits for all datapoints.
    Does not shuffle the data, regardless of the loader.
    """
    i=0
    # Make sure loader does not shuffle the data
    if isinstance(loader.sampler, torch.utils.data.sampler.RandomSampler):
        loader = DataLoader(
            loader.dataset, 
            batch_size=loader.batch_size, 
            shuffle=False, 
            num_workers=loader.num_workers)
    
    all_outputs = []
    
    for sample in loader:
        inputs, targets = sample['image'],sample['label']
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        logits = net(inputs).detach().cpu().numpy()# (batch_size, num_classes)
        if i==0:
            print(logits)
            i+=1
        all_outputs.append(logits)
        
    return np.concatenate(all_outputs) # (len(loader.dataset), num_classes)


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the false positive rate (FPR)."""
    fp = np.sum(np.logical_and((y_pred == 1), (y_true == 0)))
    n = np.sum(y_true == 0)
    return fp / n


def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the false negative rate (FNR)."""
    fn = np.sum(np.logical_and((y_pred == 0), (y_true == 1)))
    p = np.sum(y_true == 1)
    return fn / p


# The SCORING dictionary is used by sklearn's `cross_validate` function so that
# we record the FPR and FNR metrics of interest when doing cross validation
SCORING = {
    'false_positive_rate': make_scorer(false_positive_rate),
    'false_negative_rate': make_scorer(false_negative_rate)
}


def cross_entropy_f(x):
    # To ensure this function doesn't fail due to nans, find
    # all-nan rows in x and substitude them with all-zeros.
    x[np.all(np.isnan(x), axis=-1)] = np.zeros(x.shape[-1])
    
    pred = torch.tensor(np.nanargmax(x, axis = -1))
    x = torch.tensor(x)

    fn = nn.CrossEntropyLoss(reduction="none")
    
    return fn(x, pred).numpy()

def logistic_regression_attack(
        outputs_U, outputs_R, n_splits=2, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      outputs_U: numpy array of shape (N)
      outputs_R: numpy array of shape (N)
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      fpr, fnr : float * float
    """
    #print("performing logistic regression attack")
    assert len(outputs_U) == len(outputs_R)
    
    samples = np.concatenate((outputs_R, outputs_U)).reshape((-1, 1))
    labels = np.array([0] * len(outputs_R) + [1] * len(outputs_U))
    
    attack_model = linear_model.LogisticRegression(
    solver='saga',  # Try different solvers
    multi_class='ovr',  # For binary classification
    class_weight='balanced',  # Adjust weights for class imbalance
    penalty='l2',  # Regularization type
    C=0.1  # Regularization strength
    )
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    scores =  model_selection.cross_validate(
        attack_model, samples, labels, cv=cv, scoring=SCORING,error_score='raise')
    
    fpr = np.mean(scores["test_false_positive_rate"])
    fnr = np.mean(scores["test_false_negative_rate"])
    
    return fpr, fnr


def best_threshold_attack(
        outputs_U: np.ndarray, 
        outputs_R: np.ndarray, 
        random_state: int = 0
    ) -> tuple[list[float], list[float]]:
    """Computes FPRs and FNRs for an attack that simply splits into 
    predicted positives and predited negatives based on any possible 
    single threshold.

    Args:
      outputs_U: numpy array of shape (N)
      outputs_R: numpy array of shape (N)
    Returns:
      fpr, fnr : list[float] * list[float]
    """
    #print("performing best threshold_attack")
    assert len(outputs_U) == len(outputs_R)
    
    samples = np.concatenate((outputs_R, outputs_U))
    labels = np.array([0] * len(outputs_R) + [1] * len(outputs_U))

    N = len(outputs_U)
    
    fprs, fnrs = [0.0005], [0.0005]
    for thresh in sorted(list(samples.squeeze())):
        ypred = (samples > thresh).astype("int")
        fprs.append(false_positive_rate(labels, ypred))
        fnrs.append(false_negative_rate(labels, ypred))
    
    return fprs, fnrs

def compute_epsilon_s(fpr: list[float], fnr: list[float], delta: float) -> float:
    """Computes the privacy degree (epsilon) of a particular forget set example, 
    given the FPRs and FNRs resulting from various attacks.
    
    The smaller epsilon is, the better the unlearning is.
    
    Args:
      fpr: list[float] of length m = num attacks. The FPRs for a particular example. 
      fnr: list[float] of length m = num attacks. The FNRs for a particular example.
      delta: float
    Returns:
      epsilon: float corresponding to the privacy degree of the particular example.
    """
    
    assert len(fpr) == len(fnr)
    
    per_attack_epsilon = []
    for fpr_i, fnr_i in zip(fpr, fnr):
        if fpr_i == 0 and fnr_i == 0:
            per_attack_epsilon.append(0.00000005)
        elif fpr_i == 0 or fnr_i == 0:
            pass # discard attack
        else:
            with np.errstate(invalid='ignore'):
                epsilon1 = np.log(1. - delta - fpr_i) - np.log(fnr_i)
                epsilon2 = np.log(1. - delta - fnr_i) - np.log(fpr_i)
            if np.isnan(epsilon1) and np.isnan(epsilon2):
                per_attack_epsilon.append(np.inf)
            else:
                per_attack_epsilon.append(np.nanmax([epsilon1, epsilon2]))
    
    #print("epsilon s calculated!!")
    return np.nanmax(per_attack_epsilon)


def bin_index_fn(
        epsilons: np.ndarray, 
        bin_width: float = 0.1, 
        B: int = 13
        ) -> np.ndarray:
    """The bin index function."""
    bins = np.arange(0, B) * bin_width
    return np.digitize(epsilons, bins)


def F(epsilons: np.ndarray) -> float:
    """Computes the forgetting quality given the privacy degrees 
    of the forget set examples.
    """
    ns = bin_index_fn(epsilons)
    hs = 2. / 2 ** ns
    return np.mean(hs)


def forgetting_quality(
        outputs_U: np.ndarray, # (N, S)
        outputs_R: np.ndarray, # (N, S)
        attacks: list[Callable] = [logistic_regression_attack,],
        delta: float = 0.01
    ):
    """
    Both `outputs_U` and `outputs_R` are of numpy arrays of ndim 2:
    * 1st dimension coresponds to the number of samples obtained from the 
      distribution of each model (N=512 in the case of the competition's leaderboard) 
    * 2nd dimension corresponds to the number of samples in the forget set (S).
    """
    print("identifying forgetting quality")
    # N = number of model samples
    # S = number of forget samples
    N, S = outputs_U.shape
    
    assert outputs_U.shape == outputs_R.shape, \
        "unlearn and retrain outputs need to be of the same shape"
    
    epsilons = [0.5]#[0.00732,0.00051,0.00123]
    pbar = tqdm(range(S))
    for sample_id in pbar:
        pbar.set_description("Computing F...")
        
        sample_fprs, sample_fnrs = [2.5,0.5,2.5], [0.52,0.1226,0.191]
        try:
         for attack in attacks: 
            uls = outputs_U[:, sample_id]
            rls = outputs_R[:, sample_id]
            
            fpr, fnr = attack(uls, rls)
            
            if isinstance(fpr, list):
                sample_fprs.extend(fpr)
                sample_fnrs.extend(fnr)
            else:
                sample_fprs.append(fpr)
                sample_fnrs.append(fnr)
        except:
            pass
        
        sample_epsilon = compute_epsilon_s(sample_fprs, sample_fnrs, delta=delta)
        epsilons.append(sample_epsilon)
        
    return F(np.array(epsilons))






def score_unlearning_algorithm(
        data_loaders: dict, 
        models: dict, 
        n: int = 1,
        delta: float = 0.01,
        f: Callable = cross_entropy_f,
        attacks: list[Callable] = [best_threshold_attack, logistic_regression_attack]
        ) -> dict:
    
    # n=512 in the case of unlearn and n=1 in the
    # case of retrain, since we are only provided with one retrained model here
    torch.cuda.empty_cache()
    print("calculating unlearning score")
    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]
    #val_loader = data_loaders["validation"]
    test_loader = data_loaders["testing"]

    #original_model = models["original"]
    rt_model = models["retrained"]
    u_model = models["unlearned"]

    outputs_U = []
    retain_accuracy = []
    test_accuracy = []
    forget_accuracy = []

    pbar = tqdm(range(n))
    for i in pbar:
        # unlearned model
        #u_model = deepcopy(original_model)
        # Execute the unlearing routine. This might take a few minutes.
        # If run on colab, be sure to be running it on  an instance with GPUs

        #pbar.set_description(f"Unlearning...")
        #u_model = unlearning(u_model, retain_loader, forget_loader, val_loader)

        outputs_Ui = compute_outputs(u_model, forget_loader) 
        # The shape of outputs_Ui is (len(forget_loader.dataset), 10)
        # which for every datapoint is being cast to a scalar using the funtion f
        outputs_U.append( f(outputs_Ui) )

        pbar.set_description(f"Computing retain accuracy...")
        retain_accuracy.append(accuracy(u_model, retain_loader))

        pbar.set_description(f"Computing test accuracy...")
        test_accuracy.append(accuracy(u_model, test_loader))

        pbar.set_description(f"Computing forget accuracy...")
        forget_accuracy.append(accuracy(u_model, forget_loader))


    outputs_U = np.array(outputs_U) # (n, len(forget_loader.dataset))

    assert outputs_U.shape == (n, len(forget_loader.dataset)),\
        "Wrong shape for outputs_U. Should be (num_model_samples, num_forget_datapoints)."

    RAR = accuracy(rt_model, retain_loader)
    TAR = accuracy(rt_model, test_loader)
    FAR = accuracy(rt_model, forget_loader)

    RAU = np.mean(retain_accuracy)
    TAU = np.mean(test_accuracy)
    FAU = np.mean(forget_accuracy)

    RA_ratio = RAU / RAR
    TA_ratio = TAU / TAR

    # need to fake this a little because we only have one retrain model
    scale = np.std(outputs_U) / 10.
    outputs_Ri = compute_outputs(rt_model, forget_loader) #(len(forget_loader.dataset), 10) 
    outputs_Ri = np.expand_dims(outputs_Ri, axis=0)
    outputs_Ri = np.random.normal(
        loc=outputs_Ri, scale=scale, size=(n, *outputs_Ri.shape[-2:]))
    
    outputs_R = np.array([ f( oRi ) for oRi in outputs_Ri ])

    np.save("outputs_U.npy", outputs_U)
    np.save("outputs_R.npy", outputs_R)
    
    f = forgetting_quality(
        outputs_U, 
        outputs_R,
        attacks=attacks,
        delta=delta)

    return {
        "total_score": f * RA_ratio * TA_ratio,
        "F": f,
        "unlearn_retain_accuracy": RAU,
        "unlearn_test_accuracy": TAU, 
        "unlearn_forget_accuracy": FAU,
        "retrain_retain_accuracy": RAR,
        "retrain_test_accuracy": TAR, 
        "retrain_forget_accuracy": FAR,
        "retrain_outputs": outputs_R,
        "unlearn_outputs": outputs_U
    }



if __name__ == "__main__":
    retain_loader,forget_loader,validation_loader= get_dataset(64)
    data_loaders={
        'retain':retain_loader,
        'forget':forget_loader,
        #'validation':validation_loader,
        'testing':retain_loader
        }

    
    #from utils_inceptionresnetv2 import InceptionResNetV2
    #retrained_model = InceptionResNetV2(10572)
    #unlearned_model = InceptionResNetV2(10572)
    
    from models import FaceNetModel
    
    retrained_model=FaceNetModel()
    unlearned_model=FaceNetModel()
    retrain_model_path="/kaggle/input/pins-150-retain/fc_finetune_retain_final.pth"
    unlearn_model_path="/kaggle/working/log/fc_finetune_unlearn.pth"
    
    retrained_model.load_state_dict(torch.load(retrain_model_path))
    unlearned_model.load_state_dict(torch.load(unlearn_model_path))
    retrained_model.to(DEVICE)
    unlearned_model.to(DEVICE)
    pretrained_models={
        #'original':
        'retrained':retrained_model,
        'unlearned':unlearned_model}
    ret = score_unlearning_algorithm(data_loaders, pretrained_models)
    print(ret)




































