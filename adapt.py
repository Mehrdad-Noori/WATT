import numpy as np

import clip
import torch
import torch.optim as optim
from tqdm import tqdm
import logging
from collections import OrderedDict
import copy
import torch.nn as nn

import configuration
from models import tent
from utils import prepare_dataset


def setup_tent(model, name_model, niter = 10, method = 'clip'):
    """Set up tent adaptation.
    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model.visual = tent.configure_model(model.visual, name_model)
    params, param_names = tent.collect_params(model.visual, name_model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=niter,  ### Iterations
                           method=method,
                           episodic=True)
    return tent_model


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.
    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.
    For best results, try tuning the learning rate and batch size.
    """
    # if cfg.OPTIM.METHOD == 'Adam':
    return optim.Adam(params,
                lr=1e-3,
                betas=(0.9, 0.999),
                weight_decay=0.0)


def weight_average(alltheweights):
    K = len(alltheweights)
    avg_state_dict = OrderedDict()
    for param_name, param in alltheweights[0].items():
        avg_param = sum(sd[param_name] for sd in alltheweights)
        avg_param /= K
        avg_state_dict[param_name] = avg_param
    return avg_state_dict


def text_emb_ensemble(model, classnames, templates, K=None):
    with torch.no_grad():
        weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if K == None:
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings /= class_embeddings.norm()
            weights.append(class_embeddings)
        weights = torch.stack(weights, dim=1).cuda()
    return weights

# Argues
args = configuration.argparser()
logger = logging.getLogger(__name__)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

base_model, preprocess = clip.load(args.model, device)
model = setup_tent(base_model, args.model, niter=args.niter, method = args.method)

common_corruptions = [args.corruption]
fichier = open('Results/' + args.dataset + '_' + args.model.replace('/','') + '.txt', 'w')
template = ['a photo of a {}', 'itap of a {}','a bad photo of the {}.', 'a origami {}.','a photo of the large {}.','a {} in a video game.','art of the {}.','a photo of the small {}.']
K = len(template)

for cor in common_corruptions:
    # Ecrit = ''
    args.corruption = cor
    validation = args.exps
    # Download the dataset
    teloader, _, teset = prepare_dataset.prepare_test_data(args, transform=preprocess if args.dataset == 'visda' else None)
    if cor == 'cifar_new' and args.dataset != 'visda':
        args.corruption = 'original'
        _, _, teset = prepare_dataset.prepare_test_data(args)
    acc = []
    for _ in range(validation):
        correct = 0
        for batch_idx, (inputs, labels) in tqdm(enumerate(teloader)):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            text_features = text_emb_ensemble(model.model, teset.classes, template, K=K)
            text_ensemble = False

            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")
            if args.adapt:
                Y = model(inputs, text_features)  # infer and adapt

            if not args.validate_text_features:
                temp = 'a photo of a {}'
                text_inputs = torch.cat([clip.tokenize(temp.format(c)) for c in teset.classes]).to(device)
            # Calculate features
            with torch.no_grad():
                image_features = model.model.encode_image(inputs)
                if not args.validate_text_features:
                    text_features = model.model.encode_text(text_inputs)

            # Pick the top 5 most similar labels for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if not args.validate_text_features:
                text_features /= text_features.norm(dim=-1, keepdim=True)
            else:
                text_features = text_emb_ensemble(model.model, teset.classes, template)
                text_features = text_features.T
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, pred = similarity.topk(1, 1, True, True)
            pred = pred.t()
            correctness = pred.eq(labels.view(1, -1).expand_as(pred))
            correct += correctness.sum().item()

        acc.append(correct / len(teloader.dataset))
    print(str(round(np.array(acc).mean()*100,2)) + ',' + str(round(np.array(acc).std()*100,2)))
    fichier.write(str(round(np.array(acc).mean()*100,2)) + ',' + str(round(np.array(acc).std()*100,2)) + '\n')
