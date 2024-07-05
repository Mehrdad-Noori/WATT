import copy
from collections import OrderedDict

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.misc import load_templates_from_yaml

REFERENCE_TEMPLATE = 'a photo of a {}'

class WATT:
    """
    WATT (Weight Average adaptation during Test-Time) adapts a CLIP model by minimizing entropy during testing.
    The model adapts itself by updating on every forward pass.
    The code is based on TENT and CLIPArTT repos.
    TENT GitHub: https://github.com/DequanWang/tent
    CLIPArTT GitHub: https://github.com/dosowiechi/CLIPArTT
    """


    def __init__(self, model, lr, type='sequential', l=2, m=5, temps_dir='templates.yaml', ref_eval=False, device='cpu'):
        """
        Initializes the WATT module.

        Args:
            model: The CLIP model to be adapted.
            lr: Learning rate for the optimizer.
            type: Adaptation method of WATT ('parallel' or 'sequential').
            l: Number of adaptation iterations for each text embedding before performing weight averaging.
            m: Number of repetitions of the adaptation and weight averaging process.
            temps_dir: Path to the templates.yaml file which inclodes different text templates that should be used during adaptation.
            ref_eval: Whether to use REFERENCE_TEMPLATE during evaluation.
            device: The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        
        # loading the base model
        base_model, _ = clip.load(model, device) 
        self.model = base_model

        self.lr = lr
        self.type = type
        self.l = l
        self.m = m
        self.ref_eval = ref_eval
        self.device = device

        # Load the text templates
        self.all_templates = load_templates_from_yaml(temps_dir)

        # Set the gradients for LayerNorm layers only for visual encoder
        self.model.visual = self.set_ln_grads(self.model.visual)

        # Collect the LayerNorm parameters and set the optimizer
        params, _ = self.collect_ln_params(self.model.visual)
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)

        # Save the initial model and optimizer states
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)
        


    def adapt(self, x, classes):
        """
        Forward pass with adaptation.

        Args:
            x: Input image tensor.
            classes: List of class names.

        """

        self.reset()
        self.perform_adaptation(x, classes)


    @torch.no_grad()
    def evaluate(self, x, classes):
        """
        Forward pass without adaptation.

        Args:
            x: Input image tensor.
            classes: List of class names.

        Returns:
            pred: Predicted class labels for the input images.

        """

        # extracting features
        image_features = self.model.encode_image(x)

        # Pick the top most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        if self.ref_eval:
            text_features = self.extract_text_embeddings(classes, [REFERENCE_TEMPLATE], average=True)
        else:
            text_features = self.extract_text_embeddings(classes, self.all_templates, average=True)
        text_features = text_features.T

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, pred = similarity.topk(1, 1, True, True)
        pred = pred.t()
       
        return pred


    def reset(self):
        """
        Resets the model and optimizer to their initial states.
        """
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.model, self.optimizer,
                                      self.model_state, self.optimizer_state)


    def perform_adaptation(self, x, classes):
        """
        Forward pass with adaptation for test-time. The model adapts itself during testing by updating on every forward pass.

        Args:
            x: Input image tensor.
            classes: List of class names.
        """

        text_x = self.extract_text_embeddings(classes, self.all_templates, average=False)

        for m in range(self.m):
            all_weights = []
            if self.type == 'sequential':
                if m == 0:
                    self.load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
                else:
                    self.model.load_state_dict(avg_state_dict, strict=False)
                
            for text_feat in text_x:
                if self.type == 'parallel':
                    if m == 0:
                        self.load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
                    else:
                        self.model.load_state_dict(avg_state_dict, strict=False)

                for l in range(self.l):
                    with torch.no_grad():
                        image_features = self.model.encode_image(x)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    similarity = (100 * image_features @ text_feat.t()).softmax(1)
                    values, pred = similarity.topk(1, 1, True, True)
                    pred_inputs = torch.cat([text_feat[c,] for c in pred])

                    # Calculating the Loss
                    logits, image_features, text_features = self.model(x, pred_inputs, True)
                    images_similarity = image_features @ image_features.t()
                    texts_similarity = text_features @ text_features.t()
                    targets = F.softmax(((images_similarity + texts_similarity) / 2) / 0.01, dim=-1)
                    loss = self.cross_entropy(logits.t(), targets.t(), reduction='mean')
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                weights = {}
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.LayerNorm):
                        for nparam, p in module.named_parameters():
                            if nparam in ['weight', 'bias']:
                                weights[f"{name}.{nparam}"] = copy.deepcopy(p)
                all_weights.append(weights)
            avg_state_dict = self.weight_average(all_weights)
        self.model.load_state_dict(avg_state_dict, strict=False)


    
    def extract_text_embeddings(self, class_names, templates, average=True):  
        """
        Extracts text embeddings for given class names and templates.

        Args:
            class_names: List of class names to generate text embeddings for.
            templates: List of text templates to use for generating text embeddings.
            average: Boolean indicating whether to average the embeddings of different templates for each class.

        Returns:
            text_features: Tensor of text embeddings for the given class names and templates.
        """
        with torch.no_grad():
            text_features = []
            for class_name in class_names:
                texts = [template.format(class_name) for template in templates] 
                texts = clip.tokenize(texts).to(self.device) 
                class_embeddings = self.model.encode_text(texts) 
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                if average:
                    class_embeddings = class_embeddings.mean(dim=0)
                    class_embeddings /= class_embeddings.norm()
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=1).to(self.device)
        return text_features


    @staticmethod
    def set_ln_grads(model):
        """
        Set gradient settings for LayerNorm layers within the model, disabling gradients globally except for these LN layers.

        Args:
            model: The model whose LayerNorm layers' gradients are to be set.
        
        Returns:
            The model with modified gradient settings.
        """
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        return model


    @staticmethod
    def collect_ln_params(model):
        """
        Collect the affine scale and shift parameters from LayerNorm layers.

        Args:
            model: The model from which to collect LayerNorm parameters.
        
        Returns:
            params: List of LayerNorm parameters.
            names: List of parameter names.
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"visual.{nm}.{np}")
        return params, names


    @staticmethod
    def cross_entropy(preds, targets, reduction='none'):
        """
        Calculate the cross-entropy loss between predictions and targets.

        Args:
            preds: Predicted logits.
            targets: Target probabilities.
            reduction: Type of reduction to apply to the output ('none' or 'mean').

        Returns:
            The computed loss.
        """
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()


    @staticmethod
    def weight_average(all_weights):
        """
        Compute the average of the weights from multiple models.

        Args:
            all_weights: List of state dictionaries from different models.

        Returns:
            avg_state_dict: Averaged state dictionary.
        """
        K = len(all_weights)
        avg_state_dict = OrderedDict()
        for param_name, param in all_weights[0].items():
            avg_param = sum(sd[param_name] for sd in all_weights) / K
            avg_state_dict[param_name] = avg_param
        return avg_state_dict


    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        """
        Copy the model and optimizer states for resetting after adaptation.

        Args:
            model: The model to copy.
            optimizer: The optimizer to copy.

        Returns:
            model_state: Copied state of the model.
            optimizer_state: Copied state of the optimizer.
        """
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        return model_state, optimizer_state


    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        """
        Restore the model and optimizer states from copies.

        Args:
            model: The model to restore.
            optimizer: The optimizer to restore.
            model_state: The state to restore the model to.
            optimizer_state: The state to restore the optimizer to.
        """
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)