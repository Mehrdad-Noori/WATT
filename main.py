import os
import torch
import argparse
import numpy as np
from tqdm import tqdm

from adapt import get_method
from utils import datasets 
from utils.misc import set_global_seeds, save_configuration


def argparser():
    parser = argparse.ArgumentParser("Weight Average Test Time Adaptation of CLIP")

    # Directories
    parser.add_argument('--data_dir', type=str, default='/export/livia/home/vision/Mnoori/data/', help='Root directory for datasets')
    parser.add_argument('--save_dir', type=str, default='save/', help='Path for saving base training weights and results')

    # General settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Model
    parser.add_argument('--backbone', type=str, default='ViT-B/32', help='Model backbone to use') 

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10', 'cifar100', 'tiny-imagenet', 'visda', 'PACS', 'office_home', 'VLCS'), help='Dataset to use')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers for data loading')

    # Training settings
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--trials', default=3, type=int, help='Number of trials to repeat the experiments')

    # Evaluation settings
    parser.add_argument('--adapt', action='store_true', help='Enable adaptation')

    # Corruptions settings
    parser.add_argument('--corruptions_list', nargs='+', default=None, type=str, help='List of corruptions to apply to the dataset (Cifar datasets)')

    # Method name
    parser.add_argument('--method', type=str, default='watt', choices=('watt'), help='Method to use for adaptation')

    return parser

def add_method_specific_args(parser, method):
    '''
    Add method-specific arguments to the parser
    '''
    if method == 'watt':
        parser.add_argument('--watt_type', type=str, default='sequential', choices=('parallel', 'sequential'), help='Type of WATT adaptation (parallel or sequential)')
        parser.add_argument('--watt_l', default=2, type=int, help='Number of adaptation iterations for each text embedding before weight averaging')
        parser.add_argument('--watt_m', default=5, type=int, help='Number of repetitions of the adaptation and weight averaging process')
        parser.add_argument('--watt_temps', type=str, default='templates.yaml', help='Path to the templates.yaml file')
        parser.add_argument('--watt_reference_for_evaluation', action='store_true', help='Use REFERENCE_TEMPLATE during evaluation instead of averaging text embeddings of different templates')

    # Add other methods here
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return parser

def main():
    # Initial argument parsing to get the method
    initial_parser = argparser()
    initial_args, _ = initial_parser.parse_known_args()

    # Create a new parser with method-specific arguments
    parser = argparser()
    parser = add_method_specific_args(parser, initial_args.method)
    args = parser.parse_args()

    # Set the global random seed for reproducibility
    set_global_seeds(args.seed)

    # Save the configuration settings
    save_configuration(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setting up the model and the method
    adapt_method = get_method(args, device)

    results_path = os.path.join(args.save_dir, "results.txt")

    for corruption in args.corruptions_list:
        data_loader, classes = datasets.prepare_data(args.dataset, args.data_dir, corruption=corruption, batch_size=args.batch_size, num_workers=args.workers)
        acc = []
        for t in range(args.trials):
            correct = 0
            for batch_idx, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # reset the model before adapting to a new batch
                adapt_method.reset()
                
                # perform adaptation
                if args.adapt:
                    adapt_method.adapt(inputs, classes)

                # perform evaluation 
                pred = adapt_method.evaluate(inputs, classes)

                # Calculate the number of correct predictions
                correctness = pred.eq(labels.view(1, -1).expand_as(pred))
                correct += correctness.sum().item()
                print(correct)

            acc.append(correct / len(data_loader.dataset))
            print(correct / len(data_loader.dataset))
        
        print(str(round(np.array(acc).mean()*100, 2)) + ',' + str(round(np.array(acc).std()*100, 2)))
        with open(results_path, 'w') as fichier:
            fichier.write(str(round(np.array(acc).mean()*100, 2)) + ',' + str(round(np.array(acc).std()*100, 2)) + '\n')

if __name__ == "__main__":
    main()
