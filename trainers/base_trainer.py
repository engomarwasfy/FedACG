from pathlib import Path
from typing import Callable, Dict, Tuple, Union, List, Type, Any
from argparse import Namespace
from collections import defaultdict

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import tqdm
import wandb
import gc

import pickle, os
import numpy as np

import logging
logger = logging.getLogger(__name__)

import time, io, copy

from trainers.build import TRAINER_REGISTRY

from servers import Server
from clients import Client

from utils import DatasetSplit, DatasetSplitSubset, get_dataset, PoisonedDatasetSplit
from utils.logging_utils import AverageMeter

from torch.utils.data import DataLoader

from utils import terminate_processes, initalize_random_seed, save_checkpoint
from omegaconf import DictConfig,OmegaConf


#from netcal.metrics import ECE
import matplotlib.pyplot as plt



@TRAINER_REGISTRY.register()
class Trainer():

    def __init__(self,
                 model: nn.Module,
                 client_type: Type,
                 server: Server,
                 evaler_type: Type,
                 datasets: Dict,
                 device: torch.device,
                 args: DictConfig,
                 multiprocessing: Dict = None,
                 **kwargs) -> None:

        self.args = args
        self.device = device
        self.model = model

        self.checkpoint_path = Path(self.args.checkpoint_path)
        mode = self.args.split.mode 
        if self.args.split.mode == 'dirichlet':
            mode += str(self.args.split.alpha)
        self.exp_path = self.checkpoint_path / self.args.dataset.name / mode / self.args.exp_name
        logger.info(f"Exp path : {self.exp_path}")

        ### training config
        trainer_args = self.args.trainer
        self.num_clients = trainer_args.num_clients
        self.participation_rate = trainer_args.participation_rate
        self.global_rounds = trainer_args.global_rounds
        self.lr = trainer_args.local_lr
        self.local_lr_decay = trainer_args.local_lr_decay


        self.clients: List[Client] = [client_type(self.args, client_index=c, model=copy.deepcopy(self.model)) for c in range(self.args.trainer.num_clients)]
        self.server = server
        if self.args.server.momentum > 0 or self.args.client.get('Dyn'):
            self.server.set_momentum(self.model)

        self.datasets = datasets
        self.local_dataset_split_ids = get_dataset(self.args, self.datasets['train'], mode=self.args.split.mode)

        test_loader = DataLoader(self.datasets["test"],
                                batch_size=args.evaler.batch_size if args.evaler.batch_size > 0 else args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        eval_device = self.device if not self.args.multiprocessing else torch.device(f'cuda:{self.args.main_gpu}')
        eval_params = {
            "test_loader": test_loader,
            "device": eval_device,
            "args": args,
        }
        self.eval_params = eval_params
        self.eval_device = eval_device
        self.evaler = evaler_type(**eval_params)
        logger.info(f"Trainer: {self.__class__}, client: {client_type}, server: {server.__class__}, evaler: {evaler_type}")

        self.start_round = 0
        if self.args.get('load_model_path'):
            self.load_model()

        if self.args.client.get('Dyn'):
            local_g = copy.deepcopy(self.model.state_dict())
            for key in local_g.keys():
                local_g[key] = torch.zeros_like(local_g[key]).to('cpu')
            self.past_local_deltas = {net_i: copy.deepcopy(local_g) for net_i in range(self.num_clients)}

        # Add tracking for poisoning effect over time
        self.poisoning_history = {
            'epoch': [],
            'non_poisoned_acc': [],
            'poisoned_acc': [],
            'acc_difference': [],
            'cat_acc_non_poisoned': [],
            'dog_acc_non_poisoned': [],
            'cat_acc_poisoned': [],
            'dog_acc_poisoned': []
        }

    def local_update(self, device, task_queue, result_queue):
        if self.args.multiprocessing:
            torch.cuda.set_device(device)
            initalize_random_seed(self.args)

        while True:
            task = task_queue.get()
            if task is None:
                break
            client = self.clients[task['client_idx']]

            # Use PoisonedDatasetSplit for clients 1 and 3
            if task['client_idx'] in [1, 3]:
                local_dataset = PoisonedDatasetSplit(
                    self.datasets['train'],
                    idxs=self.local_dataset_split_ids[task['client_idx']],
                    client_id=task['client_idx']
                )
            else:
                local_dataset = DatasetSplitSubset(
                    self.datasets['train'],
                    idxs=self.local_dataset_split_ids[task['client_idx']],
                    subset_classes=self.args.dataset.get('subset_classes'),
                )

            setup_inputs = {
                'state_dict': task['state_dict'],
                'device': device,
                'local_dataset': local_dataset,
                'local_lr': task['local_lr'],
                'global_epoch': task['global_epoch'],
                'trainer': self,
            }
            if self.args.client.get('Dyn'):
                setup_inputs['past_local_deltas'] = self.past_local_deltas
                setup_inputs['user'] = task['client_idx']
            client.setup(**setup_inputs)
            # Local Training
            local_model, local_loss_dict = client.local_train(global_epoch=task['global_epoch'])
            result_queue.put((local_model, local_loss_dict))
            if not self.args.multiprocessing:
                break

    def train(self) -> Dict:

        result_queue = mp.Manager().Queue()

        M = max(int(self.participation_rate * self.num_clients), 1)

        if self.args.multiprocessing:
            ngpus_per_node = torch.cuda.device_count()
            task_queues = [mp.Queue() for _ in range(M)]
            processes = [mp.get_context('spawn').Process(target=self.local_update, args=(
                i % ngpus_per_node, task_queues[i], result_queue)) for i in range(M)]

            # start all processes
            for p in processes:
                p.start()


        for epoch in range(self.start_round, self.global_rounds):

            self.lr_update(epoch=epoch)

            global_state_dict = copy.deepcopy(self.model.state_dict())
            prev_model_weight = copy.deepcopy(self.model.state_dict())
            
            # Select clients
            if self.participation_rate < 1.:
                selected_client_ids = np.random.choice(range(self.num_clients), M, replace=False)
            else:
                selected_client_ids = range(len(self.clients))
            logger.info(f"Global epoch {epoch}, Selected client : {selected_client_ids}")

            current_lr = self.lr

            local_weights = defaultdict(list)
            local_loss_dicts = defaultdict(list)
            local_deltas = defaultdict(list)

            local_models = []

            # FedACG lookahead momentum
            if self.args.server.get('FedACG'):
                assert(self.args.server.momentum > 0)
                self.model= copy.deepcopy(self.server.FedACG_lookahead(copy.deepcopy(self.model)))
                global_state_dict = copy.deepcopy(self.model.state_dict())

            # Client-side
            start = time.time()
            for i, client_idx in enumerate(selected_client_ids):
                task_queue_input = {
                    'state_dict': self.model.state_dict(),
                    'client_idx': client_idx,
                    'local_lr': current_lr,
                    'global_epoch': epoch,
                }
                if self.args.multiprocessing:
                    task_queues[i].put(task_queue_input)
                else:
                    task_queue = mp.Queue()
                    task_queue.put(task_queue_input)
                    self.local_update(self.device, task_queue, result_queue)

                    local_state_dict, local_loss_dict = result_queue.get()
                    for loss_key in local_loss_dict:
                        local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                    local_models.append(local_state_dict)

                    for param_key in local_state_dict:
                        local_weights[param_key].append(local_state_dict[param_key])
                        local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])


            if self.args.multiprocessing:
                for _ in range(len(selected_client_ids)):
                    # Retrieve results from the queue
                    result = result_queue.get()
                    local_state_dict, local_loss_dict = result
                    for loss_key in local_loss_dict:
                        local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                    local_models.append(local_state_dict)

                    # If you want to save gpu memory, make sure that weights are not allocated to GPU
                    for param_key in local_state_dict:
                        local_weights[param_key].append(local_state_dict[param_key])
                        local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])

            logger.info(f"Global epoch {epoch}, Train End. Total Time: {time.time() - start:.2f}s")


            # Server-side
            updated_global_state_dict = self.server.aggregate(local_weights, local_deltas,
                                                              selected_client_ids, copy.deepcopy(global_state_dict), current_lr)
            self.model.load_state_dict(updated_global_state_dict)

            local_datasets = [DatasetSplit(self.datasets['train'], idxs=self.local_dataset_split_ids[client_id]) for client_id in selected_client_ids]

            # Logging
            wandb_dict = {loss_key: np.mean(local_loss_dicts[loss_key]) for loss_key in local_loss_dicts}
            wandb_dict['lr'] = self.lr
            
            model_device = next(self.model.parameters()).device
            if self.args.eval.freq > 0 and epoch % self.args.eval.freq == 0:
                self.evaluate(epoch=epoch, local_datasets=local_datasets)

            if (self.args.save_freq > 0 and (epoch + 1) % self.args.save_freq == 0) or (epoch + 1 == self.args.trainer.global_rounds):
                self.save_model(epoch=epoch)

            self.wandb_log(wandb_dict, step=epoch)
            gc.collect()


        if self.args.multiprocessing:
            # Terminate Processes
            terminate_processes(task_queues, processes)

        # Print poisoning statistics
        self._print_poisoning_stats()
        return

    def lr_update(self, epoch: int) -> None:
        self.lr = self.args.trainer.local_lr * (self.local_lr_decay) ** (epoch)
        return
    

    def save_model(self, epoch: int = -1, suffix: str = '') -> None:
        
        model_path = self.exp_path / self.args.output_model_path
        if not model_path.parent.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)

        if epoch < self.args.trainer.global_rounds - 1:
            model_path = Path(f"{model_path}.e{epoch+1}")

        if suffix:
            model_path = Path(f"{model_path}.{suffix}")
        
        save_checkpoint(self.model, model_path, epoch, save_torch=True, use_breakpoint=False)      
        print(f'Saved model at {model_path}')  
        return
    

    def load_model(self) -> None:
        if self.args.get('load_model_path'):
            saved_dict = torch.load(self.args.load_model_path)
            self.model.load_state_dict(saved_dict['model_state_dict'], strict=False)
            self.start_round = saved_dict["epoch"]+1
            logger.warning(f'Load model from {self.args.load_model_path}, epoch {saved_dict["epoch"]}')
            
        return


    def wandb_log(self, log: Dict, step: int = None):
        if self.args.wandb:
            wandb.log(log, step=step)

    def validate(self, epoch: int, ) -> Dict:
        return

    def evaluate(self, epoch: int, local_datasets: List[torch.utils.data.Dataset] = None) -> Dict:
        # Save current model state
        current_model = copy.deepcopy(self.model)
        
        # Get overall accuracy first
        results = self.evaler.eval(model=copy.deepcopy(self.model), epoch=epoch)
        total_acc = results["acc"]
        
        # Evaluate non-poisoned performance
        print("\nEvaluating NON-POISONED model performance...")
        non_poisoned_results = self._evaluate_cat_dog_accuracy(is_poisoned=False)
        
        # Restore model and evaluate poisoned performance
        self.model = current_model
        print("\nEvaluating POISONED model performance...")
        poisoned_results = self._evaluate_cat_dog_accuracy(is_poisoned=True)
        
        # Calculate performance difference
        acc_diff = poisoned_results['overall_acc'] - non_poisoned_results['overall_acc']
        
        # Update poisoning history
        self.poisoning_history['epoch'].append(epoch)
        self.poisoning_history['non_poisoned_acc'].append(non_poisoned_results['overall_acc'])
        self.poisoning_history['poisoned_acc'].append(poisoned_results['overall_acc'])
        self.poisoning_history['acc_difference'].append(acc_diff)
        self.poisoning_history['cat_acc_non_poisoned'].append(non_poisoned_results['cat_acc'])
        self.poisoning_history['dog_acc_non_poisoned'].append(non_poisoned_results['dog_acc'])
        self.poisoning_history['cat_acc_poisoned'].append(poisoned_results['cat_acc'])
        self.poisoning_history['dog_acc_poisoned'].append(poisoned_results['dog_acc'])
        
        print("\nOverall Performance Comparison:")
        print("="*50)
        print(f"Total Model Accuracy (All Classes): {total_acc:.2f}%")
        print(f"Non-poisoned Accuracy (Cat/Dog): {non_poisoned_results['overall_acc']:.2f}%")
        print(f"Poisoned Accuracy (Cat/Dog): {poisoned_results['overall_acc']:.2f}%")
        print(f"Accuracy Difference: {acc_diff:+.2f}%")
        print("\nClass-wise Performance Change (Cat/Dog):")
        print(f"Cat Accuracy: {non_poisoned_results['cat_acc']:.2f}% -> {poisoned_results['cat_acc']:.2f}%")
        print(f"Dog Accuracy: {non_poisoned_results['dog_acc']:.2f}% -> {poisoned_results['dog_acc']:.2f}%")
        print("="*50 + "\n")

        # Keep original logging
        logger.warning(f'[Epoch {epoch}] Test Accuracy: {total_acc:.2f}%')
        plt.close()
        
        wandb_dict = {
            f"acc/{self.args.dataset.name}": total_acc,
            "poisoning/non_poisoned_acc": non_poisoned_results['overall_acc'],
            "poisoning/poisoned_acc": poisoned_results['overall_acc'],
            "poisoning/acc_difference": acc_diff,
            "poisoning/cat_acc_non_poisoned": non_poisoned_results['cat_acc'],
            "poisoning/dog_acc_non_poisoned": non_poisoned_results['dog_acc'],
            "poisoning/cat_acc_poisoned": poisoned_results['cat_acc'],
            "poisoning/dog_acc_poisoned": poisoned_results['dog_acc']
        }
        
        self.wandb_log(wandb_dict, step=epoch)
        return {
            "acc": total_acc,
            "non_poisoned_acc": non_poisoned_results['overall_acc'],
            "poisoned_acc": poisoned_results['overall_acc'],
            "acc_difference": acc_diff,
            "poisoning_history": self.poisoning_history
        }

    def _evaluate_cat_dog_accuracy(self, is_poisoned=True) -> Dict:
        """Evaluate accuracy specifically on cat and dog classes."""
        # Move model to the correct device
        self.model = self.model.to(self.eval_device)
        self.model.eval()
        correct = 0
        total = 0
        cat_class = 3
        dog_class = 5
        
        # Track predictions and correct predictions per class
        predictions = {cat_class: 0, dog_class: 0}
        actual = {cat_class: 0, dog_class: 0}
        correct_predictions = {cat_class: 0, dog_class: 0}
        
        # Initialize confusion matrix
        confusion_matrix = {
            cat_class: {cat_class: 0, dog_class: 0},
            dog_class: {cat_class: 0, dog_class: 0}
        }
        
        with torch.no_grad():
            for images, labels in self.evaler.test_loader:
                images, labels = images.to(self.eval_device), labels.to(self.eval_device)
                
                # Only evaluate on cat and dog images
                mask = (labels == cat_class) | (labels == dog_class)
                if not mask.any():
                    continue
                    
                images = images[mask]
                labels = labels[mask]
                outputs = self.model(images)
                _, predicted = outputs["logit"].max(1)
                
                # Keep original accuracy calculation
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Additional tracking for detailed analysis
                for label, pred in zip(labels, predicted):
                    label_item = label.item()
                    pred_item = pred.item()
                    actual[label_item] += 1
                    if pred_item in [cat_class, dog_class]:
                        predictions[pred_item] += 1
                        if label_item == pred_item:
                            correct_predictions[label_item] += 1
                        confusion_matrix[label_item][pred_item] += 1
        
        # Calculate class-wise accuracy
        cat_accuracy = 100. * correct_predictions[cat_class] / actual[cat_class] if actual[cat_class] > 0 else 0
        dog_accuracy = 100. * correct_predictions[dog_class] / actual[dog_class] if actual[dog_class] > 0 else 0
        overall_accuracy = 100. * correct / total if total > 0 else 0
        
        # Print detailed analysis
        status = "POISONED" if is_poisoned else "NON-POISONED"
        print(f"\nDetailed Analysis for Cat/Dog Classes ({status}):")
        print("="*50)
        print("Class-wise Accuracy:")
        print(f"Cat Class Accuracy: {cat_accuracy:.2f}%")
        print(f"Dog Class Accuracy: {dog_accuracy:.2f}%")
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")
        print("\nPrediction Distribution:")
        print(f"Actual Cats: {actual[cat_class]}, Predicted as Cats: {predictions[cat_class]}")
        print(f"Actual Dogs: {actual[dog_class]}, Predicted as Dogs: {predictions[dog_class]}")
        print("\nConfusion Matrix:")
        print("                Predicted")
        print("Actual    Cat    Dog")
        print(f"Cat      {confusion_matrix[cat_class][cat_class]:<6d} {confusion_matrix[cat_class][dog_class]:<6d}")
        print(f"Dog      {confusion_matrix[dog_class][cat_class]:<6d} {confusion_matrix[dog_class][dog_class]:<6d}")
        print("\nMisclassification Analysis:")
        print(f"Cat->Dog misclassifications: {confusion_matrix[cat_class][dog_class]}")
        print(f"Dog->Cat misclassifications: {confusion_matrix[dog_class][cat_class]}")
        print("="*50 + "\n")
        
        # Move model back to CPU
        self.model = self.model.to('cpu')
                
        return {
            'overall_acc': overall_accuracy,
            'cat_acc': cat_accuracy,
            'dog_acc': dog_accuracy,
            'confusion_matrix': confusion_matrix
        }

    def _print_poisoning_stats(self):
        """Print statistics about the poisoning attack."""
        print("\n" + "="*50)
        print("POISONING ATTACK STATISTICS")
        print("="*50)
        
        # Get poisoned clients' datasets
        poisoned_clients = [1, 3]
        total_poisoned = 0
        
        for client_idx in poisoned_clients:
            if client_idx < len(self.clients):
                dataset = self.clients[client_idx].loader.dataset
                if hasattr(dataset, 'poisoned_count'):
                    total_poisoned += dataset.poisoned_count
                    print(f"Client {client_idx}: Poisoned {dataset.poisoned_count} labels")
        
        print(f"\nTotal labels poisoned: {total_poisoned}")
        
        # Print final performance comparison
        if self.poisoning_history['epoch']:
            final_epoch = self.poisoning_history['epoch'][-1]
            print("\nFinal Performance Comparison:")
            print(f"Total Model Accuracy: {self.evaler.eval(model=self.model, epoch=final_epoch)['acc']:.2f}%")
            print(f"Non-poisoned Cat/Dog Accuracy: {self.poisoning_history['non_poisoned_acc'][-1]:.2f}%")
            print(f"Poisoned Cat/Dog Accuracy: {self.poisoning_history['poisoned_acc'][-1]:.2f}%")
            print(f"Final Accuracy Difference: {self.poisoning_history['acc_difference'][-1]:+.2f}%")
            
            print("\nClass-wise Final Performance:")
            print(f"Cat Accuracy: {self.poisoning_history['cat_acc_non_poisoned'][-1]:.2f}% -> {self.poisoning_history['cat_acc_poisoned'][-1]:.2f}%")
            print(f"Dog Accuracy: {self.poisoning_history['dog_acc_non_poisoned'][-1]:.2f}% -> {self.poisoning_history['dog_acc_poisoned'][-1]:.2f}%")
            
            # Calculate average impact
            avg_acc_diff = sum(self.poisoning_history['acc_difference']) / len(self.poisoning_history['acc_difference'])
            print(f"\nAverage Impact Across Training: {avg_acc_diff:+.2f}%")
        
        print("="*50 + "\n")




    