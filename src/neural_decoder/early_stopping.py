import torch
import os

class EarlyStopping:
    def __init__(self, patience=5, delta=0, output_dir=os.getcwd()):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        self.output_dir = output_dir

    def __call__(self, val_loss, model):
        score = -val_loss

        state_dict = model.state_dict()
        # Filter out unwanted keys before saving
        filtered_state_dict = {k: v for k, v in state_dict.items() if "drop_masks" not in k}

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = filtered_state_dict
            torch.save(self.best_model_state, self.output_dir + "/modelWeights") # Save the model weights
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = filtered_state_dict
            torch.save(self.best_model_state, self.output_dir + "/modelWeights") # Save the model weights
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)