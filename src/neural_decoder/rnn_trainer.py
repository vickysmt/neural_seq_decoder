import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder
from .model import LSTMDecoder
from .dataset import SpeechDataset

import wandb

from .early_stopping import EarlyStopping
from tqdm import tqdm 


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        # X, y, X_lens, y_lens, days, idx = zip(*batch)
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days)
            # idx
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData



def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    num_epochs = args["nEpochs"]

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    # Select model type based on args['model_type']
    if args["model_type"] == "GRU":
        model_class = GRUDecoder
    elif args["model_type"] == "LSTM":
        model_class = LSTMDecoder
    else:
        raise ValueError(f"Unknown model type: {args['model_type']}")

    model = model_class(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)
    
    # Compile model with TorchScript (Torch JIT)
    model = torch.jit.script(model)

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nEpochs"],
    )

    early_stopping = EarlyStopping(patience=args["earlyStop_patience"], delta=args["earlyStop_minChange"], output_dir=args["outputDir"])

    for epoch in range(num_epochs):
        model.train()

        # --train--
        trainLoss = 0
        startTime = time.time()

        # for X, y, X_len, y_len, dayIdx, idx in trainLoader:
        # for batch_idx, (X, y, X_len, y_len, dayIdx, idx) in enumerate(tqdm(trainLoader, desc=f'Epoch {epoch+1}/{num_epochs}', ncols=100)):
        for batch_idx, (X, y, X_len, y_len, dayIdx) in enumerate(tqdm(trainLoader, desc=f'Epoch {epoch+1}/{num_epochs}', ncols=100)):
            X, y, X_len, y_len, dayIdx = (
                X.to(device),
                y.to(device),
                X_len.to(device),
                y_len.to(device),
                dayIdx.to(device),
            )

            # Noise augmentation is faster on GPU
            if args["whiteNoiseSD"] > 0:
                X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

            if args["constantOffsetSD"] > 0:
                X += (
                    torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                    * args["constantOffsetSD"]
                )

            # Compute prediction error
            pred = model.forward(X, dayIdx)
            # print(f"Shape of pred = {pred.shape}")

            loss = loss_ctc(
                torch.permute(pred.log_softmax(2), [1, 0, 2]),
                y,
                ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                y_len,
            )
            # loss = torch.sum(loss) # Loss is a scalar so no need to sum
            trainLoss += loss.cpu().detach().numpy()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Compute gradient norm
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5  # L2 norm of all gradients

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args['gradClipVal'])
            
            
            # Update parameters
            optimizer.step()
            scheduler.step()

            # Print training loss every 100 batches
            # if batch_idx % 100 == 0:
            #     print(f'Batch {batch_idx}, Train CTC Loss: {loss.cpu().detach().numpy():>7f}')

            # print(endTime - startTime)
        trainLoss /= len(trainLoader)
        

        # Eval
        valLoss = 0
        # valCER = []           
        with torch.no_grad():
            model.eval()
            # allLoss = []
            total_edit_distance = 0
            total_seq_length = 0
            # for X, y, X_len, y_len, testDayIdx, idx in testLoader:
            for X, y, X_len, y_len, testDayIdx in testLoader:
                X, y, X_len, y_len, testDayIdx = (
                    X.to(device),
                    y.to(device),
                    X_len.to(device),
                    y_len.to(device),
                    testDayIdx.to(device),
                )

                pred = model.forward(X, testDayIdx)
                val_loss = loss_ctc(
                    torch.permute(pred.log_softmax(2), [1, 0, 2]),
                    y,
                    ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                    y_len,
                )
                # loss = torch.sum(loss) # No need for sum since loss is already a scalar

                valLoss += val_loss.cpu().detach().numpy()

            adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                torch.int32
            )
            for iterIdx in range(pred.shape[0]):
                # decodedSeq = torch.argmax(
                #     torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                #     dim=-1,
                # )  # [num_seq,]
                decodedSeq = torch.argmax(pred[iterIdx, 0 : adjustedLens[iterIdx], :].clone().detach(), dim=-1) # torch.Size([adjusted_lens[iterIdx]]) i.e., shape (window_size,)

                decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1) # Keep unique phonemes only
                decodedSeq = decodedSeq.cpu().detach().numpy()
                decodedSeq = np.array([i for i in decodedSeq if i != 0])  # Get rid of phoneme index 0 since 0 is just the padding

                trueSeq = np.array(
                    y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                )

                matcher = SequenceMatcher(
                    a=trueSeq.tolist(), b=decodedSeq.tolist()
                )
                total_edit_distance += matcher.distance()
                total_seq_length += len(trueSeq)

        
        valLoss /= len(testLoader)
        cer = total_edit_distance / total_seq_length
        # valCER.append(cer)

        endTime = time.time()
        print(
            f"epoch {epoch}, train ctc loss: {trainLoss:>7f}, val ctc loss: {valLoss:>7f}, val cer: {cer:>7f}, grad norm: {total_grad_norm:>7f} time/epoch: {(endTime - startTime):>7.3f} seconds"
        )
        startTime = time.time()

        # if len(valCER) > 0 and cer < np.min(valCER):
        #     torch.save(model.state_dict(), args["outputDir"] + "/modelWeights") # Save the model weights if the validation CER is reduced

        # Check if valLoss has reduced for early stopping
        early_stopping(valLoss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        # testLoss.append(valAvgDayLoss)
        # testCER.append(cer)

        # tStats = {}
        # tStats["testLoss"] = np.array(testLoss)
        # tStats["testCER"] = np.array(testCER)
        # tStats["trainLoss"] = np.array(trainLoss)

        # Log on wandb
        wandb.log({"valLoss":valLoss,
                    "valCER": cer,
                    "trainLoss": trainLoss,
                    "gradNorm": total_grad_norm})

        # with open(args["outputDir"] + "/trainingStats", "wb") as file:
        #     pickle.dump(tStats, file)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)


    # Select model type based on args['model_type']
    if args["model_type"] == "GRU":
        model_class = GRUDecoder
    elif args["model_type"] == "LSTM":
        model_class = LSTMDecoder
    else:
        raise ValueError(f"Unknown model type: {args['model_type']}")

    model = model_class(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()