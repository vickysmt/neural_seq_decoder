import torch
from torch import nn

from .augmentations import GaussianSmoothing
# from speechbrain.nnet.RNN import SLiGRU
# from speechbrain.nnet.RNN import QuasiRNN

from .speechbrain_rnn import SLiGRU
from .speechbrain_rnn import QuasiRNN

class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
        ff_normalization = "batchnorm",
    ):
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim # Number of channels
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.ff_normalization = ff_normalization
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # Adjust hidden dimension based on bidirectional or not
        if self.bidirectional:
            self.layer_dim_hidden = self.layer_dim * 2
            # Normalization
            if self.ff_normalization == "batchnorm":
                self.batch_norm = nn.BatchNorm1d(self.hidden_dim*2)

            elif self.ff_normalization == "layernorm":
                self.layer_norm = nn.LayerNorm(self.hidden_dim*2)
        else:
            self.layer_dim_hidden = self.layer_dim
            # Normalization
            if self.ff_normalization == "batchnorm":
                self.batch_norm = nn.BatchNorm1d(self.hidden_dim)

            elif self.ff_normalization == "layernorm":
                self.layer_norm = nn.LayerNorm(self.hidden_dim)
        

        # GRU layers
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)
        # print(f"Shape of transformedNeural = {transformedNeural.shape}")

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # apply RNN layer
        h0 = torch.zeros(
            self.layer_dim_hidden,
            transformedNeural.size(0),
            self.hidden_dim,
            device=self.device,
        ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())
        # print(f"Shape of hid = {hid.shape}")

        # Normalization
        if self.ff_normalization == "batchnorm":
            hid_reshaped = torch.permute(hid, (0, 2, 1)) # Changed from (batch_size, seq_len, num_channels) to (batch_size, num_channels, seq_len)
            hid_norm = self.batch_norm(hid_reshaped)
            hid_norm = torch.permute(hid_norm, (0, 2, 1)) # Changed back to (batch_size, seq_len, num_channels)

        elif self.ff_normalization == "layernorm":
            hid_norm = self.layer_norm(hid)
        else:
            hid_norm = hid

        # print(f"Shape of hid_norm = {hid_norm.shape}")

        # get seq
        seq_out = self.fc_decoder_out(hid_norm)
        return seq_out
    
class SLiGRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
        batch_size=32,
        ff_normalization = "batchnorm",
    ):
        super(SLiGRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.ff_normalization = ff_normalization
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        self.batch_size = batch_size

        print(f"Batch size = {self.batch_size}")
       

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # LiGRU layers
        self.ligru_decoder = SLiGRU(
            hidden_size=self.hidden_dim,
            input_shape=(self.batch_size, 1, (self.neural_dim) * self.kernelLen),
            num_layers=layer_dim,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            ff_normalization=self.ff_normalization,
        )

        for name, param in self.ligru_decoder.named_parameters():
            if "weight_hh" in name or ".u.weight" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # for name, param in self.ligru_decoder.named_parameters():
        #     if "weight_hh" in name or ".u.weight" in name:
        #         nn.init.orthogonal_(param)

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        # print(f"Shape of neuralInput = {neuralInput.shape}")
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)
        # print(f"Shape of transformedNeural = {transformedNeural.shape}")

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )
        # print(f"Shape of stridedInputs = {stridedInputs.shape}")

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        # print(f"Shape of h0 = {h0.shape}")

        hid, _ = self.ligru_decoder(stridedInputs, h0.detach())

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out
    
class QuasiRNNDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
        batch_size=16,
        ff_normalization = "batchnorm",
    ):
        super(QuasiRNNDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.ff_normalization = "layernorm"
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.ff_normalization = ff_normalization
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        self.batch_size = batch_size

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # # Normalization Adjustedbased on bidirectional or not
        if self.bidirectional:            
            if self.ff_normalization == "batchnorm":
                self.batch_norm = nn.BatchNorm1d(self.hidden_dim*2)

            elif self.ff_normalization == "layernorm":
                self.layer_norm = nn.LayerNorm(self.hidden_dim*2)
        else:
            if self.ff_normalization == "batchnorm":
                self.batch_norm = nn.BatchNorm1d(self.hidden_dim)

            elif self.ff_normalization == "layernorm":
                self.layer_norm = nn.LayerNorm(self.hidden_dim)

        # QuasiRNN layers
        self.quasirnn_decoder = QuasiRNN(
            hidden_size=self.hidden_dim,
            input_shape=(self.batch_size, 1, (self.neural_dim) * self.kernelLen),
            num_layers=layer_dim,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
        )

        for name, param in self.quasirnn_decoder.named_parameters():
            if "weight_hh" in name or ".u.weight" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # for name, param in self.quasirnn_decoder.named_parameters():
        #     if "weight_hh" in name or ".u.weight" in name:
        #         nn.init.orthogonal_(param)

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0)*2,
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        # print(f"Shape of h0 = {h0.shape}")
        hid, _ = self.quasirnn_decoder(stridedInputs, h0.detach())

         # Normalization
        if self.ff_normalization == "batchnorm":
            hid_reshaped = torch.permute(hid, (0, 2, 1)) # Changed from (batch_size, seq_len, num_channels) to (batch_size, num_channels, seq_len)
            hid_norm = self.batch_norm(hid_reshaped)
            hid_norm = torch.permute(hid_norm, (0, 2, 1)) # Changed back to (batch_size, seq_len, num_channels)

        elif self.ff_normalization == "layernorm":
            hid_norm = self.layer_norm(hid)
        else:
            hid_norm = hid

        # print(f"Shape of hid_norm = {hid_norm.shape}")

        # get seq
        seq_out = self.fc_decoder_out(hid_norm)
        return seq_out