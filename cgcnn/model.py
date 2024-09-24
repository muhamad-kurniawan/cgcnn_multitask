from __future__ import print_function, division

import torch
import torch.nn as nn
from collections.abc import Sequence


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1, output_nodes=[1],
                 tasks=['regression']):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        # self.classification = classification
        self.tasks = tasks
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        # self.conv_to_fc_softplus = nn.Softplus()
        # if n_h > 1:
        #     self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
        #                               for _ in range(n_h-1)])
        #     self.softpluses = nn.ModuleList([nn.Softplus()
        #                                      for _ in range(n_h-1)])

        self.activation = nn.ReLU()

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.acts = nn.ModuleList([nn.ReLU() for _ in range(n_h-1)])

        # self.heads = nn.ModuleList(
        #     ResidualNetworkOut(
        #         input_dim=h_fea_len,   # Input from the hidden layer
        #         output_dim=nodes,        # 2x output for mean and log_std
        #         hidden_layer_dims=[256, 128],         # Example hidden layers
        #         activation=nn.ReLU,                # Activation function
        #         batch_norm=True,                     # Use batch normalization
        #         task=task
        #     ) for nodes, task in zip(output_nodes, self.tasks)
        # )
        
        self.heads = nn.ModuleList(
            OutNetwork(
                input_dim=h_fea_len,   # Input from the hidden layer
                output_dim=nodes,        # 2x output for mean and log_std
                hidden_layer_dims=[256],         # Example hidden layers
                # activation=nn.ReLU,                # Activation function
                batch_norm=True,                     # Use batch normalization
                task=task
            ) for nodes, task in zip(output_nodes, self.tasks)
        )
                     
        # if self.classification:
        #     self.fc_out = nn.Linear(h_fea_len, 2)
        # else:
        #     self.fc_out = nn.Linear(h_fea_len, 1)
        # if self.classification:
        #     self.logsoftmax = nn.LogSoftmax(dim=1)
        #     self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.activation(self.conv_to_fc(crys_fea))
        # crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        # crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.tasks=='classification':
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        # out = self.fc_out(crys_fea)

        # outputs = []
        # for head in self.heads:
            
        #     outputs.append(head(h))

        # for head in self.heads:
        #     outputs.append(head(crys_fea))
        
        # if self.classification:
            # out = self.logsoftmax(out)
        return tuple(head(crys_fea) for head in self.heads)

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)

class ResidualNetworkOut(nn.Module):
    """Feed forward Residual Neural Network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_dims: Sequence[int],
        task,
        activation: type[nn.Module] = nn.ReLU,
        batch_norm: bool = False,      
    ) -> None:
        """Create a feed forward neural network with skip connections.

        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output features
            hidden_layer_dims (list[int]): List of hidden layer sizes
            activation (type[nn.Module], optional): Which activation function to use.
                Defaults to nn.LeakyReLU.
            batch_norm (bool, optional): Whether to use batch_norm. Defaults to False.
        """
        super().__init__()
        self.task = task
        dims = [input_dim, *list(hidden_layer_dims)]

        self.fcs = nn.ModuleList(
            nn.Linear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)
        )

        if batch_norm:
            self.bns = nn.ModuleList(
                nn.BatchNorm1d(dims[idx + 1]) for idx in range(len(dims) - 1)
            )
        else:
            self.bns = nn.ModuleList(nn.Identity() for _ in range(len(dims) - 1))

        self.res_fcs = nn.ModuleList(
            nn.Linear(dims[idx], dims[idx + 1], bias=False)
            if (dims[idx] != dims[idx + 1])
            else nn.Identity()
            for idx in range(len(dims) - 1)
        )
        self.acts = nn.ModuleList(activation() for _ in range(len(dims) - 1))

        self.fc_out = nn.Linear(dims[-1], output_dim)
        self.softplus = nn.Softplus()  # For regression
        if self.task=='classification':
            self.logsoftmax = nn.LogSoftmax(dim=1)  # For classification
        
    def forward(self, x):
        """Forward pass through network."""
        for fc, bn, res_fc, act in zip(self.fcs, self.bns, self.res_fcs, self.acts):
            x = act(bn(fc(x))) + res_fc(x)
        x = self.softplus(x)
        x = self.fc_out(x)
        
        if self.task=='classification':
            x = self.logsoftmax(x)  # LogSoftmax for classification tasks
            
        # return self.fc_out(x)
        return x

    def __repr__(self) -> str:
        input_dim = self.fcs[0].in_features
        output_dim = self.fc_out.out_features
        activation = type(self.acts[0]).__name__
        return f"{type(self).__name__}({input_dim=}, {output_dim=}, {activation=})"


class SimpleNetwork(nn.Module):
    """Simple Feed Forward Neural Network for multitask learning."""

    def __init__(
        self,
        input_dim: int,
        output_dim: Sequence[int],  # output_dims is now a list to handle multiple tasks
        hidden_layer_dims: Sequence[int],
        task: str,  # A list of tasks corresponding to output dimensions
        activation: type[nn.Module] = nn.LeakyReLU,
        batch_norm: bool = False,
    ) -> None:
        """Create a simple feed forward neural network for multitask learning.

        Args:
            input_dim (int): Number of input features
            output_dims (list[int]): List of output features for each task
            hidden_layer_dims (list[int]): List of hidden layer sizes
            tasks (list[str]): List of tasks ('classification' or 'regression')
            activation (type[nn.Module], optional): Which activation function to use.
                Defaults to nn.LeakyReLU.
            batch_norm (bool, optional): Whether to use batch_norm. Defaults to False.
        """
        super().__init__()

        self.task = task
        self.output_dim = output_dim

        dims = [input_dim, *list(hidden_layer_dims)]

        self.fcs = nn.ModuleList(
            nn.Linear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)
        )

        if batch_norm:
            self.bns = nn.ModuleList(
                nn.BatchNorm1d(dims[idx + 1]) for idx in range(len(dims) - 1)
            )
        else:
            self.bns = nn.ModuleList(nn.Identity() for _ in range(len(dims) - 1))

        self.acts = nn.ModuleList(activation() for _ in range(len(dims) - 1))

        # Define a separate output layer for each task
        self.fc_out = nn.Linear(dims[-1], output_dim) 

        # Add task-specific activations for classification tasks
        self.softplus = nn.Softplus()  # For regression
        self.logsoftmax = nn.LogSoftmax(dim=1) if self.task == 'classification' else nn.Identity()

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        """Forward pass through network for multitask learning."""
        for fc, bn, act in zip(self.fcs, self.bns, self.acts):
            x = act(bn(fc(x)))

        # Generate separate outputs for each task        
        out = self.fc_out(x)
            # if task == 'regression':
            #     out = self.softplus(out)
        if self.task == 'classification':
            out = logsoftmax(out)
       
        return out

    def reset_parameters(self) -> None:
        """Reinitialize network weights using PyTorch defaults."""
        for fc in self.fcs:
            fc.reset_parameters()

        for fc_out in self.fc_outs:
            fc_out.reset_parameters()

    def __repr__(self) -> str:
        input_dim = self.fcs[0].in_features
        output_dims = [fc_out.out_features for fc_out in self.fc_outs]
        activation = type(self.acts[0]).__name__
        return f"{type(self).__name__}({input_dim=}, {output_dims=}, {activation=})"

class OutNetwork(nn.Module):
    """Simple Feed Forward Neural Network for multitask learning."""

    def __init__(
        self,
        input_dim: int,
        output_dim: Sequence[int],  # output_dims is now a list to handle multiple tasks
        hidden_layer_dims: Sequence[int],
        task: str,  # A list of tasks corresponding to output dimensions
        activation: type[nn.Module] = nn.LeakyReLU,
        batch_norm: bool = False,
    ) -> None:
        """Create a simple feed forward neural network for multitask learning.

        Args:
            input_dim (int): Number of input features
            output_dims (list[int]): List of output features for each task
            hidden_layer_dims (list[int]): List of hidden layer sizes
            tasks (list[str]): List of tasks ('classification' or 'regression')
            activation (type[nn.Module], optional): Which activation function to use.
                Defaults to nn.LeakyReLU.
            batch_norm (bool, optional): Whether to use batch_norm. Defaults to False.
        """
        super().__init__()

        self.task = task
        self.output_dim = output_dim

        # dims = [input_dim, *list(hidden_layer_dims)]

        # self.fcs = nn.ModuleList(
        #     nn.Linear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)
        # )

        # if batch_norm:
        #     self.bns = nn.ModuleList(
        #         nn.BatchNorm1d(dims[idx + 1]) for idx in range(len(dims) - 1)
        #     )
        # else:
        #     self.bns = nn.ModuleList(nn.Identity() for _ in range(len(dims) - 1))

        # self.acts = nn.ModuleList(activation() for _ in range(len(dims) - 1))

        # Define a separate output layer for each task
        self.fc = nn.Linear(input_dim, hidden_layer_dims[0])
        self.fc_out = nn.Linear(hidden_layer_dims[0], output_dim) 

        # Add task-specific activations for classification tasks
        # self.softplus = nn.Softplus()  # For regression
        self.logsoftmax = nn.LogSoftmax(dim=1) if self.task == 'classification' else nn.Identity()

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        """Forward pass through network for multitask learning."""
        # for fc, bn, act in zip(self.fcs, self.bns, self.acts):
        #     x = act(bn(fc(x)))

        # Generate separate outputs for each task 
        x = self.fc(x)
        out = self.fc_out(x)
            # if task == 'regression':
            #     out = self.softplus(out)
        if self.task == 'classification':
            out = logsoftmax(out)
       
        return out

    def reset_parameters(self) -> None:
        """Reinitialize network weights using PyTorch defaults."""
        for fc in self.fcs:
            fc.reset_parameters()

        for fc_out in self.fc_outs:
            fc_out.reset_parameters()

    def __repr__(self) -> str:
        input_dim = self.fcs[0].in_features
        output_dims = [fc_out.out_features for fc_out in self.fc_outs]
        activation = type(self.acts[0]).__name__
        return f"{type(self).__name__}({input_dim=}, {output_dims=}, {activation=})"    
