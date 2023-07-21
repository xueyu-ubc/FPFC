import torch
from copy import deepcopy
from typing import Dict, Optional, Tuple
from collections import OrderedDict
from fedlab.utils.serialization import SerializationTool
from torch.utils.data import DataLoader, TensorDataset


class PerFedAvgClient:
    def __init__(
        self,
        client_id,
        alpha,
        beta,
        global_model,
        criterion,
        batch_size,
        dataset,
        local_epochs
    ) -> None:
        self.device = torch.device("cpu")
        self.local_epochs = local_epochs
        self.criterion = criterion
        self.id = client_id
        self.model = deepcopy(global_model)
        self.alpha = alpha
        self.beta = beta
        (X, y) = dataset['data'][client_id]
        traindataset = TensorDataset(X, y)
        self.trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle = True)

        (tX, ty) = dataset['test'][client_id]
        testdataset = TensorDataset(tX, ty)
        self.testloader = DataLoader(testdataset, batch_size=batch_size, shuffle = False)
        
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

    def train(
        self,
        global_model,
        epochs=None,
        hessian_free=False
    ) -> Tuple[OrderedDict, Optional[Dict]]:
        self.model = deepcopy(global_model)
        _epochs = self.local_epochs if epochs is None else epochs
        self._train(_epochs, hessian_free)

        return SerializationTool.serialize_model(self.model), None

    def _train(self, epochs, hessian_free=False) -> None:
        if epochs <= 0:
            return

        dataloader = self.trainloader
        iterator = self.iter_trainloader

        if hessian_free:  # Per-FedAvg(HF)
            optimizer = torch.optim.SGD(self.model.parameters(), lr = self.alpha)
            for _ in range(epochs):
                # frz_model_params = deepcopy(self.model)
                for x, y in dataloader: 
                    optimizer.zero_grad()
                    loss = torch.nn.CrossEntropyLoss()(self.model(x.float()), y)

                    loss.backward()
                    optimizer.step()

        else:  # Per-FedAvg(FO)
            for _ in range(epochs):
                frz_model_params = deepcopy(self.model)

                data_batch_1 = self.get_data_batch(dataloader, iterator, self.device)
                grads = self.compute_grad(self.model, data_batch_1, False)

                for param, grad in zip(self.model.parameters(), grads):
                    param.data.sub_(self.alpha * grad)

                data_batch_2 = self.get_data_batch(dataloader, iterator, self.device)
                grads = self.compute_grad(self.model, data_batch_2, False)

                self.model = deepcopy(frz_model_params)

                for param, grad in zip(self.model.parameters(), grads):
                    param.data.sub_(self.beta * grad)

    def compute_grad(
        self, model, data_batch, second_order_grads=False
    ) -> Tuple[torch.Tensor]:
        x, y = data_batch
        if second_order_grads:
            frz_model_params = deepcopy(model)
            delta = 1e-3
            dummy_model_params_1 = OrderedDict()
            dummy_model_params_2 = OrderedDict()
            with torch.no_grad():
                for layer_name, param in model.named_parameters():
                    dummy_model_params_1.update({layer_name: param + delta})
                    dummy_model_params_2.update({layer_name: param - delta})

            model.load_state_dict(dummy_model_params_1, strict=False)
            logit_1 = model(x)
            loss = self.criterion(logit_1, y)
            grads_1 = torch.autograd.grad(loss, model.parameters())

            model.load_state_dict(dummy_model_params_2, strict=False)
            logit_2 = model(x)
            loss = self.criterion(logit_2, y)
            grads_2 = torch.autograd.grad(loss, model.parameters())

            model = deepcopy(frz_model_params)

            grads = []
            for u, v in zip(grads_1, grads_2):
                grads.append((u - v) / (2 * delta))
            return grads

        else:
            logit = model(x)
            loss = self.criterion(logit, y)
            grads = torch.autograd.grad(loss, model.parameters())
            return grads
        
    def get_data_batch(self,
        dataloader, iterator, device=torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            x, y = next(iterator)

        return x.to(device), y.to(device)

