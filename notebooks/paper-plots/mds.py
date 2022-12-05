import sys

sys.path.append("../../")

import copy

import numpy as np
import torch
import sklearn.datasets
import sklearn.decomposition
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

import paradime.dr
import paradime.relations
import paradime.transforms
import paradime.loss
import paradime.utils

paradime.utils.seed.seed_all(42)

diabetes = sklearn.datasets.load_diabetes()
data = diabetes["data"]


def mse(a, b):
    return torch.sum((a - b) ** 2)


df = pd.DataFrame(columns=["Model", "Batch size", "Loss"])

hd_pdist = torch.nn.functional.pdist(torch.tensor(data))

batch_sizes = [10, 50, 100, 221, 442]

for name, model in zip(
    ["Linear (10, 2)", "Non-linear (10, 2)", "Non-linear (10, 5, 2)"],
    [
        torch.nn.Linear(10, 2, bias=False),
        torch.nn.Sequential(
            torch.nn.Linear(10, 2, bias=True),
            torch.nn.Softplus(),
        ),
        torch.nn.Sequential(
            torch.nn.Linear(10, 5, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(5, 2, bias=True),
            torch.nn.Softplus(),
        ),
    ],
):
    for bs in batch_sizes:
        for _ in range(10):

            pd_mds = paradime.dr.ParametricDR(
                model=copy.deepcopy(model),
                global_relations=paradime.relations.PDist(
                    transform=[
                        paradime.transforms.ToSquareTensor(),
                        # paradime.transforms.Functional(lambda x: x.float()),
                    ]
                ),
                batch_relations=paradime.relations.DifferentiablePDist(
                    transform=paradime.transforms.ToSquareTensor()
                ),
                dataset=data,
                verbose=True,
            )
            pd_mds.add_training_phase(
                epochs=500,
                batch_size=bs,
                learning_rate=0.02,
                loss=paradime.loss.RelationLoss(
                    loss_function=mse,
                    embedding_method="forward",
                    normalize_sub=False,
                ),
                report_interval=75,
            )
            pd_mds.train()

            loss = mse(
                hd_pdist,
                torch.nn.functional.pdist(pd_mds.apply(data)),
            )
            df = pd.concat(
                [df, pd.DataFrame([[name, bs, loss]], columns=df.columns)],
                ignore_index=True,
            )


class DirectModel(torch.nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        self.coords = torch.nn.Linear(size, dim, bias=False)

    def forward(self, X):
        pass

    def embed(self, indices):
        return self.coords.weight.data[indices]


class DirectLoss(paradime.loss.Loss):
    def forward(
        self,
        model,
        global_relations,
        batch_relations,
        batch,
        device,
    ) -> torch.Tensor:
        global_rel = global_relations["rel"].sub(batch["indices"])
        batch_rel = (
            batch_relations["rel"]
            .compute_relations(model.coords.weight[batch["indices"]])
            .data
        )

        return mse(global_rel, batch_rel)


for _ in range(10):
    for bs in batch_sizes:

        pd_mds = paradime.dr.ParametricDR(
            model=DirectModel(2, len(data)),
            global_relations=paradime.relations.PDist(
                transform=[
                    paradime.transforms.ToSquareTensor(),
                    # paradime.transforms.Functional(lambda x: x.float()),
                ]
            ),
            batch_relations=paradime.relations.DifferentiablePDist(
                transform=paradime.transforms.ToSquareTensor()
            ),
            dataset=data,
            verbose=True,
        )
        pd_mds.add_training_phase(
            epochs=500,
            batch_size=len(data) // 5,
            learning_rate=0.01,
            loss=DirectLoss(),
            report_interval=75,
        )
        pd_mds.train()

        loss = mse(
            hd_pdist,
            torch.nn.functional.pdist(pd_mds.model.coords.weight.data),
        )

        df = pd.concat(
            [df, pd.DataFrame([["Direct", bs, loss]], columns=df.columns)],
            ignore_index=True,
        )


from sklearn.manifold import MDS


for _ in range(10):
    mds = torch.tensor(MDS(n_init=1).fit_transform(data))

    loss = mse(hd_pdist, torch.nn.functional.pdist(mds))

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [["SMACOF", batch_sizes[-1], loss]], columns=df.columns
            ),
        ],
        ignore_index=True,
    )

df.to_csv("mds_runs.csv")
