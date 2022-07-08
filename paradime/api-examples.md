%% spellcheck-off

tsne = ParametricDR(
    model=MyModel(),
    global_relations=NeighborBasedPDist(
        transform=[
            PerplexityBased(30),
            Symmetrize(impl='tsne'),
            Normalize()
        ])
)

tsne.add_training_phase(
    name='pca_init',
    epochs=10,
    batch_size=500,
    loss=PositionalLoss(
        position_key='pca'
    )
)

tsne.add_training_phase(
    name='main',
    epochs=20,
    batch_size=500,
    batch_relations=DifferentiablePDist(
        transform=[
            TDistributionTransform(),
            Symmetrize('tsne'),
            Normalize()
        ]
    ),
    loss=RelationLoss(
        metric=KLDivergence()
    )
)

#-------------

tsne = ParametricDR(
    model=MyModel(),
    global_relations=NeighborBasedPDist(
        transform=[
            PerplexityBased(30),
            Symmetrize(impl='tsne'),
            Normalize()
        ])
)

tsne.add_training_phase(
    name='pca_init',
    epochs=10,
    batch_size=500,
    loss=PositionalLoss(
        position_key='pca'
    )
)

tsne.add_training_phase(
    name='main',
    epochs=20,
    batch_size=500,
    batch_relations=DifferentiablePDist(
        transform=[
            TDistributionTransform(),
            Symmetrize('tsne'),
            Normalize()
        ]
    ),
    loss=RelationLoss(
        metric=KLDivergence()
    )
)