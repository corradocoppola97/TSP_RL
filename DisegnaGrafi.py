from Game_GH import gametable, RandomGraphSpecs, TableType
graphspecs_test = {
    RandomGraphSpecs.Nnodes : 100,
    RandomGraphSpecs.Nedges : 100*99,
    RandomGraphSpecs.Probability: None,
    RandomGraphSpecs.Seed: 1,
    RandomGraphSpecs.Repetitions: 1,
    RandomGraphSpecs.Distribution: None,
    RandomGraphSpecs.DistParams: (0,100,2)
}
edges_test, costs1_test, ds_test = gametable._random_graph_distances(graphspecs_test)