# End-to-End kubeflow tutorial using a Sequence-to-Sequence model

This example demonstrates how you can use `kubeflow` end-to-end to train and
serve a distributed Pytorch model on an existing kubernetes cluster. This
tutorial is based upon the below projects:
- [DDP training CPU and GPU in Pytorch-operator example](https://github.com/kubeflow/pytorch-operator/tree/master/examples/ddp/mnist)
- [Google Codelabs - "Introduction to Kubeflow on Google Kubernetes Engine"](https://github.com/googlecodelabs/kubeflow-introduction)
- [IBM FfDL - PyTorch MNIST Classifier](https://github.com/IBM/FfDL/tree/master/community/FfDL-Seldon/pytorch-model)
## Goals

There are two primary goals for this tutorial:

*   Demonstrate an End-to-End kubeflow example
*   Present an End-to-End Pytorch model

By the end of this tutorial, you should learn how to:

*   Setup a Kubeflow cluster on an existing Kubernetes deployment
*   Spawn up a shared-persistent storage across the cluster to store models
*   Train a distributed model using Pytorch and GPUs on the cluster
*   Serve the model using [Seldon Core](https://github.com/SeldonIO/seldon-core/)
*   Query the model from a simple front-end application

## Steps:

1.  [Setup a Kubeflow cluster](01_setup_a_kubeflow_cluster.md)
1.  Training the model using PyTorchJob:
    -  [Distributed Training using DDP and PyTorchJob](02_distributed_training.md)
1.  [Serving the model](03_serving_the_model.md)
1.  [Querying the model](04_querying_the_model.md)
1.  [Teardown](05_teardown.md)

#TODO
- [ ] 01_setup_a_kubeflow_cluster
- [ ] 02_distributed_training
- [ ] 03_serving_the_model
- [ ] 04_querying_the_model
- [ ] 05_teardown