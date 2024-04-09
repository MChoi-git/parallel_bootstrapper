# Philosophy
Distributed algorithm implementation is ripe with boilerplate. Let's amortize
the overhead of implementing deep learning algorithms.

## Design Principles
- Modularity and flexibility: Easily swap out parts for new shiny ones.
- Readability: People want to understand the code you write.
- PyTorch native: If you want something, make it!
- Easy to profile: You **will** optimize your code.

# Running Tests
- distributed: `torchrun --nnodes 1 --nproc-per-node 16 -m pytest pboot/distributed`
- parallel_modules: `torchrun --nnodes 1 --nproc-per-node 8 -m pytest pboot/modules/test_parallel_modules.py`
- everything else: `pytest --ignore=pboot/distributed/`
