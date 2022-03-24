# Convolutional Neural Network
This code provides and optimised version of a convolutional neural network. The target machine has four processors.  Each processor has eight out-of-order pipelined, superscalar cores.  And each core has two-way simultaneous multithreading. The cores support Intel SSE4 vector instructions.  
The optimisation was done by myself and @8n76nn98

## Optimisations
- Vectorised using SSE intrinsics
- Parallised using OpenMP
