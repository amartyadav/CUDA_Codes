#### Disclaimer

This readme has been written completely using AI. 
Everything else in this repo has been generated without any AI at all. AI is an assistant. Use it so.

## CUDA Learning Sandbox

This repository is a collection of random CUDA kernels, experiments, and exercises I'm working on while learning GPU programming. It includes solutions from LeetGPU, basic kernel implementations, and general practice codes.
Note: This is a scratchpad for learning. Major, project-worthy CUDA implementations are and will be hosted in their own dedicated repositories.

## What's Inside?

* LeetGPU Solutions: Problem-solving with a focus on parallel efficiency.
* Basic Kernels: Vector addition, matrix multiplication, and basic atomics.
* Random Experiments: Testing memory patterns, tiling, and shared memory.

## How to Run

Most files can be compiled using nvcc:

nvcc filename.cu -o output
./output

## Goals

* Understand CUDA memory hierarchy (Global, Shared, Constant).
* Master thread synchronization and warp shuffles.
* Optimize kernels for maximum throughput.

------------------------------
