# 🚀 Competing GPUs – Multi-GPU Particle Racing Simulation

A CUDA-based simulation where multiple GPUs compete in parallel to move particle teams across a 2D grid. Developed as part of the **CUDA at Scale for the Enterprise** course offered by NVIDIA through Coursera.

---

## 📌 Project Overview

This project implements a **competitive particle racing simulation** using multiple CUDA-enabled GPUs. Each GPU independently controls a team of particles racing across a 2D grid while performing physics computations such as position updates, velocity calculations, and collision detection in parallel.

The first GPU to successfully move all its particles into the target zone is declared the winner.

### Key Features

* Multi-GPU workload distribution using `cudaSetDevice()`
* Independent particle teams assigned to different GPUs
* Parallel physics computation using CUDA kernels
* Frame-level synchronization with CUDA Events
* Performance comparison across multiple GPUs
* Real-time race completion tracking

---

## 🏗️ System Architecture

```text
main.cu
├── Detect available GPUs using cudaGetDeviceCount()
├── Assign particle teams using cudaSetDevice()
├── Launch physics kernels on each GPU
│   ├── Position Update
│   ├── Velocity Update
│   └── Collision Detection
├── Synchronize execution using CUDA Events
└── Determine winning GPU based on race completion
```

---

## 🚀 Requirements

### Hardware

* NVIDIA GPU(s)
* Compute Capability 6.0 or higher

### Software

* CUDA Toolkit 11.0+
* GCC / G++ Compiler
* NVIDIA CUDA Drivers
* Linux or Windows

---

## 🔧 Build and Run

### Clone the Repository

```bash
git clone https://github.com/shanmugavasanth/GPU-Racing-Simulator-Multi-GPU-Particle-Competition.git
cd GPU-Racing-Simulator-Multi-GPU-Particle-Competition
```

### Compile

```bash
nvcc -o competing_gpus main.cu kernel.cu -lcuda
```

### Run

```bash
./competing_gpus
```

---

## ⚙️ CUDA Concepts Demonstrated

| CUDA Concept             | Purpose                                           |
| ------------------------ | ------------------------------------------------- |
| `cudaGetDeviceCount()`   | Detect available GPUs                             |
| `cudaSetDevice(id)`      | Assign work to a specific GPU                     |
| CUDA Kernels             | Execute physics calculations in parallel          |
| `cudaEventRecord()`      | Mark synchronization/timing points                |
| `cudaEventSynchronize()` | Wait for GPU tasks to complete                    |
| Multi-GPU Execution      | Run particle teams independently on separate GPUs |

---

## 🎮 Simulation Workflow

1. Detect all available CUDA GPUs.
2. Create a particle team for each GPU.
3. Assign teams using `cudaSetDevice()`.
4. Launch physics update kernels in parallel.
5. Synchronize frame completion using CUDA Events.
6. Check whether all particles reached the target zone.
7. Record completion times for each GPU.
8. Declare the fastest GPU as the winner.

---

## 📊 Example Output

```text
Detected GPUs: 2

GPU 0 Team Racing...
GPU 1 Team Racing...

Frame 125:
GPU 0 Completed Race in 2.14 ms
GPU 1 Completed Race in 1.87 ms

🏆 Winner: GPU 1
```

---

## 📈 Learning Outcomes

This project demonstrates:

* Multi-GPU programming with CUDA
* Device management and workload partitioning
* Parallel physics simulation
* GPU synchronization techniques
* CUDA event-based timing and benchmarking
* Performance analysis across multiple GPUs

---

## 📁 Project Structure

```text
GPU-Racing-Simulator-Multi-GPU-Particle-Competition/
├── main.cu        # Main simulation and GPU management
├── kernel.cu      # Physics update kernels
├── utils.h        # Utility/helper functions
├── README.md      # Project documentation
```

---

## 📚 Course Information

Developed as an Honors Peer-Graded Assignment for:

**CUDA at Scale for the Enterprise**
**Module 2: Multiple CPU/GPU Systems**
NVIDIA Deep Learning Institute (DLI) on Coursera

---

## 📝 License

This project is licensed under the MIT License.

Feel free to use, modify, and distribute this project for educational and research purposes.
