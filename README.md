
# BTSPFormer: Transformer Architecture Inspired by Behavioral Timescale Synaptic Plasticity for Damped Sine Wave Prediction

This document provides a detailed overview of the **BTSPFormer** model, a Transformer-based architecture designed for time series prediction tasks, particularly modeling damped sine waves. The design draws inspiration from the neurobiological phenomenon known as **Behavioral Timescale Synaptic Plasticity (BTSP)** [1].

---

## 1. Introduction to BTSP

Behavioral Timescale Synaptic Plasticity (BTSP) is a form of synaptic learning discovered in hippocampal neurons that occurs over several seconds and supports the rapid encoding of predictive spatial memory. The core mechanisms include:

- **Eligibility Trace (ET)**: Synaptic tagging signal generated during recent activity.
- **Instructive Signal (IS)**: Delivered via plateau potentials, guiding synaptic weight change.
- **Asymmetric Time Dynamics**: Weight change is stronger for inputs that occur slightly before plateau events.
- **Bidirectional Plasticity**: Weaker synapses are potentiated, while stronger ones are suppressed.

🧠 **Reference**:  
[1] Milstein, A. D., Li, Y., Bittner, K. C., Grienberger, C., Soltesz, I., Magee, J. C., & Romani, S. (2021). *Bidirectional synaptic plasticity rapidly modifies hippocampal representations.* eLife, 10, e73046. https://doi.org/10.7554/eLife.73046

---

## 2. BTSPFormer Model Architecture

The BTSPFormer integrates the BTSP-inspired mechanism into the Transformer architecture. It includes the following components:

### 🔹 2.1 Input Embedding
- Projects input time series data (e.g., 1D values) to a higher-dimensional latent space.
- Default: 1 → 64 dimensions.

### 🔹 2.2 Predictive Positional Encoding
- A non-symmetric position embedding simulating BTSP’s time-asymmetry.
- Peaks slightly earlier than the expected event, encoding *predictive memory*.

### 🔹 2.3 BTSP Attention Layer
- Replaces standard attention with a **BTSP-inspired plastic attention** mechanism:
  - Combines Eligibility Trace (ET) and Instructive Signal (IS) to modify attention weights during training.
  - Enables context-aware dynamic routing of information.

### 🔹 2.4 Transformer Encoder
- Stack of `n` BTSPFormerBlocks.
- Each block consists of:
  - LayerNorm → BTSP Attention → Residual → FFN → Residual.
- FFN expands to `model_dim * 4`, then compresses back.

### 🔹 2.5 Output MLP
- Final two-layer MLP:
  - `model_dim → model_dim/2 → 1`
  - GELU activation + dropout for regularization.

---

## 3. Data Generation: Damped Sine Wave

The model is trained on synthetic data generated from the damped sine equation:

\\[
x(t) = A \\cdot e^{-\\gamma t} \\cdot \\cos(\\omega t + \\phi)
\\]

Parameters:
- \\( A = 1.0 \\) (Amplitude)
- \\( \\gamma = 0.1 \\) (Damping factor)
- \\( \\omega = 2.0 \\) (Angular frequency)
- \\( \\phi = 0 \\) (Phase shift)

Input-target pairs are constructed as:
- Input: A sliding window of `seq_len` time steps.
- Target: The next step in the sequence.

---

## 4. Training Pipeline

### 🔸 4.1 Configuration
- **Split**: 80% training, 20% validation
- **Loss**: Mean Squared Error (MSE)
- **Metrics**: MSE, MAE
- **Optimizer**: Adam or AdamW
- **Scheduler**: ReduceLROnPlateau

### 🔸 4.2 Model Management
- **Checkpointing**: Saves best-performing model
- **Early Stopping**: Stops when validation loss plateaus
- **Logging**: CSV + TensorBoard for training metrics

---

## 5. Visualization Components

BTSPFormer provides visualization for both learning dynamics and model behavior:

1. **Loss Curves**: Training loss across epochs.
2. **Prediction vs Ground Truth**:
   - Standard plot (time-aligned)
   - Sorted plot for smoother visualization

---

## 6. CLI Usage

The model can be executed with the following command:

```bash
python main.py --seq_len 50 --num_samples 10000 --batch_size 128 --epochs 100 --lr 1e-3 --use_gpu
```


The model supports the following optional flags for configuration:

| Flag | Description |
|------|-------------|
| `--seq_len` | Sequence length for the input time series |
| `--num_samples` | Number of synthetic training samples to generate |
| `--batch_size` | Batch size used during training |
| `--epochs` | Number of training epochs |
| `--lr` | Learning rate for optimizer |
| `--use_gpu` | Enable GPU acceleration if available |
| `--model_dim` | Dimension of the internal Transformer representation (default: 64) |
| `--heads` | Number of attention heads (default: 4) |
| `--layers` | Number of transformer encoder layers (default: 2) |

---

## 7. Output Files

Generated outputs include:

| File | Description |
|------|-------------|
| `btspformer_model.pth` | Final trained model |
| `btspformer_loss_curve.png` | Training loss curve |
| `btspformer_inference.png` | Prediction vs target visualization |
| `logs/` | TensorBoard logs |
| `checkpoints/` | Model checkpoints (optional) |

---

## 8. Future Work

BTSPFormer serves as a biologically inspired baseline. Future extensions may include:
- EEG-to-Image generation via diffusion
- Reinforcement Learning with predictive planning
- Multimodal memory-guided attention
