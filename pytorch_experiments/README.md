
## 📁 Project Structure

- `arg_extractor.py`: Parses command-line arguments.
- `experiment_builder.py`: Handles training, validation, testing, gradient flow plotting, and model saving.
- `model_architectures.py`: Defines VGG-like convolutional blocks with optional BN and RC.
- `storage_utils.py`: Utilities for saving statistics and model checkpoints.
- `train_evaluate_image_classification_system.py`: Main entry point for training.
- `unit_tests.py`: Tests for custom model blocks.
- `.sh scripts`: Easy-to-run setups for specific model configurations:
  - `run_vgg_38_default.sh`: Vanilla VGG38
  - `vgg_38_bn.sh`: VGG38 with BatchNorm
  - `vgg_38_bn_rc.sh`: VGG38 with BatchNorm + Residual Connections
 
## How to Run

Run any of the predefined training scripts:

```bash
# Run baseline VGG38
bash run_vgg_38_default.sh

# Run VGG38 with BatchNorm
bash vgg_38_bn.sh

# Run VGG38 with BatchNorm and Residual Connections
bash vgg_38_bn_rc.sh
```

## Run Manual configuration 

```bash
python pytorch_experiments/train_evaluate_image_classification_system.py --experiment_name YOUR_EXP_NAME --args**
```
