========================================================================
   HOW TO RUN THIS ON YOUR LAPTOP (NVIDIA RTX GPU)
========================================================================

Hey! Thank you for helping run this machine learning experiment. 

All you need to do:

1. Extract this entire zip folder anywhere on your computer (e.g. Desktop or C:\).
2. Double-click "run_training_rtx.bat" to start.
   - It will automatically check Python, install CUDA PyTorch & packages.
   - It will train the model using your RTX GPU (~2-3 hours on RTX 5070).
   - Checkpoints are saved every 200 steps. If your laptop sleeps or restarts, just double-click "run_training_rtx.bat" again and it will resume right where it left off!
3. When finished, you will see a new folder named:
   "arm_a_results_package"
4. Just ZIP that "arm_a_results_package" folder and send it back to Satvik!

------------------------------------------------------------------------
Requirements:
- Windows 10/11
- NVIDIA RTX GPU with drivers installed
- Python 3.10, 3.11, or 3.12 installed (with "Add to PATH" checked)
- Internet connection (first 2 minutes only, to download the base Qwen model)
========================================================================
