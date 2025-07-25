
try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))

        # Create a small tensor and move to GPU
        x = torch.rand(5, 3)
        print("CPU tensor:", x)

        # Try GPU operations
        if torch.cuda.is_available():
            x = x.cuda()
            print("GPU tensor:", x)
    else:
        print("CUDA is not available. Checking for issues...")

        # Check if CUDA libraries can be found
        import os
        cuda_path = os.environ.get('CUDA_PATH')
        print("CUDA_PATH environment variable:", cuda_path)

        # Additional diagnostics
        try:
            from subprocess import check_output
            nvidia_smi = check_output('nvidia-smi').decode('utf-8')
            print("NVIDIA-SMI Output:\n" + nvidia_smi)
        except:
            print("nvidia-smi command failed - drivers may not be installed correctly")

except Exception as e:
    print(f"Error in GPU diagnostics: {e}")
    import traceback
    traceback.print_exc()

print("\nDiagnostics complete")
