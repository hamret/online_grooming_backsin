import torch

# 1. PyTorch가 CUDA(GPU)를 사용할 수 있는지 확인
is_available = torch.cuda.is_available()
print(f"CUDA (GPU) 사용 가능 여부: {is_available}")

if is_available:
    # 2. 현재 설치된 PyTorch가 인식하는 CUDA 버전
    print(f"PyTorch CUDA 버전: {torch.version.cuda}")

    # 3. 사용 중인 GPU 개수
    print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")

    # 4. 현재 GPU 이름
    print(f"현재 GPU 이름: {torch.cuda.get_device_name(0)}")

    # 5. 현재 GPU의 메모리 (바이트 단위)
    total_mem = torch.cuda.get_device_properties(0).total_memory
    print(f"현재 GPU 총 메모리: {total_mem / (1024**3):.2f} GB")

else:
    print("--- GPU를 찾을 수 없습니다. ---")
    print("PyTorch가 CPU 모드로 설치되었거나, NVIDIA 드라이버/CUDA가 잘못 설정되었습니다.")
    print("NVIDIA 드라이버를 업데이트하거나, PyTorch를 CUDA 버전에 맞게 재설치해야 합니다.")