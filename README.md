# Car-Detection-and-Tracking-Using-YOLOv8-and-OC-SORT

## PETUNJUK INSTALASI
Langkah-langkah untuk instalasi :
1. Install python
2. Buat virtual environment dengan python -m venv MyEnv
3. Install dependensi dengan pip install opencv-python numpy torch
torchvision torchaudio ultralytics tqdm
4. pip install ocsort
5. pip install torch torchvision torchaudio --index-url
https://download.pytorch.org/whl/cu121
6. python -m pip install --upgrade pip jika ocsort gagal diinstall lalu pip install
ocsort lagi
7. Pastikan file model yang ingin digunakan ada dalam hal ini
bestAugmentaion.pt
## PETUNJUK PENGGUNAAN
1. Import file model berbentuk .pt dan siapkan juga video yang ingin di uji
kedalam satu folder yang sama
2. Jalankan perintah MyEnv\Scripts\activate
3. Jalankan perintah python main.py --video "(Nama Video Sumber yang ingin
di uji).mp4" --weights "bestAugmentation.pt" --show-live --output "(Nama
Folder/(Nama Video untuk di save).mp4" --save-video --fixed-start-time
"HH:MM:SS" (Hours, Minutes, Seconds)
