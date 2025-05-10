wget https://huggingface.co/SWivid/E2-TTS/resolve/main/E2TTS_Base/model_1200000.pt
python python/eval_c3.py --model_path model_1200000.pt --save_path C3TTS_Base_evaluation