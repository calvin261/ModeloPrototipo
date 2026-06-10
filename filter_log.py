import sys
import re

path = sys.argv[1]
try:
    with open(path, 'rb') as f:
        data = f.read()
    # decode latin-1 (no fail)
    text = data.decode('latin-1', errors='replace')
except Exception as e:
    print('error reading:', e)
    sys.exit(1)

# Print all relevant lines
keywords = sys.argv[2].split(',') if len(sys.argv) > 2 else ['Caused', 'Error', 'Exception', 'hilt', 'dagger', 'ViewModel', 'Hilt_', 'CmaKe', 'OrtEnv', 'ort', 'Onnx', 'onnx', 'Static', 'Dynamic', 'SignClassifier', 'Sign', 'init']
for line in text.splitlines():
    if any(k.lower() in line.lower() for k in keywords):
        # strip weird chars
        clean = ''.join(c if (32 <= ord(c) < 127 or ord(c) in (9, 10, 13)) else '?' for c in line)
        print(clean)
