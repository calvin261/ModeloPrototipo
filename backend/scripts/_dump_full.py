"""Full UI dump - find all interactive elements."""
import re
import subprocess

adb = r"C:\Users\jeffe\AppData\Local\Android\Sdk\platform-tools\adb.exe"
xml = subprocess.run([adb, "-s", "R52XB09FEKE", "shell", "cat", "/sdcard/window_dump.xml"],
                     capture_output=True, text=True).stdout
print(f"xml length: {len(xml)}")
print(f"first 500 chars: {xml[:500]}")
print()
print("=== content-desc ===")
for m in re.finditer(r'content-desc="([^"]+)"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', xml):
    val = m.group(1)
    if val.strip():
        cx, cy = (int(m.group(2)) + int(m.group(4))) // 2, (int(m.group(3)) + int(m.group(5))) // 2
        print(f"  ({cx:4d},{cy:4d})  {val!r}")
print()
print("=== text ===")
for m in re.finditer(r'text="([^"]+)"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', xml):
    val = m.group(1)
    if val.strip():
        cx, cy = (int(m.group(2)) + int(m.group(4))) // 2, (int(m.group(3)) + int(m.group(5))) // 2
        print(f"  ({cx:4d},{cy:4d})  {val!r}")
