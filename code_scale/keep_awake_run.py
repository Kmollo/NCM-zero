"""Run a bash pipeline while asking Windows not to sleep (no settings are changed).

The request lasts only while this process runs. Closing the laptop lid can still sleep it,
depending on the lid setting. Started by the scheduled task "NCM Path A":
  pythonw.exe code_scale/keep_awake_run.py code_scale/pipeline_path_a.sh
"""
import ctypes
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASH = r"C:\Program Files\Git\usr\bin\bash.exe"
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def main():
    os.chdir(ROOT)
    script = sys.argv[1] if len(sys.argv) > 1 else "code_scale/pipeline_path_a.sh"
    log = open(os.path.join(ROOT, "runs", "code-scale-v1", "keep-awake.log"), "a")
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} keep-awake on, running {script}\n")
    log.flush()
    try:
        code = subprocess.run([BASH, "-l", script], cwd=ROOT, stdout=log, stderr=log,
                              creationflags=subprocess.CREATE_NO_WINDOW).returncode
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} finished with exit code {code}, keep-awake off\n")
    log.close()


if __name__ == "__main__":
    main()
