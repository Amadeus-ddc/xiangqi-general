"""Stream a match child; release the startup lease only after a new raw turn."""
import fcntl,json,os,re,subprocess,threading,time
from pathlib import Path

def raw_turns(folder):
 total=0
 for path in Path(folder).glob('game-*.json'):
  value=json.loads(path.read_text())
  total+=sum('raw' in turn for turn in value['game']['turns'])
 return total

def stream_match_process(command,log,device,progress,lease,on_release,*,poll_seconds=2):
 before=raw_turns(progress);released=False;process=None
 def release(reason,after,returncode=None):
  nonlocal released
  if released:return
  fcntl.flock(lease,fcntl.LOCK_UN);released=True
  on_release({'reason':reason,'previous_raw_turns':before,'observed_raw_turns':after,
   'child_pid':process.pid if process else None,'child_returncode':returncode})
 try:
  with Path(log).open('a') as handle:
   process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=device))
   errors=[]
   def forward():
    try:
     for line in process.stdout:
      line=re.sub(r'https?://[^\s)\]]+','[REDACTED_URL]',line)
      handle.write(re.sub(r'(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?','[REDACTED_ADDRESS]',line));handle.flush()
    except BaseException as error:errors.append(error)
   reader=threading.Thread(target=forward,daemon=True);reader.start()
   while process.poll() is None:
    if not released:
     after=raw_turns(progress)
     if after>before:release('new_raw_model_turn_observed',after)
    time.sleep(poll_seconds)
   reader.join();code=process.wait()
   release('child_exited_without_observed_new_raw_turn',raw_turns(progress),code)
   if errors:raise errors[0]
   if code:raise subprocess.CalledProcessError(code,command)
 finally:
  if process is None or process.poll() is not None:
   release('child_start_failed_or_already_exited',raw_turns(progress),process.returncode if process else None)
