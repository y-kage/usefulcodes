# 1. useful codes
Some useful codes in particular situation.

## extract_requirements_diff.py
You can know the difference of two requirements.txt files by the command below. The result will be shown in "requirements_diff_out.txt". If you pass outfile in the command line, the output will be in the name you passed.
```
python extract_requirements_diff.py <file1> <file2> (<outfile>)
```
To get your pip lists in the requirements.txt format, use the command below. 
```
pip freeze > mypiplist.txt
```

## torch_check.py
You can check the version of torch, check if you can use gpu, and check how many gpus you can use. It is useful when you are setting up your environment.

# 2. useful notes
- Build_new_env.pdf \
  Building new environment. (cuda, torch, etc.)
- run_disconnected.md \
  How to run codes at server while disconnected from server.
- ssh_server.md \
  How to connect to remote host by ssh.
