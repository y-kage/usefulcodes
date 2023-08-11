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

# 2. useful sites
Some useful sites with some abstract

## How to run process while disconnected from server
url : https://blog.mktia.com/nohup-and-bg-process-on-remote-server/

- To run process
```
nohup python sample.py &
```

- To apply to running process
1. Suspend process by "Ctrl + z"
2. Move process to background by
   ```bg```
4. Detatch process from shell by
   ```disown -h```
