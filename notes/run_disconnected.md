# How to run process while disconnected from server
## To run process
```
$ nohup python sample.py &
```

## To apply to running process
1. Suspend process by "Ctrl + z"
2. Move process to background by
   ```$ bg```
4. Detatch process from shell by
   ```$ disown -h```

## How to stop the process
- Before leaving from the shell running the code
  1. Find job number
     ```
     $ jobs
     ``` 
  2. Kill job
     ```
     $ kill %<job_number>
     ```
- After leaving from the shell running the code
   1. Find PID
      ```
      $ ps aux
      ```
   2. Kill process
      ```
      $ kill <PID>
      ```

# References
[Run Code Remote](https://blog.mktia.com/nohup-and-bg-process-on-remote-server/)
