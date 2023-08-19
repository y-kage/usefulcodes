# How to run process while disconnected from server
## To run process
```
nohup python sample.py &
```

## To apply to running process
1. Suspend process by "Ctrl + z"
2. Move process to background by
   ```bg```
4. Detatch process from shell by
   ```disown -h```

# References
https://blog.mktia.com/nohup-and-bg-process-on-remote-server/
