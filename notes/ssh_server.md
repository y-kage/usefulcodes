# How to connect to remote host by ssh

## Client

1. Move to .ssh
```
$ cd ~/.ssh
```
2. Make key
```
$ ssh-keygen -t rsa
```
3. Change "~/.ssh/config" file\
   Change <*> parts, and add to config file \
```
Host <remote_host>
   HostName <IP-adress>
   User <remote_username>
   IdentityFile ~/.ssh/id_rsa
```

5. Share "~/.ssh/id_rsa.pub" to host \
   Copy and Paste or share by other ways. Do not share "id_rsa".
```
$ cat ~/.ssh/id_rsa.pub
```

## Host (Unverified)
1. Login as root
```
$ sudo su -
```
2. Make user \
Some words may not be used
```
$ useradd -m <remote_username>
```
3. Set password \
Some words may not be used
```
$ passwd <remote_username>
```

## Connect
Change <*> to your setting
```
$ ssh <remote_username>@<remote_host>
```

## References
- [Add User](https://eng-entrance.com/linux-user-add)
- [Connect to Sever](https://www.digitalocean.com/community/tutorials/how-to-use-ssh-to-connect-to-a-remote-server-ja)
