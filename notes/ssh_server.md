# How to connect to remote host by ssh

## Client
### - Prepare
1. Move to .ssh
   ```bash
   $ cd ~/.ssh
   ```
2. Make key
   ```bash
   $ ssh-keygen -t rsa
   ```
3. Share "~/.ssh/id_rsa.pub" to host \
   Copy and Paste or share by other ways. Do not share "id_rsa".
   ```bash
   $ ssh-copy-id -i ~/.ssh/id_rsa.pub [remote user]@[remote host IP]
   ```

### - Connect
- Command
  ```
  $ ssh -i id_rsa [remote user]@[remote host IP]
  ```
- Change "~/.ssh/config" file \
   Change <*> parts, and add to config file \
   ```
   Host <remote_host_name>
      HostName <IP-adress>
      User <remote_username>
      IdentityFile ~/.ssh/id_rsa
   ```
   Then, connect by remote_host_name or IP adress
   ```
   $ ssh [remote user]@[remote host IP]
   or
   $ ssh [remote user]@[remote host name]
   ```

## Host
### - Prepare
Make Server
```bash
$ sudo apt -y update
$ sudo apt -y install openssh-server
```

### - Make User
- Add User
  ```bash
  $ sudo adduser [USER_NAME]
  ```
- Add sudo (if needed)
  ```bash
  $ sudo gpasswd -a [USER_NAME] sudo
  ```


## References
- [Connect ssh by public key](https://qiita.com/kazokmr/items/754169cfa996b24fcbf5)
- [Add User](https://eng-entrance.com/linux-user-add)
- [Connect to Sever](https://www.digitalocean.com/community/tutorials/how-to-use-ssh-to-connect-to-a-remote-server-ja)
- [Make Server](https://www.kkaneko.jp/tools/server/pubkey.html)
- [Make User](https://www-creators.com/archives/241)
