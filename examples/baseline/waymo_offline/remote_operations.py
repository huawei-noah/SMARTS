import paramiko
class remote_operations:
    def __init__(self):
        self.ssh_client = paramiko.SSHClient()

    def connect(self, hostname, username, password):
        self.ssh_client.load_system_host_keys()
        self.ssh_client.connect(hostname, username=username, password=password)
        sftp_client = self.ssh_client.open_sftp()
        return sftp_client
        # file = sftp_client.open(filename)
        # files_list = sftp_client.listdir(path)

