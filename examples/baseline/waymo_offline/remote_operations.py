import paramiko
import math

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

def goal_region_reward(threshold, goal_x, goal_y, cur_x, cur_y):
    eucl_distance = math.sqrt((goal_x - cur_x)**2 + (goal_y - cur_y)**2)

    if eucl_distance <= threshold:
        return 10
    else:
        return 0


