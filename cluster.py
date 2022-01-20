import os
import paramiko #library für ssh
from datetime import datetime

HOSTS = [
    'palma2c.uni-muenster.de'
]

PROJECT_PATHS = [
    '$HOME/unet'
]

SLURM_SCRIPTS = [
    'cluster.run.palma2c.cmd'
]


def get_ssh_connection(host):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username='m_kais13', pkey=paramiko.RSAKey.from_private_key_file("/home/momok/.ssh/id_rsa"))
    return ssh


def submit_python_script(ssh, host, slurm_script, project_path, command, tasks_per_node=16, mem=32, partition='normal', time='8:00:00', git_push=False):
    cmd_template = open(slurm_script, 'r').read()
    if git_push:
        os.system('git add -A && git commit -m "fast push"')
        os.system('git push')
        stdin, stdout, stderr = ssh.exec_command('cd {} && git pull && cd $HOME'.format(project_path))
        print(stdout.readlines())
    stdin, stdout, stderr = ssh.exec_command('squeue | grep m_kais13')
    i = len(stdout.readlines())
    now = datetime.now()
    log_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    cmd = cmd_template.format(i=i, time=time, project_path=project_path, python_script=command, tasks_per_node=tasks_per_node, mem=mem, partition=partition, log_name=log_name)
    stdin, stdout, stderr = ssh.exec_command('echo "{}" | tee $HOME/jobs/run.cmd'.format(cmd))
    stdout.readlines()
    stdin, stdout, stderr = ssh.exec_command('sbatch $HOME/jobs/run.cmd')
    print(stdout.readlines())
    stdin, stdout, stderr = ssh.exec_command('squeue | grep m_kais13')
    print("submitted {} jobs".format(len(stdout.readlines())))


commands = [
   "main.py"
]

ssh = get_ssh_connection(HOSTS[0])
for i, c in enumerate(commands):
    submit_python_script(ssh, HOSTS[0], SLURM_SCRIPTS[0], PROJECT_PATHS[0], c, tasks_per_node=8, mem=32, partition='gpu2080', git_push=True if i == 0 else False, time='24:00:00')
ssh.close()

    # os.system("python train.py {}".format(c))
    # print("Training {} finished.".format(c))
#
# for i, host in enumerate(HOSTS):
#     submit_python_script(host, SLURM_SCRIPTS[i], PROJECT_PATHS[i], ['run.py'], tasks_per_node=6, mem=30)
