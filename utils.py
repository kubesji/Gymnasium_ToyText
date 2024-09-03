SUCCESS = '\033[1m\033[92msuccess\033[0m'
FAILURE = '\033[1m\033[91mfailure\033[0m'

def success(string):
    return '\033[1m\033[92m' + string + '\033[0m'

def fail(string):
    return '\033[1m\033[91m' + string + '\033[0m'