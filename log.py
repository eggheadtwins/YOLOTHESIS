from collections import deque

web_logs = deque()


def info(message):
    web_logs.append(message)


def error(message):
    web_logs.append(f"<span style='color:red'>{message}</span>")


def success(message):
    web_logs.append(f"<span style='color:green'>{message}</span>")


def warn(message):
    web_logs.append(f"<span style='color:orange'>{message}</span>")
