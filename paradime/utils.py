from datetime import datetime

def report(message: str) -> None:
    
    now_str = datetime.now().isoformat(
        sep=' ',
        timespec='milliseconds'
    )[:-2]

    print(now_str + ': ' + message)