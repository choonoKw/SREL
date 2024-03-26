"""
Created on Fri Mar  25 22:43

@author: jbk5816

"""

def format_time(seconds):
    if seconds < 120:
        return f"{seconds:.2f} sec"
    elif seconds < 7200: # 120 minutes
        minutes = seconds / 60
        return f"{minutes:.2f} min"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) / 60
        return f"{int(hours)}h {int(minutes)}min"