"""
Created on Fri Mar  25 22:43

@author: jbk5816

"""

def format_time(time_seconds):
    if time_seconds < 120:
        return f"{time_seconds:.2f} sec"
    elif time_seconds < 7200: # 120 minutes
        minutes = time_seconds // 60
        seconds = time_seconds % 60
        return f"{int(minutes)} min {int(seconds)}"
    else:
        hours = time_seconds // 3600
        minutes = (time_seconds % 3600) / 60
        return f"{int(hours)} h {int(minutes)} min"