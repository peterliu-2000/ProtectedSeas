import time, calendar

def parse_time(d:str, t:str) -> int:
    """
    Convert a string representation of date and time to the unix time encoding
    Remark: This function assumes that the date/time is recorded in UTC.
    However, the functionality of this function shall not be affected if
    the timezone offset is consistent
    
    Args:
        d: date string, format "yyyy-mm-dd"
        t: time string, format "hh:mm:ss"
    """
    time_struct = time.strptime(d+t, "%Y-%m-%d%H:%M:%S")
    return calendar.timegm(time_struct)

def str_time(t:int) -> str:
    """
    The inverse of parse_time. Written for debugging purposes
    """
    s = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t))
    return s
    