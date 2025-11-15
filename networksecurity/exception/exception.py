import sys
from networksecurity.logging import logger

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        detailed_message = error_message_detail(error_message, error_detail)
        super().__init__(detailed_message)   # <-- Pass final message to parent
        self.error_message = detailed_message
    
    def __str__(self):
        return self.error_message
    
if __name__=='__main__':
    try:
        a = 1 / 0
    except Exception as e:
        print(CustomException(e, sys))
