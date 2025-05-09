class Response:
    def __init__(self):
        self.code = None
        self.data = None

    def success(data=None):
        respone = Response()
        respone.code = 200
        respone.data = data
        return respone
    
    def error(self, message: str):
        self.code = 400
        self.data = message
        return self
    
    def json(self):
        return {
            "code": self.code,
            "data": self.data
        }