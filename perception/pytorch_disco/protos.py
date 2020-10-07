class VoxProto:
    def __init__(self, shape):
    	self.dim = len(shape)
    	self.shape = shape

    def __repr__(self):
        return str(self.shape)