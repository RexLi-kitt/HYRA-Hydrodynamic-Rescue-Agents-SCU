from .guard import Guard

class Gaps(Guard):
    def __init__(self):
        super().__init__()
        self.options.update(width=48,depth=16,commit=5,rarity=1.4)

