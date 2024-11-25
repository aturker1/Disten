from abc import ABC


class Placement(ABC):
    pass


class Shard(Placement):
    pass


class Replicate(Placement):
    pass
