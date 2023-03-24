import numpy as np

GameDescription = {}
GameDescription["aliens"] = {
    "ascii": [".", "0", "1", "2", "A"],
    "mapping": [13, 3, 11, 12, 1],
    "state_shape": (14, 12, 32),
    "model_shape": [(3, 4), (6, 8), (12, 16), (12, 32)],
    "requirements": ["A"],
}
GameDescription["zelda"] = {
    "ascii": [".", "w", "g", "+", "1", "2", "3", "A"],
    "ascii_to_tile": {
        "goal": ["floor", "goal"],
        "": ["floor"],
        "floor": ["floor"],
        "key": ["floor", "key"],
        "nokey": ["floor", "nokey"],
        "withkey": ["floor", "withkey"],
        "sword": ["floor", "sword"],
        "monsterQuick": ["floor", "monsterQuick"],
        "monsterNormal": ["floor", "monsterNormal"],
        "monsterSlow": ["floor", "monsterSlow"],
        "wall": ["wall"],
    },
    "mapping": [13, 0, 3, 4, 10, 11, 12, 7],
    "state_shape": (13, 12, 16),
    "model_shape": [(3, 4), (6, 8), (12, 16)],
    "requirements": ["A", "g", "+"],
}


class Env:
    def __init__(self, name, length=500, wrapper_args=None):
        self.name = name
        self.version = 'v1'
        self.length = length
        self.kwargs = wrapper_args
        try:
            self.ascii = GameDescription[name]["ascii"]
            self.mapping = GameDescription[name]["mapping"]
            self.state_shape = GameDescription[name]["state_shape"]
            self.model_shape = GameDescription[name]["model_shape"]
            self.requirements = GameDescription[name]["requirements"]
            self.ascii_to_tile = GameDescription[name]["ascii_to_tile"]
        except:
            raise Exception(name + " data not implemented in env.py")

        self.map_level = np.vectorize(lambda x: self.ascii[x])

    def create_levels(self, tensor):
        lvl_array = tensor.argmax(dim=1).cpu().numpy()
        lvls = self.map_level(lvl_array).tolist()
        lvl_strs = ["\n".join(["".join(row) for row in lvl]) for lvl in lvls]
        return lvl_strs

    def pass_requirements(self, lvl_str):
        num_ok = sum(lvl_str.count(i) >= 1 for i in self.requirements)
        return num_ok == 3
