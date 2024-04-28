from robosuite.models.objects import MujocoXMLObject

class BishopObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bishop.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class Chess_BoardObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/chess_board.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class KingObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/king.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class KnightObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/knight.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class PawnObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/pawn.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class QueenObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/queen.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class RookObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/rook.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

