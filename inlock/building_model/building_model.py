from dataclasses import *
import numpy as np
import json
import rustworkx
from typing import List, Dict, Tuple
from enum import Enum
import pyodbc
import copy
import re

class LocationType(Enum):
    ROOM = 1
    CORRIDOR = 2


@dataclass
class NodeData:
    type: LocationType
    label: str
    pos: np.ndarray
    locationID: int

    def to_json(data):
        return {
            "type": data.type.name,
            "label": data.label,
            "pos": list(data.pos),
            "locationID": data.locationID,
        }

    @staticmethod
    def from_json(data):
        return NodeData(
                type= LocationType[data["type"]] if "type" in data else LocationType.ROOM,
                label=data["label"],
                pos=np.array(data["pos"]),
                locationID=int(data["locationID"])
            )

@dataclass
class EdgeData:
    length: float

    def to_json(self):
        return {
            "length": self.length
        }

    @staticmethod
    def from_json(data):
        return EdgeData(**data)

# todo: add floor relation
@dataclass
class Location:
    id: str
    label: str
    type: str
    desc: str
    contour: np.ndarray
    parent: int

    def to_json(self):
        return {
            "id": self.id,
            "label": self.label,
            "desc": self.desc,
            "type": self.type.value,
            "contour": list(list(p) for p in self.contour),
            "parent": self.parent
        }

@dataclass
class Room:
    id: int
    label: str
    type: str
    desc: str
    contour: np.ndarray

    def to_json(self):
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "desc": self.desc,
            "contour": list(list(p) for p in self.contour)
        }

    @staticmethod
    def from_json(json):
        return Room(
            id=json["id"],
            label=json["label"],
            type=json["type"],
            desc=json["desc"],
            contour=np.array(json["contour"])
        )


@dataclass
class Floor:
    z: float # floor
    min: np.ndarray # (3,)
    max: np.ndarray # (3,)
    locations: List[Room]
    outline: np.ndarray # (n,2)
    label: str
    num: int
    walkable_areas: List[np.ndarray] # (m,*,2)

    def to_json(self):
        return {
            "z": self.z,
            "min": list(self.min),
            "max": list(self.max),
            "locations": [room.to_json() for room in self.locations],
            "outline": [x.tolist() for x in self.outline],
            "label": self.label,
            "num": self.num,
            "walkable_areas": [x.tolist() for x in self.walkable_areas]
        }

    @staticmethod
    def from_json(json):
        return Floor(
            z= json["z"],
            min= json["min"],
            max= json["max"],
            locations= [Room.from_json(room) for room in json["locations"]],
            outline= np.array(json["outline"]),
            label=json["label"],
            num=json["num"],
            walkable_areas=[np.array(x) for x in json["walkable_areas"]]
        )


class BuildingModel:
    id: int
    name: str
    path: str
    creator: str
    locations: Dict[int, Room]
    floors: List[Floor]
    graph: rustworkx.PyGraph

    def __init__(self, id: int, name: str, floors: List[Floor], graph: rustworkx.PyGraph, creator: str = ""):
        self.id = id
        self.name = name
        self.floors = floors
        self.graph = graph
        self.creator = creator

        self.locations = {}
        for floor in floors:
            for location in floor.locations:
                self.locations[location.id] = location

    @staticmethod
    def graph_to_json(graph: rustworkx.PyGraph):
        nodes = {id: data.to_json() for id, data in zip(graph.node_indices(), graph.nodes())}

        edges = {}
        for edge, data in enumerate(graph.edges()):
            u, v = graph.get_edge_endpoints_by_index(edge)
            edges[edge] = {
                "u": u,
                "v": v,
                "data": data.to_json(),
            }

        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def graph_from_json(json):
        idx_to_node = {}
        graph = rustworkx.PyGraph()

        names_to_vertex = {}
        for id, data in sorted(json["nodes"].items(), key=lambda x: x[0]):
            id = int(id)
            idx_to_node[id] = graph.add_node(NodeData.from_json(data))
            names_to_vertex[data["label"]] = id

        graph.attrs = {"names_to_vertex": names_to_vertex}

        for id, data in sorted(json["edges"].items(), key=lambda x: x[0]):
            graph.add_edge(idx_to_node[data["u"]], idx_to_node[data["v"]], EdgeData.from_json(data["data"]))

        return graph

    def to_json(self):
        locations = {name: room.to_json() for name,room in self.locations.items()}
        floors = [floor.to_json() for floor in self.floors]
        graph = BuildingModel.graph_to_json(self.graph)

        return {
            "id": self.id,
            "name": self.name,
            "locations": locations,
            "floors": floors,
            "graph": graph
        }

    @staticmethod
    def from_json(json):
        floors = [Floor.from_json(floor) for floor in json["floors"]]
        graph = BuildingModel.graph_from_json(json["graph"])

        return BuildingModel(
            id=json["id"],
            name=json["name"],
            floors=floors,
            graph=graph
        )